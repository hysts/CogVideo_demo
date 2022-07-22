# This code is adapted from https://github.com/THUDM/CogVideo/blob/ff423aa169978fb2f636f761e348631fa3178b03/cogvideo_pipeline.py

from __future__ import annotations

import argparse
import functools
import logging
import pathlib
import sys
import tempfile
import time
from typing import Any

import gradio as gr
import imageio.v2 as iio
import numpy as np
import torch
from icetk import IceTokenizer
from SwissArmyTransformer import get_args
from SwissArmyTransformer.arguments import set_random_seed
from SwissArmyTransformer.generation.sampling_strategies import BaseStrategy
from SwissArmyTransformer.resources import auto_create

app_dir = pathlib.Path(__file__).parent
submodule_dir = app_dir / 'CogVideo'
sys.path.insert(0, submodule_dir.as_posix())

from coglm_strategy import CoglmStrategy
from models.cogvideo_cache_model import CogVideoCacheModel
from sr_pipeline import DirectSuperResolution

formatter = logging.Formatter(
    '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(stream_handler)

ICETK_MODEL_DIR = app_dir / 'icetk_models'


def get_masks_and_position_ids_stage1(data, textlen, framelen):
    # Extract batch size and sequence length.
    tokens = data
    seq_length = len(data[0])
    # Attention mask (lower triangular).
    attention_mask = torch.ones((1, textlen + framelen, textlen + framelen),
                                device=data.device)
    attention_mask[:, :textlen, textlen:] = 0
    attention_mask[:, textlen:, textlen:].tril_()
    attention_mask.unsqueeze_(1)
    # Unaligned version
    position_ids = torch.zeros(seq_length,
                               dtype=torch.long,
                               device=data.device)
    torch.arange(textlen,
                 out=position_ids[:textlen],
                 dtype=torch.long,
                 device=data.device)
    torch.arange(512,
                 512 + seq_length - textlen,
                 out=position_ids[textlen:],
                 dtype=torch.long,
                 device=data.device)
    position_ids = position_ids.unsqueeze(0)

    return tokens, attention_mask, position_ids


def get_masks_and_position_ids_stage2(data, textlen, framelen):
    # Extract batch size and sequence length.
    tokens = data
    seq_length = len(data[0])

    # Attention mask (lower triangular).
    attention_mask = torch.ones((1, textlen + framelen, textlen + framelen),
                                device=data.device)
    attention_mask[:, :textlen, textlen:] = 0
    attention_mask[:, textlen:, textlen:].tril_()
    attention_mask.unsqueeze_(1)

    # Unaligned version
    position_ids = torch.zeros(seq_length,
                               dtype=torch.long,
                               device=data.device)
    torch.arange(textlen,
                 out=position_ids[:textlen],
                 dtype=torch.long,
                 device=data.device)
    frame_num = (seq_length - textlen) // framelen
    assert frame_num == 5
    torch.arange(512,
                 512 + framelen,
                 out=position_ids[textlen:textlen + framelen],
                 dtype=torch.long,
                 device=data.device)
    torch.arange(512 + framelen * 2,
                 512 + framelen * 3,
                 out=position_ids[textlen + framelen:textlen + framelen * 2],
                 dtype=torch.long,
                 device=data.device)
    torch.arange(512 + framelen * (frame_num - 1),
                 512 + framelen * frame_num,
                 out=position_ids[textlen + framelen * 2:textlen +
                                  framelen * 3],
                 dtype=torch.long,
                 device=data.device)
    torch.arange(512 + framelen * 1,
                 512 + framelen * 2,
                 out=position_ids[textlen + framelen * 3:textlen +
                                  framelen * 4],
                 dtype=torch.long,
                 device=data.device)
    torch.arange(512 + framelen * 3,
                 512 + framelen * 4,
                 out=position_ids[textlen + framelen * 4:textlen +
                                  framelen * 5],
                 dtype=torch.long,
                 device=data.device)

    position_ids = position_ids.unsqueeze(0)

    return tokens, attention_mask, position_ids


def my_update_mems(hiddens, mems_buffers, mems_indexs,
                   limited_spatial_channel_mem, text_len, frame_len):
    if hiddens is None:
        return None, mems_indexs
    mem_num = len(hiddens)
    ret_mem = []
    with torch.no_grad():
        for id in range(mem_num):
            if hiddens[id][0] is None:
                ret_mem.append(None)
            else:
                if id == 0 and limited_spatial_channel_mem and mems_indexs[
                        id] + hiddens[0][0].shape[1] >= text_len + frame_len:
                    if mems_indexs[id] == 0:
                        for layer, hidden in enumerate(hiddens[id]):
                            mems_buffers[id][
                                layer, :, :text_len] = hidden.expand(
                                    mems_buffers[id].shape[1], -1,
                                    -1)[:, :text_len]
                    new_mem_len_part2 = (mems_indexs[id] +
                                         hiddens[0][0].shape[1] -
                                         text_len) % frame_len
                    if new_mem_len_part2 > 0:
                        for layer, hidden in enumerate(hiddens[id]):
                            mems_buffers[id][
                                layer, :, text_len:text_len +
                                new_mem_len_part2] = hidden.expand(
                                    mems_buffers[id].shape[1], -1,
                                    -1)[:, -new_mem_len_part2:]
                    mems_indexs[id] = text_len + new_mem_len_part2
                else:
                    for layer, hidden in enumerate(hiddens[id]):
                        mems_buffers[id][layer, :,
                                         mems_indexs[id]:mems_indexs[id] +
                                         hidden.shape[1]] = hidden.expand(
                                             mems_buffers[id].shape[1], -1, -1)
                    mems_indexs[id] += hidden.shape[1]
                ret_mem.append(mems_buffers[id][:, :, :mems_indexs[id]])
    return ret_mem, mems_indexs


def calc_next_tokens_frame_begin_id(text_len, frame_len, total_len):
    # The fisrt token's position id of the frame that the next token belongs to;
    if total_len < text_len:
        return None
    return (total_len - text_len) // frame_len * frame_len + text_len


def my_filling_sequence(
        model,
        tokenizer,
        args,
        seq,
        batch_size,
        get_masks_and_position_ids,
        text_len,
        frame_len,
        strategy=BaseStrategy(),
        strategy2=BaseStrategy(),
        mems=None,
        log_text_attention_weights=0,  # default to 0: no artificial change
        mode_stage1=True,
        enforce_no_swin=False,
        guider_seq=None,
        guider_text_len=0,
        guidance_alpha=1,
        limited_spatial_channel_mem=False,  # 空间通道的存储限制在本帧内
        **kw_args):
    '''
        seq: [2, 3, 5, ..., -1(to be generated), -1, ...]
        mems: [num_layers, batch_size, len_mems(index), mem_hidden_size]
            cache, should be first mems.shape[1] parts of context_tokens.
            mems are the first-level citizens here, but we don't assume what is memorized.
            input mems are used when multi-phase generation.
    '''
    if guider_seq is not None:
        logger.debug('Using Guidance In Inference')
    if limited_spatial_channel_mem:
        logger.debug("Limit spatial-channel's mem to current frame")
    assert len(seq.shape) == 2

    # building the initial tokens, attention_mask, and position_ids
    actual_context_length = 0

    while seq[-1][
            actual_context_length] >= 0:  # the last seq has least given tokens
        actual_context_length += 1  # [0, context_length-1] are given
    assert actual_context_length > 0
    current_frame_num = (actual_context_length - text_len) // frame_len
    assert current_frame_num >= 0
    context_length = text_len + current_frame_num * frame_len

    tokens, attention_mask, position_ids = get_masks_and_position_ids(
        seq, text_len, frame_len)
    tokens = tokens[..., :context_length]
    input_tokens = tokens.clone()

    if guider_seq is not None:
        guider_index_delta = text_len - guider_text_len
        guider_tokens, guider_attention_mask, guider_position_ids = get_masks_and_position_ids(
            guider_seq, guider_text_len, frame_len)
        guider_tokens = guider_tokens[..., :context_length -
                                      guider_index_delta]
        guider_input_tokens = guider_tokens.clone()

    for fid in range(current_frame_num):
        input_tokens[:, text_len + 400 * fid] = tokenizer['<start_of_image>']
        if guider_seq is not None:
            guider_input_tokens[:, guider_text_len +
                                400 * fid] = tokenizer['<start_of_image>']

    attention_mask = attention_mask.type_as(next(
        model.parameters()))  # if fp16
    # initialize generation
    counter = context_length - 1  # Last fixed index is ``counter''
    index = 0  # Next forward starting index, also the length of cache.
    mems_buffers_on_GPU = False
    mems_indexs = [0, 0]
    mems_len = [(400 + 74) if limited_spatial_channel_mem else 5 * 400 + 74,
                5 * 400 + 74]
    mems_buffers = [
        torch.zeros(args.num_layers,
                    batch_size,
                    mem_len,
                    args.hidden_size * 2,
                    dtype=next(model.parameters()).dtype)
        for mem_len in mems_len
    ]

    if guider_seq is not None:
        guider_attention_mask = guider_attention_mask.type_as(
            next(model.parameters()))  # if fp16
        guider_mems_buffers = [
            torch.zeros(args.num_layers,
                        batch_size,
                        mem_len,
                        args.hidden_size * 2,
                        dtype=next(model.parameters()).dtype)
            for mem_len in mems_len
        ]
        guider_mems_indexs = [0, 0]
        guider_mems = None

    torch.cuda.empty_cache()
    # step-by-step generation
    while counter < len(seq[0]) - 1:
        # we have generated counter+1 tokens
        # Now, we want to generate seq[counter + 1],
        # token[:, index: counter+1] needs forwarding.
        if index == 0:
            group_size = 2 if (input_tokens.shape[0] == batch_size
                               and not mode_stage1) else batch_size

            logits_all = None
            for batch_idx in range(0, input_tokens.shape[0], group_size):
                logits, *output_per_layers = model(
                    input_tokens[batch_idx:batch_idx + group_size, index:],
                    position_ids[..., index:counter + 1],
                    attention_mask,  # TODO memlen
                    mems=mems,
                    text_len=text_len,
                    frame_len=frame_len,
                    counter=counter,
                    log_text_attention_weights=log_text_attention_weights,
                    enforce_no_swin=enforce_no_swin,
                    **kw_args)
                logits_all = torch.cat(
                    (logits_all,
                     logits), dim=0) if logits_all is not None else logits
                mem_kv01 = [[o['mem_kv'][0] for o in output_per_layers],
                            [o['mem_kv'][1] for o in output_per_layers]]
                next_tokens_frame_begin_id = calc_next_tokens_frame_begin_id(
                    text_len, frame_len, mem_kv01[0][0].shape[1])
                for id, mem_kv in enumerate(mem_kv01):
                    for layer, mem_kv_perlayer in enumerate(mem_kv):
                        if limited_spatial_channel_mem and id == 0:
                            mems_buffers[id][
                                layer, batch_idx:batch_idx + group_size, :
                                text_len] = mem_kv_perlayer.expand(
                                    min(group_size,
                                        input_tokens.shape[0] - batch_idx), -1,
                                    -1)[:, :text_len]
                            mems_buffers[id][layer, batch_idx:batch_idx+group_size, text_len:text_len+mem_kv_perlayer.shape[1]-next_tokens_frame_begin_id] =\
                                mem_kv_perlayer.expand(min(group_size, input_tokens.shape[0]-batch_idx), -1, -1)[:, next_tokens_frame_begin_id:]
                        else:
                            mems_buffers[id][
                                layer, batch_idx:batch_idx +
                                group_size, :mem_kv_perlayer.
                                shape[1]] = mem_kv_perlayer.expand(
                                    min(group_size,
                                        input_tokens.shape[0] - batch_idx), -1,
                                    -1)
                mems_indexs[0], mems_indexs[1] = mem_kv01[0][0].shape[
                    1], mem_kv01[1][0].shape[1]
                if limited_spatial_channel_mem:
                    mems_indexs[0] -= (next_tokens_frame_begin_id - text_len)

            mems = [
                mems_buffers[id][:, :, :mems_indexs[id]] for id in range(2)
            ]
            logits = logits_all

            # Guider
            if guider_seq is not None:
                guider_logits_all = None
                for batch_idx in range(0, guider_input_tokens.shape[0],
                                       group_size):
                    guider_logits, *guider_output_per_layers = model(
                        guider_input_tokens[batch_idx:batch_idx + group_size,
                                            max(index -
                                                guider_index_delta, 0):],
                        guider_position_ids[
                            ...,
                            max(index - guider_index_delta, 0):counter + 1 -
                            guider_index_delta],
                        guider_attention_mask,
                        mems=guider_mems,
                        text_len=guider_text_len,
                        frame_len=frame_len,
                        counter=counter - guider_index_delta,
                        log_text_attention_weights=log_text_attention_weights,
                        enforce_no_swin=enforce_no_swin,
                        **kw_args)
                    guider_logits_all = torch.cat(
                        (guider_logits_all, guider_logits), dim=0
                    ) if guider_logits_all is not None else guider_logits
                    guider_mem_kv01 = [[
                        o['mem_kv'][0] for o in guider_output_per_layers
                    ], [o['mem_kv'][1] for o in guider_output_per_layers]]
                    for id, guider_mem_kv in enumerate(guider_mem_kv01):
                        for layer, guider_mem_kv_perlayer in enumerate(
                                guider_mem_kv):
                            if limited_spatial_channel_mem and id == 0:
                                guider_mems_buffers[id][
                                    layer, batch_idx:batch_idx + group_size, :
                                    guider_text_len] = guider_mem_kv_perlayer.expand(
                                        min(group_size,
                                            input_tokens.shape[0] - batch_idx),
                                        -1, -1)[:, :guider_text_len]
                                guider_next_tokens_frame_begin_id = calc_next_tokens_frame_begin_id(
                                    guider_text_len, frame_len,
                                    guider_mem_kv_perlayer.shape[1])
                                guider_mems_buffers[id][layer, batch_idx:batch_idx+group_size, guider_text_len:guider_text_len+guider_mem_kv_perlayer.shape[1]-guider_next_tokens_frame_begin_id] =\
                                    guider_mem_kv_perlayer.expand(min(group_size, input_tokens.shape[0]-batch_idx), -1, -1)[:, guider_next_tokens_frame_begin_id:]
                            else:
                                guider_mems_buffers[id][
                                    layer, batch_idx:batch_idx +
                                    group_size, :guider_mem_kv_perlayer.
                                    shape[1]] = guider_mem_kv_perlayer.expand(
                                        min(group_size,
                                            input_tokens.shape[0] - batch_idx),
                                        -1, -1)
                    guider_mems_indexs[0], guider_mems_indexs[
                        1] = guider_mem_kv01[0][0].shape[1], guider_mem_kv01[
                            1][0].shape[1]
                    if limited_spatial_channel_mem:
                        guider_mems_indexs[0] -= (
                            guider_next_tokens_frame_begin_id -
                            guider_text_len)
                guider_mems = [
                    guider_mems_buffers[id][:, :, :guider_mems_indexs[id]]
                    for id in range(2)
                ]
                guider_logits = guider_logits_all
        else:
            if not mems_buffers_on_GPU:
                if not mode_stage1:
                    torch.cuda.empty_cache()
                    for idx, mem in enumerate(mems):
                        mems[idx] = mem.to(next(model.parameters()).device)
                    if guider_seq is not None:
                        for idx, mem in enumerate(guider_mems):
                            guider_mems[idx] = mem.to(
                                next(model.parameters()).device)
                else:
                    torch.cuda.empty_cache()
                    for idx, mem_buffer in enumerate(mems_buffers):
                        mems_buffers[idx] = mem_buffer.to(
                            next(model.parameters()).device)
                    mems = [
                        mems_buffers[id][:, :, :mems_indexs[id]]
                        for id in range(2)
                    ]
                    if guider_seq is not None:
                        for idx, guider_mem_buffer in enumerate(
                                guider_mems_buffers):
                            guider_mems_buffers[idx] = guider_mem_buffer.to(
                                next(model.parameters()).device)
                        guider_mems = [
                            guider_mems_buffers[id]
                            [:, :, :guider_mems_indexs[id]] for id in range(2)
                        ]
                    mems_buffers_on_GPU = True

            logits, *output_per_layers = model(
                input_tokens[:, index:],
                position_ids[..., index:counter + 1],
                attention_mask,  # TODO memlen
                mems=mems,
                text_len=text_len,
                frame_len=frame_len,
                counter=counter,
                log_text_attention_weights=log_text_attention_weights,
                enforce_no_swin=enforce_no_swin,
                limited_spatial_channel_mem=limited_spatial_channel_mem,
                **kw_args)
            mem_kv0, mem_kv1 = [o['mem_kv'][0] for o in output_per_layers
                                ], [o['mem_kv'][1] for o in output_per_layers]

            if guider_seq is not None:
                guider_logits, *guider_output_per_layers = model(
                    guider_input_tokens[:,
                                        max(index - guider_index_delta, 0):],
                    guider_position_ids[...,
                                        max(index -
                                            guider_index_delta, 0):counter +
                                        1 - guider_index_delta],
                    guider_attention_mask,
                    mems=guider_mems,
                    text_len=guider_text_len,
                    frame_len=frame_len,
                    counter=counter - guider_index_delta,
                    log_text_attention_weights=0,
                    enforce_no_swin=enforce_no_swin,
                    limited_spatial_channel_mem=limited_spatial_channel_mem,
                    **kw_args)
                guider_mem_kv0, guider_mem_kv1 = [
                    o['mem_kv'][0] for o in guider_output_per_layers
                ], [o['mem_kv'][1] for o in guider_output_per_layers]

            if not mems_buffers_on_GPU:
                torch.cuda.empty_cache()
                for idx, mem_buffer in enumerate(mems_buffers):
                    mems_buffers[idx] = mem_buffer.to(
                        next(model.parameters()).device)
                if guider_seq is not None:
                    for idx, guider_mem_buffer in enumerate(
                            guider_mems_buffers):
                        guider_mems_buffers[idx] = guider_mem_buffer.to(
                            next(model.parameters()).device)
                mems_buffers_on_GPU = True

            mems, mems_indexs = my_update_mems([mem_kv0, mem_kv1],
                                               mems_buffers, mems_indexs,
                                               limited_spatial_channel_mem,
                                               text_len, frame_len)
            if guider_seq is not None:
                guider_mems, guider_mems_indexs = my_update_mems(
                    [guider_mem_kv0, guider_mem_kv1], guider_mems_buffers,
                    guider_mems_indexs, limited_spatial_channel_mem,
                    guider_text_len, frame_len)

        counter += 1
        index = counter

        logits = logits[:, -1].expand(batch_size,
                                      -1)  # [batch size, vocab size]
        tokens = tokens.expand(batch_size, -1)
        if guider_seq is not None:
            guider_logits = guider_logits[:, -1].expand(batch_size, -1)
            guider_tokens = guider_tokens.expand(batch_size, -1)

        if seq[-1][counter].item() < 0:
            # sampling
            guided_logits = guider_logits + (
                logits - guider_logits
            ) * guidance_alpha if guider_seq is not None else logits
            if mode_stage1 and counter < text_len + 400:
                tokens, mems = strategy.forward(guided_logits, tokens, mems)
            else:
                tokens, mems = strategy2.forward(guided_logits, tokens, mems)
            if guider_seq is not None:
                guider_tokens = torch.cat((guider_tokens, tokens[:, -1:]),
                                          dim=1)

            if seq[0][counter].item() >= 0:
                for si in range(seq.shape[0]):
                    if seq[si][counter].item() >= 0:
                        tokens[si, -1] = seq[si, counter]
                        if guider_seq is not None:
                            guider_tokens[si,
                                          -1] = guider_seq[si, counter -
                                                           guider_index_delta]

        else:
            tokens = torch.cat(
                (tokens, seq[:, counter:counter + 1].clone().expand(
                    tokens.shape[0], 1).to(device=tokens.device,
                                           dtype=tokens.dtype)),
                dim=1)
            if guider_seq is not None:
                guider_tokens = torch.cat(
                    (guider_tokens,
                     guider_seq[:, counter - guider_index_delta:counter + 1 -
                                guider_index_delta].clone().expand(
                                    guider_tokens.shape[0], 1).to(
                                        device=guider_tokens.device,
                                        dtype=guider_tokens.dtype)),
                    dim=1)

        input_tokens = tokens.clone()
        if guider_seq is not None:
            guider_input_tokens = guider_tokens.clone()
        if (index - text_len - 1) // 400 < (input_tokens.shape[-1] - text_len -
                                            1) // 400:
            boi_idx = ((index - text_len - 1) // 400 + 1) * 400 + text_len
            while boi_idx < input_tokens.shape[-1]:
                input_tokens[:, boi_idx] = tokenizer['<start_of_image>']
                if guider_seq is not None:
                    guider_input_tokens[:, boi_idx -
                                        guider_index_delta] = tokenizer[
                                            '<start_of_image>']
                boi_idx += 400

        if strategy.is_done:
            break
    return strategy.finalize(tokens, mems)


class InferenceModel_Sequential(CogVideoCacheModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args,
                         transformer=transformer,
                         parallel_output=parallel_output,
                         window_size=-1,
                         cogvideo_stage=1)

    # TODO: check it

    def final_forward(self, logits, **kwargs):
        logits_parallel = logits
        logits_parallel = torch.nn.functional.linear(
            logits_parallel.float(),
            self.transformer.word_embeddings.weight[:20000].float())
        return logits_parallel


class InferenceModel_Interpolate(CogVideoCacheModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args,
                         transformer=transformer,
                         parallel_output=parallel_output,
                         window_size=10,
                         cogvideo_stage=2)

    # TODO: check it

    def final_forward(self, logits, **kwargs):
        logits_parallel = logits
        logits_parallel = torch.nn.functional.linear(
            logits_parallel.float(),
            self.transformer.word_embeddings.weight[:20000].float())
        return logits_parallel


def get_default_args() -> argparse.Namespace:
    known = argparse.Namespace(generate_frame_num=5,
                               coglm_temperature2=0.89,
                               use_guidance_stage1=True,
                               use_guidance_stage2=False,
                               guidance_alpha=3.0,
                               stage_1=True,
                               stage_2=False,
                               both_stages=False,
                               parallel_size=1,
                               stage1_max_inference_batch_size=-1,
                               multi_gpu=False,
                               layout='64, 464, 2064',
                               window_size=10,
                               additional_seqlen=2000,
                               cogvideo_stage=1)

    args_list = [
        '--tokenizer-type',
        'fake',
        '--mode',
        'inference',
        '--distributed-backend',
        'nccl',
        '--fp16',
        '--model-parallel-size',
        '1',
        '--temperature',
        '1.05',
        '--top_k',
        '12',
        '--sandwich-ln',
        '--seed',
        '1234',
        '--num-workers',
        '0',
        '--batch-size',
        '1',
        '--max-inference-batch-size',
        '8',
    ]
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.layout = [int(x) for x in args.layout.split(',')]
    args.do_train = False
    return args


class Model:
    def __init__(self, only_first_stage: bool = False):
        self.args = get_default_args()
        if only_first_stage:
            self.args.stage_1 = True
            self.args.both_stages = False
        else:
            self.args.stage_1 = False
            self.args.both_stages = True

        self.tokenizer = self.load_tokenizer()

        self.model_stage1, self.args = self.load_model_stage1()
        self.model_stage2, self.args = self.load_model_stage2()

        self.strategy_cogview2, self.strategy_cogvideo = self.load_strategies()
        self.dsr = self.load_dsr()

        self.device = torch.device(self.args.device)

    def load_tokenizer(self) -> IceTokenizer:
        logger.info('--- load_tokenizer ---')
        start = time.perf_counter()

        tokenizer = IceTokenizer(ICETK_MODEL_DIR.as_posix())
        tokenizer.add_special_tokens(
            ['<start_of_image>', '<start_of_english>', '<start_of_chinese>'])

        elapsed = time.perf_counter() - start
        logger.info(f'--- done ({elapsed=:.3f}) ---')
        return tokenizer

    def load_model_stage1(
            self) -> tuple[CogVideoCacheModel, argparse.Namespace]:
        logger.info('--- load_model_stage1 ---')
        start = time.perf_counter()

        args = self.args
        model_stage1, args = InferenceModel_Sequential.from_pretrained(
            args, 'cogvideo-stage1')
        model_stage1.eval()
        if args.both_stages:
            model_stage1 = model_stage1.cpu()

        elapsed = time.perf_counter() - start
        logger.info(f'--- done ({elapsed=:.3f}) ---')
        return model_stage1, args

    def load_model_stage2(
            self) -> tuple[CogVideoCacheModel | None, argparse.Namespace]:
        logger.info('--- load_model_stage2 ---')
        start = time.perf_counter()

        args = self.args
        if args.both_stages:
            model_stage2, args = InferenceModel_Interpolate.from_pretrained(
                args, 'cogvideo-stage2')
            model_stage2.eval()
            if args.both_stages:
                model_stage2 = model_stage2.cpu()
        else:
            model_stage2 = None

        elapsed = time.perf_counter() - start
        logger.info(f'--- done ({elapsed=:.3f}) ---')
        return model_stage2, args

    def load_strategies(self) -> tuple[CoglmStrategy, CoglmStrategy]:
        logger.info('--- load_strategies ---')
        start = time.perf_counter()

        invalid_slices = [slice(self.tokenizer.num_image_tokens, None)]
        strategy_cogview2 = CoglmStrategy(invalid_slices,
                                          temperature=1.0,
                                          top_k=16)
        strategy_cogvideo = CoglmStrategy(
            invalid_slices,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            temperature2=self.args.coglm_temperature2)

        elapsed = time.perf_counter() - start
        logger.info(f'--- done ({elapsed=:.3f}) ---')
        return strategy_cogview2, strategy_cogvideo

    def load_dsr(self) -> DirectSuperResolution | None:
        logger.info('--- load_dsr ---')
        start = time.perf_counter()

        if self.args.both_stages:
            path = auto_create('cogview2-dsr', path=None)
            dsr = DirectSuperResolution(self.args,
                                        path,
                                        max_bz=12,
                                        onCUDA=False)
        else:
            dsr = None

        elapsed = time.perf_counter() - start
        logger.info(f'--- done ({elapsed=:.3f}) ---')
        return dsr

    @torch.inference_mode()
    def process_stage1(self,
                       model,
                       seq_text,
                       duration,
                       video_raw_text=None,
                       video_guidance_text='视频',
                       image_text_suffix='',
                       batch_size=1):
        process_start_time = time.perf_counter()

        generate_frame_num = self.args.generate_frame_num
        tokenizer = self.tokenizer
        use_guide = self.args.use_guidance_stage1

        if next(model.parameters()).device != self.device:
            move_start_time = time.perf_counter()
            logger.debug('moving stage 1 model to cuda')

            model = model.to(self.device)

            elapsed = time.perf_counter() - move_start_time
            logger.debug(f'moving in model1 takes time: {elapsed:.2f}')

        if video_raw_text is None:
            video_raw_text = seq_text
        mbz = self.args.stage1_max_inference_batch_size if self.args.stage1_max_inference_batch_size > 0 else self.args.max_inference_batch_size
        assert batch_size < mbz or batch_size % mbz == 0
        frame_len = 400

        # generate the first frame:
        enc_text = tokenizer.encode(seq_text + image_text_suffix)
        seq_1st = enc_text + [tokenizer['<start_of_image>']] + [-1] * 400
        logger.info(
            f'[Generating First Frame with CogView2] Raw text: {tokenizer.decode(enc_text):s}'
        )
        text_len_1st = len(seq_1st) - frame_len * 1 - 1

        seq_1st = torch.tensor(seq_1st, dtype=torch.long,
                               device=self.device).unsqueeze(0)
        output_list_1st = []
        for tim in range(max(batch_size // mbz, 1)):
            start_time = time.perf_counter()
            output_list_1st.append(
                my_filling_sequence(
                    model,
                    tokenizer,
                    self.args,
                    seq_1st.clone(),
                    batch_size=min(batch_size, mbz),
                    get_masks_and_position_ids=
                    get_masks_and_position_ids_stage1,
                    text_len=text_len_1st,
                    frame_len=frame_len,
                    strategy=self.strategy_cogview2,
                    strategy2=self.strategy_cogvideo,
                    log_text_attention_weights=1.4,
                    enforce_no_swin=True,
                    mode_stage1=True,
                )[0])
            elapsed = time.perf_counter() - start_time
            logger.info(f'[First Frame] Elapsed: {elapsed:.2f}')
        output_tokens_1st = torch.cat(output_list_1st, dim=0)
        given_tokens = output_tokens_1st[:, text_len_1st + 1:text_len_1st +
                                         401].unsqueeze(
                                             1
                                         )  # given_tokens.shape: [bs, frame_num, 400]

        # generate subsequent frames:
        total_frames = generate_frame_num
        enc_duration = tokenizer.encode(f'{float(duration)}秒')
        if use_guide:
            video_raw_text = video_raw_text + ' 视频'
        enc_text_video = tokenizer.encode(video_raw_text)
        seq = enc_duration + [tokenizer['<n>']] + enc_text_video + [
            tokenizer['<start_of_image>']
        ] + [-1] * 400 * generate_frame_num
        guider_seq = enc_duration + [tokenizer['<n>']] + tokenizer.encode(
            video_guidance_text) + [tokenizer['<start_of_image>']
                                    ] + [-1] * 400 * generate_frame_num
        logger.info(
            f'[Stage1: Generating Subsequent Frames, Frame Rate {4/duration:.1f}] raw text: {tokenizer.decode(enc_text_video):s}'
        )

        text_len = len(seq) - frame_len * generate_frame_num - 1
        guider_text_len = len(guider_seq) - frame_len * generate_frame_num - 1
        seq = torch.tensor(seq, dtype=torch.long,
                           device=self.device).unsqueeze(0).repeat(
                               batch_size, 1)
        guider_seq = torch.tensor(guider_seq,
                                  dtype=torch.long,
                                  device=self.device).unsqueeze(0).repeat(
                                      batch_size, 1)

        for given_frame_id in range(given_tokens.shape[1]):
            seq[:, text_len + 1 + given_frame_id * 400:text_len + 1 +
                (given_frame_id + 1) * 400] = given_tokens[:, given_frame_id]
            guider_seq[:, guider_text_len + 1 +
                       given_frame_id * 400:guider_text_len + 1 +
                       (given_frame_id + 1) *
                       400] = given_tokens[:, given_frame_id]
        output_list = []

        if use_guide:
            video_log_text_attention_weights = 0
        else:
            guider_seq = None
            video_log_text_attention_weights = 1.4

        for tim in range(max(batch_size // mbz, 1)):
            input_seq = seq[:min(batch_size, mbz)].clone(
            ) if tim == 0 else seq[mbz * tim:mbz * (tim + 1)].clone()
            guider_seq2 = (guider_seq[:min(batch_size, mbz)].clone()
                           if tim == 0 else guider_seq[mbz * tim:mbz *
                                                       (tim + 1)].clone()
                           ) if guider_seq is not None else None
            output_list.append(
                my_filling_sequence(
                    model,
                    tokenizer,
                    self.args,
                    input_seq,
                    batch_size=min(batch_size, mbz),
                    get_masks_and_position_ids=
                    get_masks_and_position_ids_stage1,
                    text_len=text_len,
                    frame_len=frame_len,
                    strategy=self.strategy_cogview2,
                    strategy2=self.strategy_cogvideo,
                    log_text_attention_weights=video_log_text_attention_weights,
                    guider_seq=guider_seq2,
                    guider_text_len=guider_text_len,
                    guidance_alpha=self.args.guidance_alpha,
                    limited_spatial_channel_mem=True,
                    mode_stage1=True,
                )[0])

        output_tokens = torch.cat(output_list, dim=0)[:, 1 + text_len:]

        if self.args.both_stages:
            move_start_time = time.perf_counter()
            logger.debug('moving stage 1 model to cpu')
            model = model.cpu()
            torch.cuda.empty_cache()
            elapsed = time.perf_counter() - move_start_time
            logger.debug(f'moving in model1 takes time: {elapsed:.2f}')

        # decoding
        res = []
        for seq in output_tokens:
            decoded_imgs = [
                self.postprocess(
                    torch.nn.functional.interpolate(tokenizer.decode(
                        image_ids=seq.tolist()[i * 400:(i + 1) * 400]),
                                                    size=(480, 480))[0])
                for i in range(total_frames)
            ]
            res.append(decoded_imgs)  # only the last image (target)

        assert len(res) == batch_size
        tokens = output_tokens[:, :+total_frames * 400].reshape(
            -1, total_frames, 400).cpu()

        elapsed = time.perf_counter() - process_start_time
        logger.info(f'--- done ({elapsed=:.3f}) ---')
        return tokens, res[0]

    @torch.inference_mode()
    def process_stage2(self,
                       model,
                       seq_text,
                       duration,
                       parent_given_tokens,
                       video_raw_text=None,
                       video_guidance_text='视频',
                       gpu_rank=0,
                       gpu_parallel_size=1):
        process_start_time = time.perf_counter()

        generate_frame_num = self.args.generate_frame_num
        tokenizer = self.tokenizer
        use_guidance = self.args.use_guidance_stage2

        stage2_start_time = time.perf_counter()

        if next(model.parameters()).device != self.device:
            move_start_time = time.perf_counter()
            logger.debug('moving stage-2 model to cuda')

            model = model.to(self.device)

            elapsed = time.perf_counter() - move_start_time
            logger.debug(f'moving in stage-2 model takes time: {elapsed:.2f}')

        try:
            sample_num_allgpu = parent_given_tokens.shape[0]
            sample_num = sample_num_allgpu // gpu_parallel_size
            assert sample_num * gpu_parallel_size == sample_num_allgpu
            parent_given_tokens = parent_given_tokens[gpu_rank *
                                                      sample_num:(gpu_rank +
                                                                  1) *
                                                      sample_num]
        except:
            logger.critical('No frame_tokens found in interpolation, skip')
            return False, []

        # CogVideo Stage2 Generation
        while duration >= 0.5:  # TODO: You can change the boundary to change the frame rate
            parent_given_tokens_num = parent_given_tokens.shape[1]
            generate_batchsize_persample = (parent_given_tokens_num - 1) // 2
            generate_batchsize_total = generate_batchsize_persample * sample_num
            total_frames = generate_frame_num
            frame_len = 400
            enc_text = tokenizer.encode(seq_text)
            enc_duration = tokenizer.encode(str(float(duration)) + '秒')
            seq = enc_duration + [tokenizer['<n>']] + enc_text + [
                tokenizer['<start_of_image>']
            ] + [-1] * 400 * generate_frame_num
            text_len = len(seq) - frame_len * generate_frame_num - 1

            logger.info(
                f'[Stage2: Generating Frames, Frame Rate {int(4/duration):d}] raw text: {tokenizer.decode(enc_text):s}'
            )

            # generation
            seq = torch.tensor(seq, dtype=torch.long,
                               device=self.device).unsqueeze(0).repeat(
                                   generate_batchsize_total, 1)
            for sample_i in range(sample_num):
                for i in range(generate_batchsize_persample):
                    seq[sample_i * generate_batchsize_persample +
                        i][text_len + 1:text_len + 1 +
                           400] = parent_given_tokens[sample_i][2 * i]
                    seq[sample_i * generate_batchsize_persample +
                        i][text_len + 1 + 400:text_len + 1 +
                           800] = parent_given_tokens[sample_i][2 * i + 1]
                    seq[sample_i * generate_batchsize_persample +
                        i][text_len + 1 + 800:text_len + 1 +
                           1200] = parent_given_tokens[sample_i][2 * i + 2]

            if use_guidance:
                guider_seq = enc_duration + [
                    tokenizer['<n>']
                ] + tokenizer.encode(video_guidance_text) + [
                    tokenizer['<start_of_image>']
                ] + [-1] * 400 * generate_frame_num
                guider_text_len = len(
                    guider_seq) - frame_len * generate_frame_num - 1
                guider_seq = torch.tensor(
                    guider_seq, dtype=torch.long,
                    device=self.device).unsqueeze(0).repeat(
                        generate_batchsize_total, 1)
                for sample_i in range(sample_num):
                    for i in range(generate_batchsize_persample):
                        guider_seq[sample_i * generate_batchsize_persample +
                                   i][text_len + 1:text_len + 1 +
                                      400] = parent_given_tokens[sample_i][2 *
                                                                           i]
                        guider_seq[sample_i * generate_batchsize_persample +
                                   i][text_len + 1 + 400:text_len + 1 +
                                      800] = parent_given_tokens[sample_i][2 *
                                                                           i +
                                                                           1]
                        guider_seq[sample_i * generate_batchsize_persample +
                                   i][text_len + 1 + 800:text_len + 1 +
                                      1200] = parent_given_tokens[sample_i][2 *
                                                                            i +
                                                                            2]
                video_log_text_attention_weights = 0
            else:
                guider_seq = None
                guider_text_len = 0
                video_log_text_attention_weights = 1.4

            mbz = self.args.max_inference_batch_size

            assert generate_batchsize_total < mbz or generate_batchsize_total % mbz == 0
            output_list = []
            start_time = time.perf_counter()
            for tim in range(max(generate_batchsize_total // mbz, 1)):
                input_seq = seq[:min(generate_batchsize_total, mbz)].clone(
                ) if tim == 0 else seq[mbz * tim:mbz * (tim + 1)].clone()
                guider_seq2 = (
                    guider_seq[:min(generate_batchsize_total, mbz)].clone()
                    if tim == 0 else guider_seq[mbz * tim:mbz *
                                                (tim + 1)].clone()
                ) if guider_seq is not None else None
                output_list.append(
                    my_filling_sequence(
                        model,
                        tokenizer,
                        self.args,
                        input_seq,
                        batch_size=min(generate_batchsize_total, mbz),
                        get_masks_and_position_ids=
                        get_masks_and_position_ids_stage2,
                        text_len=text_len,
                        frame_len=frame_len,
                        strategy=self.strategy_cogview2,
                        strategy2=self.strategy_cogvideo,
                        log_text_attention_weights=
                        video_log_text_attention_weights,
                        mode_stage1=False,
                        guider_seq=guider_seq2,
                        guider_text_len=guider_text_len,
                        guidance_alpha=self.args.guidance_alpha,
                        limited_spatial_channel_mem=True,
                    )[0])
            elapsed = time.perf_counter() - start_time
            logger.info(f'Duration {duration:.2f}, Elapsed: {elapsed:.2f}\n')

            output_tokens = torch.cat(output_list, dim=0)
            output_tokens = output_tokens[:, text_len + 1:text_len + 1 +
                                          (total_frames) * 400].reshape(
                                              sample_num, -1,
                                              400 * total_frames)
            output_tokens_merge = torch.cat(
                (output_tokens[:, :, :1 * 400], output_tokens[:, :,
                                                              400 * 3:4 * 400],
                 output_tokens[:, :, 400 * 1:2 * 400],
                 output_tokens[:, :, 400 * 4:(total_frames) * 400]),
                dim=2).reshape(sample_num, -1, 400)

            output_tokens_merge = torch.cat(
                (output_tokens_merge, output_tokens[:, -1:, 400 * 2:3 * 400]),
                dim=1)
            duration /= 2
            parent_given_tokens = output_tokens_merge

        if self.args.both_stages:
            move_start_time = time.perf_counter()
            logger.debug('moving stage 2 model to cpu')
            model = model.cpu()
            torch.cuda.empty_cache()
            elapsed = time.perf_counter() - move_start_time
            logger.debug(f'moving out model2 takes time: {elapsed:.2f}')

        elapsed = time.perf_counter() - stage2_start_time
        logger.info(f'CogVideo Stage2 completed. Elapsed: {elapsed:.2f}\n')

        # direct super-resolution by CogView2
        logger.info('[Direct super-resolution]')
        dsr_start_time = time.perf_counter()

        enc_text = tokenizer.encode(seq_text)
        frame_num_per_sample = parent_given_tokens.shape[1]
        parent_given_tokens_2d = parent_given_tokens.reshape(-1, 400)
        text_seq = torch.tensor(enc_text, dtype=torch.long,
                                device=self.device).unsqueeze(0).repeat(
                                    parent_given_tokens_2d.shape[0], 1)
        sred_tokens = self.dsr(text_seq, parent_given_tokens_2d)

        decoded_sr_videos = []
        for sample_i in range(sample_num):
            decoded_sr_imgs = []
            for frame_i in range(frame_num_per_sample):
                decoded_sr_img = tokenizer.decode(
                    image_ids=sred_tokens[frame_i + sample_i *
                                          frame_num_per_sample][-3600:])
                decoded_sr_imgs.append(
                    self.postprocess(
                        torch.nn.functional.interpolate(decoded_sr_img,
                                                        size=(480, 480))[0]))
            decoded_sr_videos.append(decoded_sr_imgs)

        elapsed = time.perf_counter() - dsr_start_time
        logger.info(
            f'Direct super-resolution completed. Elapsed: {elapsed:.2f}')

        elapsed = time.perf_counter() - process_start_time
        logger.info(f'--- done ({elapsed=:.3f}) ---')
        return True, decoded_sr_videos[0]

    @staticmethod
    def postprocess(tensor: torch.Tensor) -> np.ndarray:
        return tensor.cpu().mul(255).add_(0.5).clamp_(0, 255).permute(
            1, 2, 0).to(torch.uint8).numpy()

    def run(self, text: str, seed: int,
            only_first_stage: bool) -> list[np.ndarray]:
        logger.info('==================== run ====================')
        start = time.perf_counter()

        set_random_seed(seed)
        self.args.seed = seed

        if only_first_stage:
            self.args.stage_1 = True
            self.args.both_stages = False
        else:
            self.args.stage_1 = False
            self.args.both_stages = True

        parent_given_tokens, res = self.process_stage1(
            self.model_stage1,
            text,
            duration=4.0,
            video_raw_text=text,
            video_guidance_text='视频',
            image_text_suffix=' 高清摄影',
            batch_size=self.args.batch_size)
        if not only_first_stage:
            _, res = self.process_stage2(
                self.model_stage2,
                text,
                duration=2.0,
                parent_given_tokens=parent_given_tokens,
                video_raw_text=text + ' 视频',
                video_guidance_text='视频',
                gpu_rank=0,
                gpu_parallel_size=1)  # TODO: 修改

        elapsed = time.perf_counter() - start
        logger.info(f'Elapsed: {elapsed:.3f}')
        logger.info('==================== done ====================')
        return res


class AppModel(Model):
    def __init__(self, only_first_stage: bool):
        super().__init__(only_first_stage)
        self.translator = gr.Interface.load(
            'spaces/chinhon/translation_eng2ch')

    def to_video(self, frames: list[np.ndarray]) -> str:
        out_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        if self.args.stage_1:
            fps = 4
        else:
            fps = 8
        writer = iio.get_writer(out_file.name, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        return out_file.name

    def run_with_translation(
        self, text: str, translate: bool, seed: int, only_first_stage: bool
    ) -> tuple[str | None, str | None, list[np.ndarray] | None]:
        logger.info(f'{text=}, {translate=}, {seed=}, {only_first_stage=}')
        if translate:
            text = translated_text = self.translator(text)
        else:
            translated_text = None
        frames = self.run(text, seed, only_first_stage)
        video_path = self.to_video(frames)
        return translated_text, video_path, frames
