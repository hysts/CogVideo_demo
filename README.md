# CogVideo demo
This is an unofficial demo app for [CogVideo](https://github.com/THUDM/CogVideo).

You can try web demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/THUDM/CogVideo) (This version currently supports only the first stage.)

https://user-images.githubusercontent.com/25161192/180413610-63f2b76b-684f-404b-9d13-6c0033987b1f.mp4

https://user-images.githubusercontent.com/25161192/180413654-20ce822f-be7d-40cb-aff3-1712a7505a2c.mp4

It takes about 7 minutes to load models on startup and about 11 minute to generate one video.

## Prerequisite
An A100 instance is required to run CogVideo.

## Installation
### Change default-runtime of docker
First, put `"default-runtime": "nvidia"` in `/etc/docker/daemon.json`.
See: https://github.com/NVIDIA/nvidia-docker/issues/1033#issuecomment-519946473
```json
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

Then, restart docker.
```bash
sudo systemctl restart docker
```

### Clone this repo
```bash
git clone --recursive https://github.com/hysts/CogVideo_demo
cd CogVideo_demo
```

### Build docker image
```bash
docker build . -t cogvideo
```

### Apply patch to CogVideo repo
```bash
cd CogVideo
patch -p1 < ../patch
```

### Download pretrained models (Optional)
The pretrained models will be downloaded automatically on the first run,
but it may take quite some time.
So you may want to download them in advance.

This repo assumes the pretrained models are stored in the `pretrained` directory as follows:
```
pretrained
├── cogvideo-stage1
│   ├── 27000
│   │   └── mp_rank_00_model_states.pt
│   ├── latest
│   └── model_config.json
├── cogvideo-stage2
│   ├── 38000
│   │   └── mp_rank_00_model_states.pt
│   ├── latest
│   └── model_config.json
└── cogview2-dsr
     ├── 20000
     │   └── mp_rank_00_model_states.pt
     ├── latest
     └── model_config.json
```

## Run
You can run the app with the following command:
```bash
docker compose run --rm app
```

The app will start up on port 7860 by default.
You can change the port using `GRADIO_SERVER_PORT` environment variable.
Use port forwarding when running on GCP, etc.
