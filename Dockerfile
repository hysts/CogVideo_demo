FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    # pyenv dependencies \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG PYTHON_VERSION=3.9.13
ENV PYENV_ROOT /opt/pyenv
ENV PATH ${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}
RUN curl https://pyenv.run | bash
RUN pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION}
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    git checkout 1403c21 && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
RUN rm -rf ${HOME}/.cache/pip

WORKDIR /work
ENV PYTHONPATH /work/:${PYTHONPATH}

CMD ["python", "app.py"]
