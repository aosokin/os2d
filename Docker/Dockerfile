FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

ARG USERNAME=torch-docker
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG OS2D_ROOT=/workspace

RUN apt-get update && apt-get upgrade -y

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir /etc/sudoers.d \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get install -y \
        sudo \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libhdf5-dev \
        libpq-dev

RUN mkdir -p ${OS2D_ROOT} && chmod -R a+rwx ${OS2D_ROOT}
WORKDIR ${OS2D_ROOT}

RUN git clone https://github.com/aosokin/os2d.git ${OS2D_ROOT}

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# COPY *.sh /tmp/
# RUN /tmp/install_maskrcnn_benchmark.sh
# RUN /tmp/install_cnnimageretrieval-pytorch.sh
# RUN /tmp/install_CoAE_code.sh
RUN "${OS2D_ROOT}/os2d/utils/wget_gdrive.sh" "${OS2D_ROOT}/models/os2d_v2-train.pth" "1l_aanrxHj14d_QkCpein8wFmainNAzo8"

USER ${USERNAME}

ENV SHELL=/bin/bash