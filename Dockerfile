# CUDA 11.4
FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

MAINTAINER Yun Chang "yunchang@mit.edu"

ARG DEBIAN_FRONTEND=noninteractive
MAINTAINER Yun Chang "yunchang@mit.edu"

ARG OPENCV_VERSION=4.7.0

RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    apt-get install -y \
    python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        vim \
        pkg-config \
        lsb-release \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libtbb-dev \
        libgtk2.0-dev \
        pkg-config \
        ## Python
        python3-dev \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN cd /opt/ &&\
    # Download and unzip OpenCV and opencv_contrib and delte zip files
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    # Create build folder and switch to it
    mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build &&\
    # Cmake configure
    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DWITH_CUBLAS=ON \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.4 \
        -DPYTHON_EXECUTABLE=~/usr/bin/python3 \
        -DOPENCV_DNN_CUDA=ON \
        -DENABLE_FAST_MATH=1 \
        -DCUDA_FAST_MATH=1 \
        -DCUDA_ARCH_BIN=7.5 \
        -DWITH_CUBLAS=1 \
        -DHAVE_opencv_python3=ON \
        -DCMAKE_BUILD_TYPE=RELEASE \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. &&\
    # Make
    make -j"$(nproc)" && \
    # Install to /usr/local/lib
    make install && \
    ldconfig &&\
    # Remove OpenCV sources and build folder
    rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}

SHELL ["/bin/bash", "-c"]

# ROS install
RUN sed -i -e 's/us.archive.ubuntu.com/archive.ubuntu.com/g' /etc/apt/sources.list &&\
    apt-get update &&\
    apt-get install curl &&\
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' &&\
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - &&\
    apt-get update &&\
    apt-get install -y ros-noetic-ros-base &&\
    apt-get install -y ros-noetic-cv-bridge &&\
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc &&\
    source ~/.bashrc

# Set working directory to /root
ENV DIRPATH /root/
WORKDIR $DIRPATH
COPY ./ blurring_pipeline/

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r blurring_pipeline/req.txt
