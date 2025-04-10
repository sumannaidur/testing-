FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    git \
    libyaml-dev \
    ffmpeg \
    libfftw3-dev \
    libsamplerate0-dev \
    libtag1-dev \
    libchromaprint-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libboost-dev \
    libeigen3-dev \
    cmake \
    wget \
    unzip

# Clone and install Essentia
RUN git clone --depth 1 https://github.com/MTG/essentia.git && \
    cd essentia && \
    mkdir build && cd build && \
    cmake -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=$(which python3) .. && \
    make -j4 && make install && ldconfig

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app
WORKDIR /app

CMD ["python3", "main.py"]
