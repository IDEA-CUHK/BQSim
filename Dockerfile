# Use the official NVIDIA CUDA 12.6 base image
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    libeigen3-dev \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Install GCC 12.3.0
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update && apt-get install -y \
    gcc-12 \
    g++-12 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
    && rm -rf /var/lib/apt/lists/*

# Install CMake 3.22.1
RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh \
    && chmod +x cmake-3.22.1-linux-x86_64.sh \
    && ./cmake-3.22.1-linux-x86_64.sh --skip-license --prefix=/usr/local \
    && rm cmake-3.22.1-linux-x86_64.sh

# Install cuQuantum SDK
RUN wget https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-24.11.0.21_cuda12-archive.tar.xz \
    && tar -xf cuquantum-linux-x86_64-24.11.0.21_cuda12-archive.tar.xz -C /usr/local \
    && rm cuquantum-linux-x86_64-24.11.0.21_cuda12-archive.tar.xz
ENV CUSTATEVEC_ROOT=/usr/local/cuquantum-linux-x86_64-24.11.0.21_cuda12-archive
ENV LD_LIBRARY_PATH=$CUSTATEVEC_ROOT/lib:$LD_LIBRARY_PATH
ENV CUSTATEVEC_LIBRARY=${CUSTATEVEC_ROOT}/lib
# ENV CPATH=$CUSTATEVEC_ROOT/include:$CPATH

# Install Python 3.10.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py

# Install Python packages
RUN python3.10 -m pip install --upgrade pip \
    && python3.10 -m pip install \
    numpy==1.26.4 \
    qiskit-aer==0.15.0 \
    qiskit==1.2.0

# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Verify installations
RUN gcc --version && nvcc --version && cmake --version && python3 --version

# Clone the BQSim repository
RUN git clone https://github.com/IDEA-CUHK/BQSim.git /workspace/BQSim

# Set the working directory
WORKDIR /workspace/BQSim

# Run the compile.sh script
RUN chmod +x docker_compile.sh && ./docker_compile.sh

# Default command
CMD ["bash"]