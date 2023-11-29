# Use an NVIDIA CUDA base image
FROM ian40306/cuda10.0-cudnn7-devel-ubuntu18.04

# Set non-interactive timezone and locales
ENV TZ=Asia/Seoul
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

# Replace the source list (if needed)
RUN sed -i 's|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|http://archive.ubuntu.com/ubuntu/|g' /etc/apt/sources.list

# Remove the existing CUDA lists, delete old keys
RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80

# Add the NVIDIA GPG keys
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Prevent services from starting automatically after installation
RUN echo '#!/bin/sh\nexit 0' > /usr/sbin/policy-rc.d
RUN chmod +x /usr/sbin/policy-rc.d

# Install software-properties-common to add PPAs, SSH server, and clean up
RUN apt-get update && \
    apt-get install -y software-properties-common openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add repository for Python 3.7 and install it
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.7 python3.7-dev python3.7-distutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update alternatives to make Python 3.7 the default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    update-alternatives --set python3 /usr/bin/python3.7

# Install pip for Python 3.7
RUN apt-get update && \
    apt-get install -y curl && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.7 get-pip.py && \
    rm get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the timezone
RUN ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Configure SSH for remote connections
RUN echo "root:password" | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#ChallengeResponseAuthentication no/ChallengeResponseAuthentication yes/' /etc/ssh/sshd_config

# Install PyTorch, torchvision, and torchaudio for the specific CUDA version
RUN python3.7 -m pip install torch==1.6.0 torchvision==0.7.0 torchaudio==0.6.0 -f https://download.pytorch.org/whl/cu102

# Set the working directory to /workspace
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . .

# Expose port 8000 for the service
EXPOSE 8000
