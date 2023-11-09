FROM ubuntu:22.04
LABEL maintainer="wanaylor"

ARG DEBIAN_FRONTEND=noninteractive

# Use login shell to read variables from `~/.profile` (to pass dynamic created variables between RUN commands)
SHELL ["sh", "-lc"]

RUN apt update
RUN apt-get install -y libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio libgstreamer1.0-dev
RUN apt-get install -y libglib2.0-dev libgstrtspserver-1.0-dev gstreamer1.0-rtsp
RUN apt-get install -y python3-pip

# For video model testing
RUN python3 -m pip install --no-cache-dir decord av==9.2.0

# cv2 for vision tasks
RUN python3 -m pip install --no-cache-dir opencv-python

# Onnx runtime
RUN python3 -m pip install --no-cache-dir onnxruntime

# OpenVino Runtime
RUN python3 -m pip install --no-cache-dir openvino

# nncf for model optimization
RUN python3 -m pip install --no-cache-dir nncf
