# RTSP-Object-Detection
An RTSP man in the middle for real time object detection.

## Overview
This is a (near) real time object detection container that is a man in the middle between an RSTP source and end user. The model used is YOLOV8n which needs to be downloaded/converted into onnx and stored in models/. RTSP GStreamer implemtation from [prabhakar-sivanesan](https://github.com/prabhakar-sivanesan/OpenCV-rtsp-server) and YOLOV8n bounding boxes from [ibaiGorordo/ONNX-YOLOv8-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection/tree/main)
Inference time has been tested on 8th gen Intel i7 and allows 10fps streaming

## Configurations
Add the parameters for the rtsp source in the environment variables of the docker-compose file

## Build the Container Locally
`docker build . -t rtsp-object-detection:latest`

## Start the Container
To run the container in the background: `docker compose up -d`

Or remove the -d flag to attach to the stdout and ensure the stream is running

## View Steam
Open a network stream in VLC with the IP of your host system `rtsp://192.168.1.100:8554/video_stream`

The video_sream endpoint can be modified in the docker-compose file.

## Manually Downloading the YOLOV8s INT8 model
```
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
path = model.export(format="openvino", int8=True, data='coco8.yaml') 
```
