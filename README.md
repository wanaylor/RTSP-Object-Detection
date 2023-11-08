# RTSP-Object-Detection
An RTSP man in the middle for real time object detection.

## Overview
This is a (near) real time object detection container that is a man in the middle between an RSTP source and end user. The model used is YOLOV8n which needs to be downloaded/converted into onnx and stored in models/. RTSP GStreamer implemtation from [prabhakar-sivanesan](https://github.com/prabhakar-sivanesan/OpenCV-rtsp-server) and YOLOV8n bounding boxes from [ibaiGorordo/ONNX-YOLOv8-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection/tree/main)
Inference time has been tested on 8th gen Intel i7 and allows 10fps streaming

## Configurations
Add the parameters for the rtsp source in the environment variables of the Dockerfile

## Once Inside the Container
`python3 stream.py --fps 10 --image_width 1920 --image_height 1080 --port 8554 --stream_uri /video_stream`
