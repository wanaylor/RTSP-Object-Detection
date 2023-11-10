import os
import shutil
from ultralytics import YOLO
class YOLOV8Utils():

    @classmethod
    def download(cls, target_path):
        print('Downloading YOLOV8...')
        # Load a model
        model = YOLO("yolov8n.yaml")  # build a new model from scratch
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Use the model
        print(f'saving yolov8 to {target_path}')
        path = model.export(format="onnx")  # export the model to ONNX format
        shutil.move(path, target_path)
        os.remove('./yolov8n.pt') 
        return None
