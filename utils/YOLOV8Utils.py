import os
import shutil
from ultralytics import YOLO
#import onnx
import openvino as ov
#import torch
#from torchvision import datasets, transforms
#import nncf

class YOLOV8Utils():

    @classmethod
    def download(cls, target_path):
        print('Downloading YOLOV8...')
        # Load a model
        model = YOLO("yolov8n.yaml")  # build a new model from scratch
        model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

        # Use the model
        print(f'saving yolov8 to {target_path}')
        path = model.export(format="openvino", int8=True, data='./data.yml')  # export the model to ONNX format
        #shutil.move(path, target_path)
        #os.remove('./yolov8n.pt') 
        return None

    @classmethod
    def onnx2ov(cls, onnx_path, ov_dir):
        ov_model = ov.convert_model('./models/yolov8n.onnx')
        val_dataset = datasets.ImageFolder("./root", transform=transforms.Compose([transforms.ToTensor()]))
        dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

        # Step 1: Initialize transformation function
        onnx_model = onnx.load_model(onnx_path)
        input_name = onnx_model.graph.input[0].name
        def transform_fn(data_item):
            images, _ = data_item
            return {input_name: images.numpy().reshape([1,3,640,640])}

        calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
        quantized_ov_model = nncf.quantize(ov_model, calibration_dataset)
        ov.save_model(quantized_ov_model, f'{ov_dir}/yolov8n.xml')

