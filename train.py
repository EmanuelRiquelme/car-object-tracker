from ultralytics import YOLO
import torch
from torchvision import transforms 
from PIL import Image
import argparse
import os

# Train the model
#results = model.train(data='data.yaml', epochs=5, imgsz=720)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str,default = 'yolov8n.pt',
                    help='Name of the model')

parser.add_argument('--epochs', type=int,default = 50,
                    help='Number of epochs')
parser.add_argument('--name', type=str,default = 'vehicle_tacker',
                    help='name of the model')
parser.add_argument('--img_res', type=int,default = 1080 ,
                    help='resolution of the images to be trained')

if __name__ == '__main__':
    args = parser.parse_args()
    model = YOLO(args.model_name)
    results = model.train(data='data.yaml', epochs=args.epochs, imgsz=args.img_res,name = args.name,project = os.getcwd(),exist_ok = True)
