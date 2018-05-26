#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
import predicthelper as ph
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')



argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

# camera or vedio
group = argparser.add_mutually_exclusive_group()

group.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

group.add_argument(
    '--camera',
    help = 'use camera for real time prediction',
    action = 'store_true'                    )

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    input   = args.input
    camera = args.camera

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'],
                threshold           = config['predict']['threshold'],
                max_sur             = config["predict"]["max_sur"])

    ###############################
    #   Load trained weights
    ###############################
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################
    if camera:
        ph.predict_with_camera(yolo)

    elif input[-4:] == '.mp4':
        ph.predict_with_video(yolo,input,config['predict']['saved'])

    else:
        image = cv2.imread(input)
        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'])
        #print(len(boxes), 'boxes are found')
        cv2.imwrite(input[:-4] + '_detected' + input[-4:], image)



if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
