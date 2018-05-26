#! /usr/bin/env python

import argparse
import os
import numpy as np
from preprocessing import parse_annotation
from frontend import YOLO
import json

# only use number 0 GUP to train
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations
    #   预处理部分
    ###############################

    # parse annotations of the training set
    """
    train_imgs是一个list，每个元素是一个dict，里面包括了每个图片的信息：
    img['filename']: str, 图片完整路径
    img['width']: int, 图片宽度
    img['height']: int, 图片长度
    img['object']: list, 所有标注object的list，每个object是一个dict
        obj['name']: str object的label名字
        obj['xmin']: int
        obj['xmax']: int
        obj['ymin']: int
        obj['ymax']: int
    
    train label是一个dictionary，key为label，value为出现的次数
    """




    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'], 
                                                config['train']['train_image_folder'], 
                                                config['model']['labels'])

    # parse annotations of the validation set, if any, otherwise split the training set
    np.random.shuffle(train_imgs)
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'], 
                                                    config['valid']['valid_image_folder'], 
                                                    config['model']['labels'])
    else:
        train_valid_split = int(0.8*len(train_imgs))


        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        print('Seen labels:\t', train_labels)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)           

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        print(f"labels:{train_labels}")
        config['model']['labels'] = train_labels.keys()
        
    ###############################
    #   Construct the model 
    ###############################

    """
    这里创建了一个yolo类型的对象
    模型特征提取部分使用config里backend的设置
    注意这里的模型建立只对模型的特征提取部分，即卷积部分做了构建，没有对后面的分类部分，即全连接层做构建
    labels是config中定义的labels，当空时为全部找到的label（看preprocessing的代码）
    max_box_per_image顾名思义，但是注意在config中的anchors的设置数量一定要一样，不然没有意义
    anchors到底是什么可以看论文中对anchor权重的说明。
    """
    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load the pretrained weights (if any)
    """
    注意这里load的weight是load之前训练好的
    也就是说，之前跑过一遍这个代码训练出来的weight才能fit上去。
    因为其实这里结构比较特殊，特征提取部分被独立成一个layer了
    """
    ###############################    

    if os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in", config['train']['pretrained_weights'])
        yolo.load_weights(config['train']['pretrained_weights'])

    ###############################
    #   Start the training process 
    ###############################

    yolo.train(train_imgs         = train_imgs,
               valid_imgs         = valid_imgs,
               train_times        = config['train']['train_times'],
               valid_times        = config['valid']['valid_times'],
               nb_epochs          = config['train']['nb_epochs'], 
               learning_rate      = config['train']['learning_rate'], 
               batch_size         = config['train']['batch_size'],
               warmup_epochs      = config['train']['warmup_epochs'],
               object_scale       = config['train']['object_scale'],
               no_object_scale    = config['train']['no_object_scale'],
               coord_scale        = config['train']['coord_scale'],
               class_scale        = config['train']['class_scale'],
               saved_weights_name = config['train']['saved_weights_name'],
               debug              = config['train']['debug'])

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
