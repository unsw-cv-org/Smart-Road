# CBLC datasets annotation to VOC format

import os
import xml.etree.ElementTree as ET
import argparse
import cv2
import numpy as np

argparser = argparse.ArgumentParser(description="Convert CBLC dataset to VOC format.")
argparser.add_argument('--annotation_path',
                   help="path to annotations",
                   required=True
                   )

argparser.add_argument('--image_path',
                       help="path to images",
                       required=True)

argparser.add_argument("--saved_path",
                       help="path to save converted annotations",
                       default='Annotation_XML/')




def _main_(args):

    annot_path = args.annotation_path
    image_path = args.image_path
    saved_path = args.saved_path

    for ann in os.listdir(annot_path):
        tree = ET.parse(annot_path + ann)
        width = ET.Element("width")
        height = ET.Element("height")
        root = tree.getroot()

        elem = root.find('filename')
        file_path = os.path.join(image_path,elem.text[1:-4]+"jpg")
        assert(os.path.exists(file_path))
        elem.text = elem.text[1:-4]+"jpg"
        img = cv2.imread(file_path)
        height.text = str(img.shape[0])
        width.text = str(img.shape[1])
        root.append(height)
        root.append(width)

        for elem in root.iter('object'):
            x_list = []
            y_list = []
            for object in elem.iter("name"):
                if "pedestrian" in object.text:
                    #print("change to person")
                    object.text = "person"
                if "car" in object.text:
                    #print("change to car")
                    object.text = "car"
            for x in elem.iter('x'):
                x_list.append(int(x.text))
            for y in elem.iter('y'):
                y_list.append(int(y.text))
            if len(x_list)==0 or len(y_list)==0:
                continue
            bndbox = ET.Element("bndbox")
            xmin = ET.Element("xmin")
            xmax = ET.Element("xmax")
            ymin = ET.Element("ymin")
            ymax = ET.Element("ymax")
            xmin.text = str(min(x_list))
            xmax.text = str(max(x_list))
            ymin.text = str(min(y_list))
            ymax.text = str(max(y_list))
            bndbox.append(xmin)
            bndbox.append(xmax)
            bndbox.append(ymin)
            bndbox.append(ymax)
            elem.append(bndbox)
        print(f"writing:{saved_path+ann}")
        tree.write(saved_path+ann)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)

