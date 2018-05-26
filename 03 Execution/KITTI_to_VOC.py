# txt_to_xml.py
# encoding:utf-8
# 根据一个给定的XML Schema，使用DOM树的形式从空白文件生成一个XML
from xml.dom.minidom import Document
import cv2
import os
import argparse

argspar = argparse.ArgumentParser(
    description="KITTI_to_VOC.py --annotaion_dir annot_dir --image_dir img_dir --classes class1 --classes class2 ..."
)
argspar.add_argument("--annotation_dir",
                  required = True,
                  help = "dir of .txt annotation")
argspar.add_argument("--image_dir",
                  required = True,
                  help = "dir of images")
argspar.add_argument("--xml_dir",
                  required = True,
                  help = "dir for xml annotation")
argspar.add_argument("--classes",
                  action='append',
                  help = "all class desired")



def generate_xml(name,split_lines,img_size,class_ind,xml_dir):
    doc = Document()  # 创建DOM文档对象

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    title = doc.createElement('folder')
    title_text = doc.createTextNode('KITTI')
    title.appendChild(title_text)
    annotation.appendChild(title)

    img_name=name+'.png'

    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement('source')
    annotation.appendChild(source)

    title = doc.createElement('database')
    title_text = doc.createTextNode('The KITTI Database')
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement('annotation')
    title_text = doc.createTextNode('KITTI')
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement('size')
    annotation.appendChild(size)

    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    for split_line in split_lines:
        line=split_line.strip().split()
        if line[0] in class_ind:
            object = doc.createElement('object')
            annotation.appendChild(object)

            title = doc.createElement('name')
            title_text = doc.createTextNode(line[0])
            title.appendChild(title_text)
            object.appendChild(title)

            bndbox = doc.createElement('bndbox')
            object.appendChild(bndbox)
            title = doc.createElement('xmin')
            title_text = doc.createTextNode(str(int(float(line[4]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymin')
            title_text = doc.createTextNode(str(int(float(line[5]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('xmax')
            title_text = doc.createTextNode(str(int(float(line[6]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymax')
            title_text = doc.createTextNode(str(int(float(line[7]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)

    # 将DOM对象doc写入文件
    f = open(xml_dir+name+'.xml','w')
    f.write(doc.toprettyxml(indent = ''))
    f.close()

if __name__ == '__main__':
    argspar = argspar.parse_args()

    class_ind= argspar.classes
    annot_dir=argspar.annotation_dir
    img_dir = argspar.image_dir
    xml_dir = argspar.xml_dir


    for parent, dirnames, filenames in os.walk(annot_dir): # 分别得到根目录，子目录和根目录下文件
        for file_name in filenames:
            full_path=os.path.join(parent, file_name) # 获取文件全路径
            f=open(full_path)
            split_lines = f.readlines()
            name= file_name[:-4] # 后四位是扩展名.txt，只取前面的文件名
            img_name=name+'.png'
            img_path=os.path.join(img_dir,img_name) # 路径需要自行修改
            img_size=cv2.imread(img_path).shape
            generate_xml(name,split_lines,img_size,class_ind,xml_dir)
print('all txts has converted into xmls')