from __future__ import print_function
import os
import torch
import numpy as np
import cv2
import time
from openvino.inference_engine import IECore
import torchvision.transforms as T
from scipy import spatial
import cv2
import time
from retinaface import RetinaFace
from collections import Counter
from statistics import mean
from model_detection_test import FaceDetector



def make_gallery(name):
    path_to_xml="model_vectori/face_v2_16.xml"  #link model
    model_xml = path_to_xml
    model_bin = os.path.splitext(model_xml)[0] + '.bin'
    net = IECore().read_network(model_xml, os.path.splitext(model_xml)[0] + ".bin")
    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))
    exec_net = IECore().load_network(network=net,device_name='CPU', num_requests=1)
    n, c, h, w=net.input_info[input_blob].input_data.shape
    #det_ = RetinaFace('model_detection_2021/FaceDetector.xml', 'CPU')
    path_to_xml_detection="model_detection_2021/face-detection-0204.xml"
    face_detector = FaceDetector(path_to_xml_detection, 0.8, 'CPU')
    folder_path = 'data_nvup/'+name+"/"  #folder chua anh nhan vien
    if not os.path.exists("database/images/"+name):
        os.makedirs("database/images/"+name)
    features=[]
    count=0
    for i in os.listdir(folder_path):     
        file_name, extension = os.path.splitext(i)
        img_raw = cv2.imread(folder_path+i, cv2.IMREAD_COLOR)
        # dets = det_.get_location(img_raw,confidence_threshold= 0.8)
        dets = face_detector.get_detections(img_raw)
        for det in dets:
            boxes= det[:4]
            x_min = int(boxes[0])
            y_min = int(boxes[1])
            x_max = int(boxes[2])
            y_max = int(boxes[3])
            crop_img=img_raw[y_min:y_max, x_min:x_max]
            in_frame = cv2.resize(crop_img, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            res = exec_net.infer(inputs={input_blob: in_frame})
            res = res[output_blob]
            res=res[0]
            features.append(res)
            cv2.imwrite("database/images/"+name+"/"+str(count)+".jpg",crop_img)
        count+=1
    file1 = open("database/vectors/"+name+".txt","w")
    arr1=""
    for m in features:
        for n in m:
            arr1=arr1+str(n[0][0])+" "
        arr1=arr1+"\n"
    file1.write(arr1)
    file1.close()

if __name__ == '__main__':
    name="hung"
    make_gallery(name=name)

