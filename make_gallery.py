from __future__ import print_function
import os
import torchvision.transforms as T
from statistics import mean
from model_detection_test import FaceDetector
from vectorization import Vectorization
import cv2


def make_gallery(name,path_vectori,path_detection,folder_nhanvienup,folder_database):
    vectori= Vectorization(path_vectori)
    face_detector = FaceDetector(path_detection, 0.6, 'CPU')
    folder_path = folder_nhanvienup +name+"/"  #folder chua anh nhan vien
    if not os.path.exists(folder_database+"images/"+name):
        os.makedirs(folder_database+"images/"+name)
    features=[]
    count=0
    for i in os.listdir(folder_path):     
        img_raw = cv2.imread(folder_path+i, cv2.IMREAD_COLOR)
        dets = face_detector.get_detections(img_raw)
        for det in dets:
            boxes= det[:4]
            x_min = int(boxes[0])
            y_min = int(boxes[1])
            x_max = int(boxes[2])
            y_max = int(boxes[3])
            crop_img=img_raw[y_min:y_max, x_min:x_max]
            res=vectori.extract_vector_gallery(crop_img)
            features.append(res)
            cv2.imwrite(folder_database+"images/"+name+"/"+str(count)+".jpg",crop_img)
        count+=1
    file1 = open(folder_database+"vectors/"+name+".txt","w")
    arr1=""
    for m in features:
        for n in m:
            arr1=arr1+str(n[0][0])+" "
        arr1=arr1+"\n"
    file1.write(arr1)
    file1.close()

if __name__ == '__main__':
    path_detection="model_detection_2021/face-detection-0204.xml"
    path_vectori="model_vectori/face_v2_16.xml"
    folder_nhanvienup="data_nvup/"
    folder_database="database/"  # chua images va vectors
    name="hung"
    make_gallery(name,path_vectori,path_detection,folder_nhanvienup,folder_database)

