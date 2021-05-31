import cv2
import os
import numpy as np
import cv2
import time
from openvino.inference_engine import IECore
class Vectorization(object):
    def __init__(self, path_to_xml):
        self.cur_request_id = 0
        self.next_request_id = 0
        model_xml = path_to_xml
        model_bin = os.path.splitext(model_xml)[0] + '.bin'
        net = IECore().read_network(model_xml, os.path.splitext(model_xml)[0] + ".bin")
        self.input_blob = next(iter(net.input_info))
        self.output_blob = next(iter(net.outputs))
        self.exec_net = IECore().load_network(network=net,device_name='CPU', num_requests=1)
        self.n, self.c, self.h, self.w=net.input_info[self.input_blob].input_data.shape
    
    def extract_vector(self, img):
        in_frame = cv2.resize(img, (self.w,self.h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.n, self.c, self.h, self.w)
        res = self.exec_net.infer(inputs={self.input_blob: in_frame})
        res = res[self.output_blob]
        res=res[0]
        res=np.reshape(res,res.shape[0:1])
        return res
    def extract_vector_gallery(self, img):
        in_frame = cv2.resize(img, (self.w,self.h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.n, self.c, self.h, self.w)
        res = self.exec_net.infer(inputs={self.input_blob: in_frame})
        res = res[self.output_blob]
        res=res[0]
        return res
    def build_gallery(self,folder_vector):
        names=[]
        vectors=[]
        for i in os.listdir(folder_vector):
            name=i.split(".")[0]
            vector_all=open(folder_vector+name+".txt","r").read()
            for j in vector_all.split('\n')[0:-1]:
                vector=j.split(" ")[:-1]
                vector = [float(i) for i in vector]
                vectors.append(np.array(vector))
                names.append(name)
        gallery_id=names
        embeddings = np.array(vectors, dtype='f')
        embeddings=np.reshape(embeddings,embeddings.shape[0:2])
        gallery=embeddings
        return gallery,gallery_id

    


