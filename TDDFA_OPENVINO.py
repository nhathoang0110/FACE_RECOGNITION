import os

import cv2
import numpy as np
from openvino.inference_engine import IECore
from utils.box_utils import crop_img, _load


class TDDFA_OPENVINO(object):
    def __init__(self):
        # config
        self.size = 120

        model_bin = "model_headpose/mb05_120x120.bin"
        model_xml="model_headpose/mb05_120x120.xml"
        net=IECore().read_network(model_xml, model_bin)
        self.input_blob = next(iter(net.input_info))
        self.exec_net = IECore().load_network(network=net,device_name='CPU', num_requests=1)

        # params normalization config
        r = _load("model_headpose/param_mean_std_62d_120x120.pkl")
        self.param_mean = r.get('mean')
        self.param_std = r.get('std')

    def __call__(self, img_ori, roi_boxes):
        param_lst = []
        for roi_box in roi_boxes:
            img = self.preprocess(img_ori, roi_box)
            param = self.exec_net.infer(inputs={self.input_blob: img})
            key = list(param.keys())[0]
            param = param[key]
            param = param.flatten().astype(np.float32)
            param = param * self.param_std + self.param_mean  # re-scale
            param_lst.append(param)

        return param_lst

    def preprocess(self, img_ori, roi_box):
        img = crop_img(img_ori, roi_box)
        img = cv2.resize(img, dsize=(self.size, self.size),
                         interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]

        return img