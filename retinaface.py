from __future__ import print_function
import os
import torch
import numpy as np
from layers.prior_box import PriorBox,PriorBox_faceboxs
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from utils.box_utils import decode, decode_landm
import time
# from openvino.inference_engine import IENetwork, IEPlugin
from openvino.inference_engine import IECore


cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}
cfg_mnet_faceboxs = {
    'name': 'FaceBoxes',
    'min_sizes': [[32, 64, 128], [256], [512]],
    'steps': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False
}

class RetinaFace(object):
    def __init__(self, path_to_xml, device, plugin_dir=None):
        self.cur_request_id = 0
        self.next_request_id = 0
        model_xml = path_to_xml
        model_bin = os.path.splitext(model_xml)[0] + '.bin'
        # plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
        # net = IENetwork(model=model_xml, weights=model_bin)

        # self.output_blob = next(iter(net.outputs))
        # self.input_blob = next(iter(net.inputs))
        net = IECore().read_network(model_xml, os.path.splitext(model_xml)[0] + ".bin")
        self.input_blob = next(iter(net.input_info))
        self.output_blob = next(iter(net.outputs))

        #self.exec_net = plugin.load(network=net, num_requests=1)
        self.exec_net = IECore().load_network(network=net,device_name='CPU', num_requests=1)
        # print(net.input_info[self.input_blob])
        # exit()
        self.n, self.c, self.h, self.w=net.input_info[self.input_blob].input_data.shape
        #self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        # print (self.n, self.c, self.h, self.w)
        
    
    def get_location(self, next_frame, confidence_threshold=0.4, top_k=5000, nms_threshold=0.6,keep_top_k=750):
        boxes =[]
        scores = []
        initial_h = next_frame.shape[0]
        initial_w = next_frame.shape[1]

        in_frame = cv2.resize(next_frame, (self.w, self.h))
        in_frame = np.float32(in_frame)       
        in_frame -= [104, 117, 123]
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.n, self.c, self.h, self.w)

        self.exec_net.start_async(request_id = self.next_request_id,inputs = {self.input_blob: in_frame}) # 
        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            # exit()
            # loc = torch.tensor(self.exec_net.requests[self.cur_request_id].outputs['Concat_151'])
            # conf = torch.tensor(self.exec_net.requests[self.cur_request_id].outputs['Softmax_202'])
            # landms = torch.tensor(self.exec_net.requests[self.cur_request_id].outputs['Concat_201'])
            loc = torch.tensor(self.exec_net.requests[self.cur_request_id].outputs['output0'])
            conf = torch.tensor(self.exec_net.requests[self.cur_request_id].outputs['586'])
            # loc = torch.tensor(self.exec_net.requests[self.cur_request_id].outputs['boxes'])
            # conf = torch.tensor(self.exec_net.requests[self.cur_request_id].outputs['scores'])
            #landms = torch.tensor(self.exec_net.requests[self.cur_request_id].outputs['585'])
            # print(loc)
            # print(conf)
            # print(landms)
            # exit()    
            scale = torch.Tensor([ initial_w, initial_h, initial_w, initial_h])
            #scale1 = torch.Tensor([ initial_w, initial_h, initial_w, initial_h,initial_w, initial_h, initial_w, initial_h,initial_w, initial_h])
            priorbox = PriorBox(cfg_mnet, image_size=(self.h, self.w))
            #priorbox = PriorBox_faceboxs(cfg_mnet_faceboxs, image_size=(self.h, self.w))
            priors = priorbox.forward()
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
            boxes = boxes * scale 
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            # landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
            # landms = landms * scale1 
            # landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > confidence_threshold)[0]
            boxes = boxes[inds]
            # landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:top_k]
            boxes = boxes[order]
            # landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            # landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:keep_top_k, :]
            # landms = landms[:keep_top_k, :]
            return dets


