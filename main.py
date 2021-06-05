import cv2
import time
from retinaface import RetinaFace
from sort import Sort
import numpy as np
import time
import random
import collections
from TDDFA_OPENVINO import TDDFA_OPENVINO
from utils.box_utils import parse_roi_box_from_bbox, get_poses
from model_detection_test import FaceDetector
from vectorization import Vectorization
from matching import Matching
import datetime
import json
import os

def run(flag,type_cam,path_detection,path_vectori,path_to_headpose,folder_vector):
    time_now=datetime.datetime.now()
    day_now= str(time_now.day)+"_"+str(time_now.month)+"_"+str(time_now.year)
    if not os.path.exists("image_to_debug/"+day_now):
        os.makedirs("image_to_debug/"+day_now)
    if not os.path.exists("results/"+day_now):
        os.makedirs("results/"+day_now)
    tracker = Sort()
    #det=RetinaFace('model_detection_2021/FaceDetector.xml','CPU')
    face_detector = FaceDetector(path_detection, 0.6, 'CPU')
    tddfa = TDDFA_OPENVINO(path_to_headpose)     
    vectori= Vectorization(path_vectori)
    gallery,gallery_id=vectori.build_gallery(folder_vector)
    matching_model= Matching(type_cam,0.4,0.44,0.6,gallery,gallery_id)
    cap=cv2.VideoCapture("video_test.mp4")
    #cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,368))
    ids = []
    count=0
    time_all=0
    color=(0,0,255)
    list_track=collections.defaultdict(list)
    list_checkin=[]
    nhanvien_ok=collections.defaultdict(list)
    scale=0
    count_nguoila=0
    while(True):
        ret, frame = cap.read()
        if(ret):
            h_img,w_img,_=frame.shape
            frame_use=frame
            frameid=count+1
            det_track=[]
            t1=time.time()
            #dets1 = det.get_location(frame_use,confidence_threshold= 0.8)
            # test new detection
            dets1 = face_detector.get_detections(frame_use)

            roi_boxes = [parse_roi_box_from_bbox(box) for box in dets1]
            if len(dets1) > 0:
                param_lst = tddfa(frame_use, roi_boxes)
                pose=get_poses(param_lst,dets1)
            dets=[]
            for j in range(len(dets1)):
                if(abs(pose[j][0])<30 ):   #and abs(pose[j][1])<30
                    dets.append(dets1[j])
            for b in dets:
                b = list(map(int, b))
                det_track.append((b[0], b[1], b[2], b[3],1))
            #tracking 
            predict,dead_track=tracker.update(np.array(det_track))
            if(len(dead_track)>0):
                dead_track.sort()
                for id_dead in dead_track:                 
                    if(len(ids)==0 or id_dead>=max(ids)):
                        break
                    list_vector=[a[-1] for a in list_track[id_dead+1]]
                    if(len(list_vector)<20):
                        continue
                    id_name=matching_model.matching(list_vector)
                    if(id_name==-1):       
                        count_nguoila+=1
                        now = datetime.datetime.now()
                        time_checkin= '{}:{}:{}'.format(now.hour,now.minute,now.second)                
                        img=list_track[id_dead+1][15][0]
                        xmin,ymin,xmax,ymax=list_track[id_dead+1][15][2:6]
                        cv2.imwrite('image_to_debug/'+day_now+"/"+str(time_checkin)+"_"+str(count_nguoila)+"_"+str(flag)+".jpg",img[ymin:ymax,xmin:xmax]) 
                    # if(flag==0):
                    #     if not os.path.exists("image_to_debug/"+day_now+"/"+str(id_name)+"_"+str(flag)+".jpg"):
                    #         cv2.imwrite('image_to_debug/'+day_now+"/"+str(id_name)+"_"+str(flag)+".jpg",img[ymin:ymax,xmin:xmax])   
                    # else: 
                    #     cv2.imwrite('image_to_debug/'+day_now+"/"+str(id_name)+"_"+str(flag)+".jpg",img[ymin:ymax,xmin:xmax])              
                    list_checkin.append(id_name)
                    now = datetime.datetime.now()
                    time_checkin= '{}:{}:{}'.format(now.hour,now.minute,now.second)
                    if(id_name!=-1):
                        nhanvien_ok[id_name]=time_checkin

            if(len(predict>0)):
                for i in range(len(predict)):
                    x1, y1, x2, y2,id=int(predict[i][0]),int(predict[i][1]),int(predict[i][2]),int(predict[i][3]),int(predict[i][4])
                    if(x1<0 or y1<0 or x2>frame_use.shape[1] or y2>frame_use.shape[0]):
                        continue
                    box=frame_use[y1:y2,x1:x2]
                    vector=vectori.extract_vector(box) 
                    if count>1:
                        if id not in ids:
                            ids.append(id)
                            #color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    # cv2.rectangle(frame,(x1,y1+int(h_img*scale)) ,(x2,y2+int(h_img*scale)), color, 2)
                    # cv2.putText(frame,str(id),(x1, y1+int(h_img*scale)+12),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                    # cv2.imwrite('debug/'+str(count-1)+".jpg",frame[y1+int(h_img*scale):y2+int(h_img*scale),x1:x2])
                    cv2.rectangle(frame,(x1,y1) ,(x2,y2), color, 2)
                    cv2.putText(frame,str(id),(x1, y1+12),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                    box_info=(frame,frameid,x1,y1,x2,y2,vector)
                    list_track[id].append(box_info)
            for i in range(len(list_checkin)):
                color = (255,0,0)
                cv2.putText(frame,str(list_checkin[i]).split("_")[0],(w_img-300, 100+15*i),cv2.FONT_HERSHEY_DUPLEX, 1, color)
            count+=1
            time_all=time_all+time.time()-t1
            cv2.imshow('frame',frame)
            frame1=cv2.resize(frame,(640,368))
            out.write(frame1)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # if cv2.waitKey(1) & ((time.time()-t1)>3600):
        #         break
    if(flag==0):
        for i in nhanvien_ok:
            nhanvien_ok[i]=nhanvien_ok[i].split(',')[0]
        with open('results/'+day_now+"/"+'morning.json', 'w') as fp:
            json.dump(nhanvien_ok, fp)
    if(flag==1):
        for i in nhanvien_ok:
            nhanvien_ok[i]=nhanvien_ok[i].split(',')[-1]
        with open('results/'+day_now+"/"+'afternoon.json', 'w') as fp:
            json.dump(nhanvien_ok, fp)
   
    # print(time_all)
    # print(count)
    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path_detection="model_detection_2021/face-detection-0204.xml"
    path_vectori="model_vectori/face_v2_16.xml"
    folder_vector="database/vectors/"
    path_to_headpose="model_headpose/"
    flag=0 # 0 is morning 1 is afternoon
    type_cam=0  # 0 is front, 1 is high camera
    run(0,type_cam,path_detection,path_vectori,path_to_headpose,folder_vector)   # chay sang
    #run(1,type_cam,path_detection,path_vectori,path_to_headpose,folder_vector)


    # schedule.every(3).minutes.do(job)  # test phut
    # schedule.every().day.at("04:14").do(run,flag,path_detection,path_vectori,folder_vector)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1) # wait one minute

