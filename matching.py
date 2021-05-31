import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


class Matching(object):
    def __init__(self, type_cam,distance_thresh,distance_thresh_v2,distance_thresh_camcao,gallery,gallery_id):
        self.type_cam = type_cam
        self.distance_thresh = distance_thresh
        if(self.type_cam==0):
            self.distance_thresh = distance_thresh
        else:
            self.distance_thresh = distance_thresh_camcao
        self.gallery = gallery
        self.gallery_id = gallery_id
    
    def matching(self, list_vector):
        distance = 1-np.absolute(cosine_similarity(list_vector, self.gallery))
        ids = np.argsort(distance).astype(np.int32)[:, :-1]
        scores=[[0]]*len(distance)
        for i in range(len(distance)):
            scores[i]=distance[i][ids[i]]
        best_scores = []
        best=[]
        best_v2=[]
        for i in range(len(ids)):
            if scores[i][0]<self.distance_thresh:
                best.append(self.gallery_id[ids[i][0]])
            else:
                best.append(-1)
        b = Counter(best)
        # for i in range(len(ids)):
        #     print(scores[i][0])
        #     print(self.gallery_id[ids[i][0]])
        #     print("\n")
        # print(b)
        name=b.most_common(1)[0][0]
        if(name==-1 and len(b)==2 and b.most_common(2)[1][1]>20):
            name=b.most_common(2)[1][0]
        # if(name==-1):
        #     for i in range(len(ids)):
        #         if scores[i][0]<self.distance_thresh_v2:
        #             best_v2.append(self.gallery_id[ids[i][0]])
        #         else:
        #             best_v2.append(-1)
        #     b2=Counter(best_v2)
        #     name=b2.most_common(1)[0][0]
        #     print(b2)
        # if(name==-1 and len(b2)==2 and b2.most_common(2)[1][1]>20) or (name==-1 and len(b2)>2  and b2.most_common(2)[1][1]>20 and b2.most_common(3)[2][1]<5):
        #     name=b2.most_common(2)[1][0]
        return name

    


