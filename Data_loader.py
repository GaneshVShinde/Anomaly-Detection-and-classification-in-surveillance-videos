import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime



def get_video_list(path):
    videos=[]
    with open(path, 'r') as f:
        for line in f:
            videos.append(line.strip())
    return videos


def get_training_set(path): #returns just video frames

    abnormal_path =path

    normal_path = path

    batchsize=60
    batch_Ab_size = 30

    num_abnormal = 170

    num_normal =160

    abnorm_list = np.random.permutation(num_abnormal)
    abnorm_list=abnorm_list[:batch_Ab_size]

    norm_list = np.random.permutation(num_normal)
    norm_list = norm_list[:batch_Ab_size]
    
    #print("Loading abnornmal Features...")
    videos = get_video_list(abnormal_path+"/anomaly.txt")
    
    abNormal_features =[]

    abnormal_features=[]
    for i in abnorm_list:
        vid_path=os.path.join(abnormal_path,videos[i] )
        with open(vid_path,'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            abNormal_features.append(np.float32(line.split()))

    abNormal_features=np.array(abNormal_features)    
    
    videos = get_video_list(abnormal_path+"/normal.txt")
    normal_features=[]

    for i in norm_list:
        vid_path=os.path.join(normal_path,videos[i] )
        if (os.path.isfile(vid_path) ):
            with open(vid_path,'r') as f:
                lines = f.read().splitlines()
            for line in lines:
                  normal_features.append(np.float32(line.split()))

    normal_features=np.array(normal_features)
    
    #print("Normal Features Loaded successfully")
    return abNormal_features, normal_features


def get_testing_videos(path): # returns videos with dictionary
    abnormal_path =path

    normal_path = path

    batchsize=60
    batch_Ab_size = 30

    num_abnormal = 170

    num_normal =160

    abnorm_list = np.random.permutation(num_abnormal)
    abnorm_list=abnorm_list[:batch_Ab_size]

    norm_list = np.random.permutation(num_normal)
    norm_list = norm_list[:batch_Ab_size]

    videos = get_video_list(abnormal_path+"/anomaly.txt")
    
    abNormal_features ={}

    for i in abnorm_list:
        vid_path=os.path.join(abnormal_path,videos[i] )
        with open(vid_path,'r') as f:
            lines = f.read().splitlines()
            key=os.path.basename(videos[i])
            key=os.path.splitext(key)[0]
            #print(key)
            abNormal_features[key]=[]
        for line in lines:
            abNormal_features[key].append(np.float32(line.split()))
        abNormal_features[key]=np.array(abNormal_features[key])
    
    videos = get_video_list(abnormal_path+"/normal.txt")
    normal_features={}

    for i in norm_list:
        vid_path=os.path.join(normal_path,videos[i] )
        if (os.path.isfile(vid_path) ):
            with open(vid_path,'r') as f:
                lines = f.read().splitlines()
                key=os.path.basename(videos[i])
                key=os.path.splitext(key)[0]
                normal_features[key]=[]
            for line in lines:
                  normal_features[key].append(np.float32(line.split()))
            
            normal_features[key]=np.array(normal_features[key])

    #normal_features=np.array(normal_features)
    
    #print("Normal Features Loaded successfully")
    return abNormal_features, normal_features


def get_features_specific_video(key,path,is_abnormal=True):
    
    if is_abnormal:
        videos = get_video_list(path+"/anomaly.txt")
    else:
        videos = get_video_list(path+"/normal.txt")
        
    for i,video in enumerate(videos):
        
        if key in video:
            vid_path=os.path.join(path,videos[i] )
            with open(vid_path,'r') as f:
                lines = f.read().splitlines()
            features=[]
            for line in lines:
                features.append(np.float32(line.split()))
            return np.array(features)



def load_Normal_test_data():
    path="./out"
    videos = get_video_list("./out/testing.txt")
    
    features =[]
    for i,video in enumerate(videos):
        vid_path=os.path.join(path,video)
        with open(vid_path,'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            features.append(np.float32(line.split()))
    
    return np.array(features)
        
