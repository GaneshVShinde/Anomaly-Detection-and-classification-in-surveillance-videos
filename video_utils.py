import numpy as np
import os
import array
import cv2

import sys

def get_frame_count(video):
    ''' Get frame counts and FPS for a video '''
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("[Error] video={} can not be opened.".format(video))
        sys.exit(-6)

    # get frame counts
    num_frames = int(cap.get(7))
    fps = cap.get(5)
    width = cap.get(3)   # float
    height = cap.get(4)
    print("width = ",end='')
    print(width)
    print("height = ",end='')
    print(height)
    # in case, fps was not available, use default of 29.97
    if not fps or fps != fps:
        fps = 29.97

    return num_frames, fps



def extract_frames(video, start_frame, frame_dir, num_frames_to_extract=16):
    ''' Extract frames from a video using opencv '''

    # check output directory
    if os.path.isdir(frame_dir):
        pass
        #print ("[Warning] frame_dir={} does exist. Will overwrite".format(frame_dir))
    else:
        os.makedirs(frame_dir)

    # get number of frames
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print ("[Error] video={} can not be opened.".format(video))
        sys.exit(-6)

    # move to start_frame
    cap.set(1, start_frame)

    # grab each frame and save
    for frame_count in range(num_frames_to_extract):
        frame_num = frame_count + start_frame
        #print ("{} ".format(frame_num),end='')
        ret, frame = cap.read()
        if not ret:
            print ("[Error] Frame extraction was not successful")
            sys.exit(-7)

        frame_file = os.path.join(
                frame_dir,
                '{0:06f}.jpg'.format(frame_num)
                )
        cv2.imwrite(frame_file, frame)

    return


def create_video(Image_source,output_video):
    img_array = []
    for filename in glob.glob(Image_source+'/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    out = cv2.VideoWriter(output_video,cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    
    out.release()

# def genrate_video_frame(file_path):
#     tmp_dir = "./test2" #to save extracted Frame
#     num_frames_per_clip = 16 

#     # sampling rate (in seconds)
#     sample_every_N_sec = 1

#     max_processing_sec = 599

#     num_frames, fps = get_frame_count(file_path)
#     frame_inc = int(sample_every_N_sec * fps)
#     frame_inc=16
#     start_frame = 1
#     # make sure not to reach the edge of the video
#     end_frame = min(num_frames, int(max_processing_sec * fps)) -num_frames_per_clip
#     start_frames = []
#     for frame_index in range(start_frame, end_frame, frame_inc):
#         #print "[Debug] adding frame_index={}".format(frame_index)
#         start_frames.append(frame_index)

#     video_id, video_ext = os.path.splitext(
#             os.path.basename(file_path)
#             )
#     # each line corresponds to a 16-frame video clip
#     #f_input = open(input_file, 'w')
#     for i,start_frame in enumerate(start_frames):
#         # where to save extracted frames
#         frame_dir = os.path.join(tmp_dir, video_id,str(i),video_id)
#         print(frame_dir)
#         #extract_frames(file_path, start_frame, frame_dir)

def generate_video_from_imgs(video_path):
    pass
