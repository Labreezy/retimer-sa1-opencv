import cv2
import numpy as np
import sys
from multiprocessing import Pool, Queue
FPS = 30
VASL_WINDOW_DIMS = (1440,1080)

regions_normalized = {
    'b1': [0,0,600/1440,100/1080],
    'b2': [0,800/1440,600/1440,100/1080]
}
user_game_region = [0,0,720,960]

def crop_video(frame,x,y,w,h):
    vidh,vidw = frame.shape[:2] #just in case there are channels
    x_adj = int(x*vidw)
    w_adj = int(w*vidw)
    y_adj = int(y*vidh)
    h_adj = int(h*vidh)
    x_end = x_adj + w_adj
    y_end = y_adj + h_adj
    crop_frame = frame[y_adj:y_end, x_adj:x_end]
    return crop_frame

def mean_abs_error_black(frame):
    return frame.flatten().mean()



vid_fn = 'wither_tails_start.mp4'
cap = cv2.VideoCapture(vid_fn)





frame_counter = 0
load_intervals = []
last_frame_loading = False
metrics = {}
WRITE_VIDEO = False
if WRITE_VIDEO:
    res = cv2.VideoWriter('nim_tails_start_noloads.mp4',cv2.VideoWriter_fourcc(*"MP4V"),FPS,(1280,720))
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255.0
        if frame_counter == 0:
            cv2.imshow('frame', grayframe)
            k = cv2.waitKey(0)
        grayframe_reg = crop_video(grayframe, *user_game_region)

        for region_name,dims in regions_normalized.items():
            region = crop_video(grayframe_reg,*dims)
            if region_name == 'b2' and frame_counter == 1000:
                cv2.imshow('fuckin panda',region)
                k = cv2.waitKey(0)
            metric = mean_abs_error_black(region)
            metrics[region_name] = metric

        if metrics['b1'] < .01 and metrics['b2'] < .01:
            if not last_frame_loading:
                load_intervals.append(frame_counter)
                last_frame_loading = True
        elif metrics['b1'] >= .01 or metrics['b2'] >= .01:
            if last_frame_loading:
                load_intervals.append(frame_counter)
                last_frame_loading = False
            if WRITE_VIDEO:
                res.write(frame)
    else:
        print("ret is false, probably done with stream")
        if len(load_intervals) % 2 != 0:
            load_intervals.append(frame_counter)
        break
    frame_counter += 1
cap.release()
res.release()
print(f"{frame_counter} frames processed @ 30fps")
for i in range(0,len(load_intervals),2):
    start = load_intervals[i]
    end = load_intervals[i+1]
    print(f"Load detectecd from  {start/FPS} s to {end/FPS} s")


