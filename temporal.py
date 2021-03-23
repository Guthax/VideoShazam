import numpy as np
import cv2
import matplotlib.pyplot as plt
from video_tools import *
import feature_extraction as ft
from scikits.talkbox.features import mfcc

def temporal_diff(frame, prev_frame, threshold):

    framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameprevgray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diff = np.abs(frameprevgray.astype('int16')-framegray.astype('int16'))
    diff = diff > threshold    
    diff = np.sum(diff)
    #diff = [0 if x < thresh else 1 for x in diff]
    return diff

# Path to video file to analyse 
video = '../Videos/TUDelft_Ambulance_Drone.mp4'

# starting point
S = 20 # seconds
# stop at
E = 30 # seconds

# Retrieve frame count. We need to add one to the frame count because cv2 somehow 
# has one extra frame compared to the number returned by avprobe.
frame_count = get_frame_count(video) + 1
frame_rate = get_frame_rate(video)

# create an cv2 capture object
cap = cv2.VideoCapture(video)

# store previous frame
prev_frame = None

# set video capture object to specific point in time
cap.set(cv2.CAP_PROP_POS_MSEC, S*1000)
TD = []
while(cap.isOpened() and cap.get(cv2.CAP_PROP_POS_MSEC) < (E*1000)):

    # 
    retVal, frame = cap.read()
    # 
    if retVal == False:
        break

    #== Do your processing here ==#


    # 
    cv2.imshow('Video', frame)
    if prev_frame is not None:
        TD.append(temporal_diff(frame,prev_frame, 150))


    # 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_frame = frame

#
print(np.max(TD))
timestamps = [S+(i+1)/frame_rate for i in range(len(TD))]
plt.stem(timestamps, TD)
plt.show()
cap.release()
cv2.destroyAllWindows()
