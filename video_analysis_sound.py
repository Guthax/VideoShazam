import numpy as np
import cv2
import matplotlib.pyplot as plt
from video_tools import *
import feature_extraction as ft
from scikits.talkbox.features import mfcc
import scipy.io.wavfile as wav


def compute_signal_power(frame, samplerate, framerate, samples):
    samplesperframe = samplerate / framerate
    s = frame * samplesperframe
    s = int(round(s))
    T = int(round(s+samplesperframe))
    res = np.sum(np.square(samples[s:T].astype('int64')))
    print(res)
    return res / samplesperframe
    
    
# Path to video file to analyse 
video = '../Videos/BlackKnight.avi'

# starting point
S = 33 # seconds
# stop at
E = 35 # seconds

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

audiopath= '../Videos/BlackKnight_mono.wav'
samplerate, samples = wav.read(audiopath)
res = []
counter = -1
while(cap.isOpened() and cap.get(cv2.CAP_PROP_POS_MSEC) < (E*1000)):
    counter+=1
    # 
    retVal, frame = cap.read()
    # 
    if retVal == False:
        break

    #== Do your processing here ==#

    res.append(compute_signal_power(counter, samplerate, frame_rate, samples))
    # 
    cv2.imshow('Video', frame)

    
    # 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_frame = frame

#
timestamps = [S+(i)/frame_rate for i in range(len(res))]
plt.stem(timestamps, res)
plt.show()
cap.release()
cv2.destroyAllWindows()
