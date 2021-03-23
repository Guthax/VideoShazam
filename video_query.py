#!/usr/bin/env python
import argparse
import video_search
import numpy as np
import cv2
import glob
from scipy.io import wavfile
from video_tools import *
import feature_extraction as ft    
import sys
import os
from fractions import gcd
import matplotlib.pyplot as plt
from video_features import *
  

def normalize_histogram(res1, hist1, res2, hist2):
    denom = gcd(res1, res2)
    print res1, res2, denom, res1/ denom, res2 / denom
    factor1 = res1 / denom
    factor2 = res2 / denom
    print factor1, factor2
    hist1 = normalize(res1, hist1)
    hist2 = normalize(res2, hist2) 
    return hist1, hist2
    
def normalize(factor, hist):
    print 'dividing by', factor
    hist = hist / factor
    return hist
features = ['colorhists', 'tempdiffs', 'audiopowers', 'mfccs', 'colorhistdiffs']
 
parser = argparse.ArgumentParser(description="Video Query tool")
parser.add_argument("training_set", help="Path to training videos and wav files")
parser.add_argument("query", help="query video")
parser.add_argument("-s", help="Timestamp for start of query in seconds", default=0.0)
parser.add_argument("-e", help="Timestamp for end of query in seconds", default=0.0)
parser.add_argument("-f", help="Select features "+str(features)+" for the query ", default='colorhists')
args = parser.parse_args()

if not args.f in features:
    print "Requested feature '"+args.f+"' is not a valid feature. Please use one of the following features:"
    print features
    

cap = cv2.VideoCapture(args.query)
frame_count = get_frame_count(args.query) + 1
frame_rate = get_frame_rate(args.query )
q_duration = float(args.e) - float(args.s)
q_total = get_duration(args.query)

videoheight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
videowidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

if not float(args.s) < float(args.e) < q_total:
    print 'Timestamp for end of query set to:', q_duration
    args.e = q_total

# Load audio data if necessary
if args.f == features[2] or args.f == features[3]:
    filename, fileExtension = os.path.splitext(args.query)
    audio = filename + '.wav'
    fs, wav_data = wavfile.read(audio)

query_features = []
prev_frame = None
prev_colorhist = None
frame_nbr = int(args.s)*frame_rate
cap.set(cv2.CAP_PROP_POS_MSEC, int(args.s)*1000)



print(cap.isOpened())
while(cap.isOpened() and cap.get(cv2.CAP_PROP_POS_MSEC) < (int(args.e)*1000)):

    ret, frame = cap.read()
    if frame is None:
        break

    #perform histogram equalization on the frame

    h =  None
    if args.f == features[0]: 
        h = ft.colorhist(frame)
    elif args.f == features[1]:
        h = temporal_diff(prev_frame, frame, 10)
    elif args.f == features[2] or args.f == features[3]:
        audio_frame = frame_to_audio(frame_nbr, frame_rate, fs, wav_data)
        if args.f == features[2]:
            h = np.mean(audio_frame**2)
        elif args.f == features[3]:
            h, mspec, spec = ft.extract_mfcc(audio_frame, fs)
    elif args.f == features[4]:
        colorhist = ft.colorhist(frame)
        cv2.normalize(colorhist, colorhist)
        h = colorhist_diff(prev_colorhist, colorhist)
        prev_colorhist = colorhist
            
    if h is not None:
        query_features.append(h)
    prev_frame = frame
    frame_nbr += 1


# Compare with database

video_types = ('*.mp4', '*.MP4', '*.avi')
audio_types = ('*.wav', '*.WAV')

# grab all video file names
video_list = []
for type_ in video_types:
    files = args.training_set + '/' +  type_
    video_list.extend(glob.glob(files))	

db_name = 'db/video_database.db'
search = video_search.Searcher(db_name)

def sliding_window(x, w, compare_func):
    """ Slide window w over signal x. 

        compare_func should be a functions that calculates some score between w and a chunk of x
    """
    wl = len(w)
    minimum = sys.maxint
    average = 0
    frame = 0
    total = len(x) - wl
    if(total == 0):
        total = 1
    for i in range(len(x) - wl):
        diff = compare_func(w, x[i:(i+wl)])
        if diff < minimum:
            minimum = diff
            frame   = i
        average = average + diff
    average = average / total
    return frame, minimum, average

def cosine_similarity(x,y):
    x1 = np.squeeze(np.asarray(x))
    y1 = np.squeeze(np.asarray(y))
    dotprod = np.dot(x1,y1)
    cos_sim = dotprod/(np.linalg.norm(x1)*np.linalg.norm(y1))
    return cos_sim

def euclidean_norm_mean(x,y):
    x = np.mean(x, axis=0)
    y = np.mean(y, axis=0)
    return np.linalg.norm(x-y)

def euclidean_norm(x,y):
    return np.linalg.norm(x-y)


# Loop over all videos in the database and compare frame by frame
bestscore = sys.maxint
bestvideo = None
score = sys.maxint
for video in video_list:
    print video
    if get_duration(video) < q_duration:
        print get_duration(video), q_duration
        print 'Error: query is longer than database video'
        continue

    w = np.array(query_features)
    if args.f == features[0]: 
        x = search.get_colorhists_for(video)

        check = cv2.VideoCapture(video)
        testheight = check.get(cv2.CAP_PROP_FRAME_HEIGHT)
        testwidth = check.get(cv2.CAP_PROP_FRAME_WIDTH)
        check.release()
        print testwidth, testheight, videowidth, videoheight

        wres = testwidth * testheight
        xres = videowidth * videoheight
        w, x = normalize_histogram(wres, w, xres, x)
        frame, score, average = sliding_window(x,w, euclidean_norm_mean)
    elif args.f == features[1]:
        x = search.get_temporaldiffs_for(video)
        frame, score, average = sliding_window(x,w, euclidean_norm)
    elif args.f == features[2]:
        x = search.get_audiopowers_for(video)
        frame, score, average = sliding_window(x,w, euclidean_norm)
    elif args.f == features[3]:
        x = search.get_mfccs_for(video)
        #frame, score = sliding_window(x,w, euclidean_norm_mean)
        availableLength= min(x.shape[1],w.shape[1])
        frame, score, average = sliding_window(x[:,:availableLength,:],w[:,:availableLength,:], euclidean_norm_mean)
    elif args.f == features[4]:
        x = search.get_chdiffs_for(video)
        frame, score, average = sliding_window(x,w, euclidean_norm)

    if(score == 0):
        bestvideo = video
        bestscore = score
        break        
    if(score < bestscore):
        bestscore = score
        bestvideo = video
    print 'Best match at:', frame/frame_rate, 'seconds, with score of:', score, 'and an average of', average
    print ''
print 'best video match: ', bestvideo

 
