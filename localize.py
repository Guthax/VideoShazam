import argparse
import numpy as np
import cv2
import glob
from scipy.io import wavfile
 


def correctAspectRatio(box):        
    sortedList = sorted(box, key=lambda point: point[1])
    tops = sorted(sortedList[:2], key=lambda point: point[0])
    topLeft = tops[0]
    topRight = tops[1]
    bottoms = sorted(sortedList[2:4], key=lambda point: point[0])
    bottomLeft = bottoms[0]
    bottomRight = bottoms[1]
    res = box
    res[0] = topLeft
    res[1] = topRight
    res[2] = bottomLeft
    res[3] = bottomRight
          
    width = (abs(res[0][0] - res[1][0]) + abs(res[3][0] - res[2][0]))/2
    height = int(round(width*0.5625))
    res[0][1] = res[3][1] + height
    res[1][1] = res[2][1] + height

    return res

def crop(frame, topLeft, bottomRight):
    return frame[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]
    
def findCorners(box, w, h):
    sortedList = sorted(box, key=lambda point: point[1])
    tops = sorted(sortedList[:2], key=lambda point: point[0])
    topLeft = tops[0]
    bottoms = sorted(sortedList[2:4], key=lambda point: point[0])
    bottomRight = bottoms[1]

    #correcting for values outside range
    topLeft[0] = max(0, topLeft[0])
    topLeft[1] = max(0, topLeft[1]) 

    return topLeft, bottomRight

def localizeVideo(video, output):
    print("localizing...")
  
    # Read input video
    cap = cv2.VideoCapture(video)
    cap2 = cv2.VideoCapture(video)  
     
    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
     
    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
     


    # Read first frame
    _, prev = cap.read()

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray,(5,5),0)

    # Create temporal diff vector
    optical_flow = []

    for i in range(n_frames/4): 
        success, curr = cap.read()  
        if not success: 
            break 
        flow = None
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0)
        CC = cv2.split(flow)
        avg = np.abs(np.divide(np.add(CC[0], CC[1]), 2))
        optical_flow.append(avg)
        prev_gray = curr_gray

    #generating and binarizing average optical flow image
    image = np.average(optical_flow, 0)
    av = np.average(image)
    image = np.array([[255 if x>av else 0 for x in line] for line in image])
    image = np.ascontiguousarray(image, dtype=np.uint8)
    image = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    image = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    image = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    #finding contours and creating bounding boxes
    img, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        bboxes.append(box)

    #finding largest box
    largestArea = 0
    boxIndex = 0
    i = 0
    for box in bboxes:
        area = cv2.contourArea(box)
        if area > largestArea:
            largestArea = area
            boxIndex = i
        i = i+1        
        
    bigBox = correctAspectRatio(bboxes[boxIndex])
    topLeft, bottomRight = findCorners(bigBox, w, h)

    sucess, rawFrame = cap2.read()
    croppedFrame = crop(rawFrame, topLeft, bottomRight)

    fshape = croppedFrame.shape
    fheight = fshape[0]
    fwidth = fshape[1]

    out = cv2.VideoWriter(output, fourcc, fps, (fwidth, fheight))
    out.write(croppedFrame)

    while sucess:   
        sucess, rawFrame = cap2.read()
        if sucess:
            croppedFrame = crop(rawFrame, topLeft, bottomRight)
            #cv2.imshow("cropped", croppedFrame)
            #cv2.waitKey(1)
            out.write(croppedFrame)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.drawContours(image, [bigBox], -1, (0, 0, 255), 5)
    #cv2.imshow("avg flow", image)
    #cv2.waitKey(0)
    out.release()
    cv2.destroyAllWindows()



#for i in range(n_frames-2): 
#    success, curr = cap.read()
#    success, curr = cap.read()  
#    if not success: 
#        break 
#    flow = None
#    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
#    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0)
#    CC = cv2.split(flow)
#    cv2.imshow("flow", np.divide(np.add(CC[0], CC[1]), 2))
#    cv2.waitKey(1)
#    optical_flow.append(flow)
#    prev_gray = curr_gray
 
