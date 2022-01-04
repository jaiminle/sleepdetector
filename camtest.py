# import modules
import cv2
import time
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import classes
import mywebcam as wc
from facedetect2 import DetectFace2


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def test1():
    cam = cv2.VideoCapture(0)
    nf = 0
    startTime = time.time()
    while True:
        (grabbed, frame) = cam.read()
        nf = nf+1
        frame = cv2.flip(frame, 1)
        cv2.imshow("camtest", frame)
        if cv2.waitKey(1) == 32:
            break
        elapsedTime = time.time() - startTime
        fps = nf / elapsedTime
        print('t=%.2f sec fps=%.1f'%(elapsedTime,fps))


def test2():
    gv = wc.GetVideo(0, fx=1, fy=1).start()
    sv = wc.ShowVideo(gv.frame).start()

    startTime = time.time()
    while True:
        sv.frame = gv.frame
        if sv.stopped:
            break
        elapsedTime = time.time() - startTime
        fps = min(gv.tick, sv.tick) /max(elapsedTime, 0.0001)
        print('t=%.2f sec fps=%.1f'%(elapsedTime,fps))

    sv.stop()
    gv.stop()

def test3():
    gv = wc.GetVideo(0, fx=1, fy=1).start()
    sv = wc.ShowVideo(gv.frame).start()
    faceDetector = dlib.get_frontal_face_detector()
    landmarkPredictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    df = pd.DataFrame(columns=['Elapsed','EAR','Face Mean X','Face Mean Y'])
    fd = DetectFace2()

    startTime = time.time()
    while True:
        frame = gv.frame
        #
        #
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.putText(frame, "abc", (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        rects = faceDetector(gray, 1)
        for rect in rects:
            shape = landmarkPredictor(gray, rect)
            shape = shape_to_np(shape)
            left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
            cv2.rectangle(frame,(left,top),(right,bottom),(0,255, 0),2)
            for (x,y) in shape:
                cv2.circle(frame,(x,y),1,(0,0,255),-1)

        elapsedTime = time.time() - startTime
        detected, frame, ear, pos, size = fd.detect(frame, elapsedTime)
        fps = min(gv.tick, sv.tick)/max(elapsedTime, 0.0001)
        if detected:
            print('t=%.2f sec fps=%.1f'%(elapsedTime,fps))
            df.loc[len(df.index)] = list([elapsedTime, ear, pos[0], pos[1]])
            print('size=%.2f'%(size))

        #
        #
        
        sv.frame = frame
        
        if sv.stopped:
            break   

    sv.stop()
    gv.stop()


def test4():
    # start collecting frames
    gv = wc.GetVideo(0, fx=1, fy=1).start()
    # returning frames
    sv = wc.ShowVideo(gv.frame).start()
    # built in face detector module from dlib
    faceDetector = dlib.get_frontal_face_detector()
    # landmark detector file
    landmarkPredictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # collected data
    df = pd.DataFrame(columns=['Elapsed','EAR','Face Mean X','Face Mean Y', 'Mouth Size'])
    # imported class
    fd = DetectFace2()

    # collecting time
    startTime = time.time()
    # loop for collecting and showing frames
    while True:
        # receive frame
        frame = gv.frame
        # convert frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.putText(frame, "abc", (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # drawing rectangles on gray scale frame (can only face detect on gray scale)
        rects = faceDetector(gray, 1)
        # going through rectangles to determine shape within rectangle
        for rect in rects:
            # predicting landmark in rectangle
            shape = landmarkPredictor(gray, rect)
            # predicted landmark converted to shape (68 xy points converted to NumPy array for easier utilization)
            shape = shape_to_np(shape)
            # converting detection by dlib to (x,y,w,h) which can be used for OpenCV
            left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
            # drawing rectangles on frame from detection with dlib
            cv2.rectangle(frame,(left,top),(right,bottom),(0,255, 0),2)
            # drawing points over facial landmarks
            for (x,y) in shape:
                cv2.circle(frame,(x,y),1,(0,0,255),-1)

        elapsedTime = time.time() - startTime
        # detected = boolean, captured frame, calculated EAR, position
        detected, frame, ear, pos, size = fd.detect(frame, elapsedTime)
        # calculating frames per second
        fps = min(gv.tick, sv.tick) /max(elapsedTime, 0.0001)
        # if detected boolean is true, print elapsedTime and frames per second (fps)
        if detected:
            print('t=%.2f sec fps=%.1f'%(elapsedTime,fps))
            # adding data into data frame array
            df.loc[len(df.index)] = list([elapsedTime, ear, pos[0], pos[1], size])

        #
        #
        
        # frame to show is the frame current frame from loop
        sv.frame = frame
        
        # stop showing frame
        if sv.stopped:
            break   
    
    # stop running getting and showing frames after exiting loop
    sv.stop()
    gv.stop()

    # converting information from data frame
    df.to_pickle("./data.pkl")
    
    #
    # plotting data
    #
    print("Plotting ...")
    #
    # eye detection
    #
    #
    #wnd = int(len(df.index)/df['Elapsed'].max())*5
    #df['mavgEAR']=df['EAR'].rolling(wnd).mean()
    #plt.figure(figsize=(8,4))
    plt.subplot(211)
    plt.plot(df['Elapsed'], df['EAR'], marker='.', label='EAR')
    #plt.plot(df['Elapsed'], df['mavgEAR'],'r--', label='EAR_MAVG')
    plt.ylim(0, 0.5)
    plt.grid(True)
    plt.xlabel('Time (sec)')
    plt.ylabel('Eye Area Ratio')
    plt.legend()

    #wnd = int(len(df.index)/df['Elapsed'].max())*5
    plt.subplot(223)
    plt.plot(df['Elapsed'], df['Mouth Size'], marker='.', label='Mouth Size')
    plt.ylim(0.6, 1.0)
    plt.grid(True)
    plt.xlabel('Time (sec)')
    plt.ylabel('Mouth Size')
    plt.legend()

    #
    # head detection
    #
    plt.subplot(224)
    #plt.plot(df['Elapsed'], df['Face Mean X'],'.', label='X')
    #plt.plot(df['Elapsed'], df['Face Mean Y'],'.', label='Y')
    #plt.plot(df['Elapsed'], df['Face Mean X'].diff(),'r', marker='.', label='dX')
    #plt.plot(df['Elapsed'], df['Face Mean Y'].diff(),'g', marker='.', label='dY')
    plt.plot(df['Elapsed'], df['Face Mean X'].diff().rolling(wnd).mean(),'r--', label='dX')
    plt.plot(df['Elapsed'], df['Face Mean Y'].diff().rolling(wnd).mean(),'g--', label='dY')
    plt.ylim(-0.8, 0.8)
    plt.grid(True)
    plt.xlabel('Time (sec)')
    plt.ylabel('Head Position')
    plt.legend()

    #
    # mouth detection
    #

    plt.show()


#
# Main
#
test4()