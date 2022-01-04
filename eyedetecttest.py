# import
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

import mywebcam as wc
from facedetect import DetectFace

def grayscale1():
    gv = wc.GetVideo(0, fx=1, fy=1).start()
    sv = wc.ShowVideo(gv.frame).start()
    
    while True:
        sv.frame = gv.frame
        gray = cv2.cvtColor(sv.frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray', gray)
        if cv2.waitKey(1) == 32:
            break

    gv.stop()
    sv.stop()

def grayscale2():
    gv = wc.GetVideo(0, fx=1, fy=1).start()
    
    while True:
        frame = gv.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray', gray)
        if cv2.waitKey(1) == 32:
            break

    gv.stop()

def detectface1():
    gv = wc.GetVideo(0, fx=1, fy=1).start()
    sv = wc.ShowVideo(gv.frame).start()
    fd = DetectFace()
    df = pd.DataFrame(columns=['Elapsed','EAR','Face Mean X','Face Mean Y'])

    start_time = time.time()
    elapsed = 0
    while True:

        #read image from webcam
        frame = cv2.cvtColor(gv.frame, cv2.COLOR_BGR2GRAY)
        elapsed = time.time() - start_time

        #detect face
        detected, frame, ear, pos = fd.detect(frame, elapsed)
        if detected:
            print('%.1f sec - %.1f fps'%(elapsed, len(df.index)/df['Elapsed'].max()))
            df.loc[len(df.index)] = list([elapsed, ear, pos[0], pos[1]])

        #display
        sv.frame = frame

        if cv2.waitKey(1) == 32:
            break

    #
    # cap.release() & stop threading
    #
    gv.stop()
    sv.stop()

def detectface2():
    #

#
# Main
#