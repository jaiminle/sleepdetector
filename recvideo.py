import os
import time
import numpy as np
import cv2
from datetime import datetime

# parameters
Width = 320
Height = 240
FileStr = 'sleeping'
Duration = 30 #seconds
Wait = 30     #seconds
NR = 10        #number of recordings


# create videocapture object
cap = cv2.VideoCapture(0)
if(cap.isOpened() == False):
    print("Unable to open web camera")
fourcc = cv2.VideoWriter_fourcc(*'XVID')

path = os.path.join('.\\', 'video')
if not os.path.isdir(path):
    os.mkdir(path)

for n in range(NR):
    #fname = 'sleeping' + '{:04d}'.format(n) + '.avi'
    fname = FileStr + datetime.now().strftime("%Y%m%d-%H%M%S") + '.avi'
    fname = os.path.join(path,fname)
    
    out = cv2.VideoWriter(fname, fourcc, 28, (Width, Height))

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, (Width, Height), interpolation=cv2.INTER_AREA)
        if ret == True:
            out.write(frame)
            cv2.imshow(fname,frame)
            cv2.waitKey(1)

        elapsed = time.time() - start_time      
        if elapsed > Duration:
            print(elapsed)
            break
    cv2.destroyAllWindows()
    print("%d/%d - %s"%(n+1,NR, fname))
    time.sleep(Wait)

cap.release()
out.release()
cv2.destroyAllWindows()