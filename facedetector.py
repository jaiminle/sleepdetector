#
# Modules
#
import numpy as np
from imutils import face_utils
import dlib
import cv2
from scipy.spatial import distance as dist

from keyboard import kb


#
# utility functions
#
def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def bb_to_rect(x,y,w,h):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
    rect = dlib.rectangle(np.long(x),np.long(y),np.long(x+w),np.long(y+h))

    return rect


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
 
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])

    D = dist.euclidean(mouth[0], mouth[4])

    msize = (A + B + C) / (3.0 * D)

    return msize

#
# CLASS - Detect Face
#

class DetectFace:

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        (self.eyeLs, self.eyeLe) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.eyeRs, self.eyeRe) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.mouths, self.mouthe = 60, 68

        print("DectectFace Initialized!")

    def detect(self,frame,elapsed, eyeplot=True, boxplot=False):
        hght, wdth, ch = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        detected = False
        ear, pos, mar = 0.0, 0.0, 0.0
        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)

            shape = shape_to_np(shape)
            pos = np.mean(shape,axis=0)

            leftEye = shape[self.eyeLs:self.eyeLe]
            rightEye = shape[self.eyeRs:self.eyeRe]
            mouth = shape[self.mouths:self.mouthe]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR)/2.0 
            
            mar = mouth_aspect_ratio(mouth)

            # add face marks
            if eyeplot:
                #leftEyeHull = cv2.convexHull(leftEye)
                #rightEyeHull = cv2.convexHull(rightEye)
                #cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1) 
                #cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                #cv2.putText(frame, "ear=%.2f et=%.1fs"%(ear, elapsed), (int(wdth/3), 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                for (x, y) in leftEye:
                    cv2.circle(frame,(x,y),1,(0,0,255),-1)
                for (x, y) in rightEye:
                    cv2.circle(frame,(x,y),1,(0,0,255),-1)
                for (x, y) in mouth:
                    cv2.circle(frame,(x,y),1,(0,0,255),-1)

            if boxplot:
                cv2.rectangle(frame,(rect.left(),rect.top()),(rect.right(), rect.bottom()),(0,255, 0),1)

            detected = True

        return (detected, frame, ear, pos, mar)