#
# Modules
#

from threading import Thread
import cv2
import playsound

#
# Get Video
#
class GetVideo:
    """
    Class that continuously get frames
    """
    def __init__(self, src=0,fx=0.5,fy=0.5):
        self.fx, self.fy = fx, fy
        self.tick = 0
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, frame) = self.stream.read()
        frame = cv2.resize(frame, None, fx=self.fx, fy=self.fy, interpolation=cv2.INTER_AREA)
        self.frame = cv2.flip(frame,1)
        self.stopped = False
        print("GetVideo Initialized!")
    
    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, frame) = self.stream.read()
                frame = cv2.resize(frame, None, fx=self.fx, fy=self.fy, interpolation=cv2.INTER_AREA)
                self.frame = cv2.flip(frame,1)
                self.tick = self.tick + 1

    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()                            

# End of GetVideo

#
# Show Video
#
class ShowVideo:
    """
    Class that continuously shows a frame
    """
    def __init__(self, frame=None):
        self.frame = frame
        self.tick = 0
        self.stopped = False
        print("ShowVideo Initialized!")

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Drowsiness Detection System 1.0", self.frame)
            self.tick = self.tick + 1
            if cv2.waitKey(1) == 32:
                self.stopped = True

    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()                                    

# End of ShowVideo


#
# Play Alarm
#
class PlayAlarm:
    """
    Class that continuously shows a frame
    """
    def __init__(self, fname='alarm.wav'):
        self.fname = fname
        self.alarm = False
        self.stopped = False
        print("PlayAlarm Initialized!")

    def start(self):
        Thread(target=self.play, args=()).start()
        return self

    def play(self):
        while not self.stopped:
            if self.alarm:
                playsound.playsound(self.fname)


    def stop(self):
        self.stopped = True                                

# End of PlayAlarm




