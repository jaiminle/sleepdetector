# 
# modules
# 
import keyboard
import time
import msvcrt

#
# kb.py
#
class kb:

    def __init__(self):
        self.tick = 0
        self.startTime = time.time()
        self.elapsed = 0
        self.kbRate = 0

        self.stopped = False

    def kbTest(self):
        while not self.stopped:
            if msvcrt.kbhit():
                tick = tick + 1
            
            self.elapsed = time.time() - self.startTime

            self.kbRate = tick / self.elapsed

        return self.kbRate

    def stop(self):
        self.stopped = True