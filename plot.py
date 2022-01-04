#
# modules
#
import matplotlib.pyplot as plt
import pandas as pd

#
# facedetector.py
#
import facedetector as fd

class PlotData:

    def __init__(self):
        print("Plotting...")
    
    def plot(self, data):
        
        # Eye aspect ratio
        plt.subplot(411)
        plt.plot(data['Elapsed'], data['EAR'], marker='.', label='EAR')
        plt.ylim(0, 0.5)
        plt.grid(True)
        plt.xlabel('Time (sec)')
        plt.ylabel('Eye Aspect Ratio')
        plt.legend()

        # Mouth aspect ratio
        plt.subplot(412)
        plt.plot(data['Elapsed'], data['MAR'], marker='.', label='MAR')
        plt.ylim(0,0.9)
        plt.grid(True)
        plt.xlabel('Time (sec)')
        plt.ylabel('Mouth Aspect Ratio')
        plt.legend()

        # Head movement in X and Y direction
        wnd = int(len(data.index)/data['Elapsed'].max())*5
        plt.subplot(413)
        plt.plot(data['Elapsed'], data['Face Mean X'].diff().rolling(wnd).mean(),'r--', label='dX')
        plt.plot(data['Elapsed'], data['Face Mean Y'].diff().rolling(wnd).mean(),'g--', label='dY')
        plt.ylim(-5.0, 5.0)
        plt.grid(True)
        plt.xlabel('Time (sec)')
        plt.ylabel('Head Movement')
        plt.legend()

        # Keyboard activity
        plt.subplot(414)
        plt.plot(data['Elapsed'], data['KBDyn'], marker='.', label='KBDyn')
        plt.ylim(0,1.2)
        plt.grid(True)
        plt.xlabel('Time (sec)')
        plt.ylabel('Keyboard Dynamics')
        plt.legend()

        plt.show()
