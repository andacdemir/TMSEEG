import sys
import numpy as np
import scipy.io as spio

class parser():
    ''' 
        Loads the .mat files for the melon data and the tms-eeg data
        MSO: Monophasic TMS-EEG data have stimulation intensities range from 
             10 to 80 (increments of 10)
        channel: There are 63 channels, tagged as (0,1,..62)
        start: index of the first sample. Samples before start are truncated
        end: index of the first sample. Samples after start are truncated
    '''
    def __init__(self, MSO, channel, start, end):
        self.MSO = 'MSO%d'%MSO
        self.channel = channel
        self.start = start
        self.end = end
        try:
            self.eeg_data = spio.loadmat('../tmseegData.mat', 
                                         squeeze_me=True)
            print("TMS-EEG data is successfully loaded.")
        except Exception:
            print("Sorry, tmseegData.mat does not exist.")
            sys.exit(1)  

    def get_intensity(self):
        self.intensity_data = self.eeg_data[self.MSO]
    
    def get_channel(self):
        self.channel_data = self.intensity_data[self.channel, 
                                                self.start:self.end, :]
               
 
