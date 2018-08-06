import sys
import numpy as np
import scipy.io as spio

class parser:
    
    filepath = '../tmseegData.mat'
    start = 9990
    end = 10090

    ''' 
        Loads the .mat files for the melon data and the tms-eeg data
        MSO: Monophasic TMS-EEG data have stimulation intensities range from 
             10 to 80 (increments of 10)
        channel: There are 63 channels, tagged as (0,1,..62)
        start: index of the first sample. Samples before start are truncated
        end: index of the first sample. Samples after start are truncated
    '''
    def __init__(self, MSO, channel):
        self.MSO = 'MSO%d'%MSO
        self.channel = channel
        try:
            self.eeg_data = spio.loadmat(parser.filepath, squeeze_me=True)
            print("TMS-EEG data is successfully loaded.")
        except Exception:
            print("Sorry, tmseegData.mat does not exist.")
            sys.exit(1)  

    def get_intensity(self):
        self.intensity_data = self.eeg_data[self.MSO]
    
    def get_channel(self):
        self.channel_data = self.intensity_data[self.channel, 
                                                parser.start:parser.end, :]
               
    # Used as an alternative constructor. 
    # Returns the tms-eeg dataset for a different MSO intensity.
    @classmethod
    def from_intensity(cls, MSO):
        cls.MSO = MSO
        return cls
    
    # Used as an alternative constructor. 
    # Returns the tms-eeg dataset for a different MSO intensity.
    @classmethod
    def from_channel(cls, channel):
        cls.channel = channel
        return cls

