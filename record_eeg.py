import numpy as np  # Module that simplifies computations on matrices
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import bci_workshop_tools as BCIw  # Our own functions for the workshop
import pandas as pd
import pygame as pg
import time
import argparse
from threading import Thread


def play():
    pg.mixer.init()
    sound0 = pg.mixer.Sound("Airline-2020.wav")
    sound1 = pg.mixer.Sound('IVR.wav')
    channel0 = sound0.play()
    channel0.set_volume(1.0, 0.0)
    channel1 = sound1.play()
    channel1.set_volume(0.0, 1.0)
    time.sleep(60)
    sound0.stop()
    sound1.stop()
    
def acquire_data(index_channel,inlet,fs,file_name):    
    eeg_data, timestamp = inlet.pull_chunk(
                    timeout=61, max_samples=int(60*fs))
    eeg_data = np.array(eeg_data)[:, index_channel]
    np.save(file_name,eeg_data)
    
if __name__ == "__main__":

    """  PARSE ARGUMENTS """
    parser = argparse.ArgumentParser(description='BCI Workshop example 2')
    parser.add_argument('channels', metavar='N', type=int, nargs='*',
        default=[0, 1, 2, 3],
        help='channel number to use. If not specified, all the channels are used')

    args = parser.parse_args()
    
    """CONNECT TO EEG STREAM """

    # Search for active LSL stream
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info, description, sampling frequency, number of channels
    info = inlet.info()
    description = info.desc()
    fs = int(info.nominal_srate())
    n_channels = info.channel_count()
    
    # Get names of all channels
    ch = description.child('channels').first_child()
    ch_names = [ch.child_value('label')]
    for i in range(1, n_channels):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value('label'))

    index_channel = args.channels

    
    """ RECORD TRAINING DATA """
    
        
    print('\nFocus on left speaker!\n')
    
    Thread(target = play).start()
    Thread(target = acquire_data, args= (index_channel,inlet, fs,"mydata_left"+str(i)+".npy")).start()

    time.sleep(60)
    
    print('\nFocus on right speaker!\n')
    Thread(target = play).start()
    Thread(target = acquire_data, args= (index_channel,inlet, fs,"mydata_right"+str(i)+".npy")).start()

    time.sleep(60)
   
