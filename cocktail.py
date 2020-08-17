from scipy.io.wavfile import read
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.ndimage import zoom
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn import svm
import argparse
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import bci_workshop_tools as BCIw  # Our own functions for the workshop
import pygame as pg
import time
from threading import Thread
import itertools


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b,a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def preprocess_audio(p):
   p = p.mean(axis=1)
   p = signal.resample(p, 44096*60)
   pc = signal.hilbert(p)
   u = abs(pc).reshape(-1,689).mean(axis=1)
   uf = butter_lowpass_filter(u,8,64)
   return uf

def preprocess_eeg(p):
   p = p.mean(axis=1)
   u = abs(p).reshape(-1,4).mean(axis=1)
   uf = butter_bandpass_filter(u,2,8,64)
   return uf

def cca_fit(X, Y):
    cca = CCA(n_components=1)
    cca.fit(X,Y)
    #X,Y = cca.transform(X,Y)
    
    X = list(itertools.islice(X, 10))
    Y = list(itertools.islice(Y, 10))
    #X= np.array(X).reshape(1,-1)
    #Y= np.array(Y)
    #return np.vstack((X,Y))
    #return np.corrcoef(X,Y)[1,0]
    return cca.score(X,Y)

def svm_fit(data,labels):
    clf = svm.SVC()
    clf.fit(data,labels)
    return clf

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
    
def acquire_data(inlet,fs):
    eeg_data = np.load("mydata.npy")
    eeg_data, timestamp = inlet.pull_chunk(
                    timeout=61, max_samples=int(60*fs))
    np.save("mydata.npy",eeg_data)
    




fs1,audio2 = read("IVR.wav")
fs2,audio1 = read("Airline-2020.wav")
eeg1 = np.load("mydata0.npy")
eeg2 = np.load("mydata1.npy")


Y_sc1 = preprocess_eeg(eeg1)
Y_sc2 = preprocess_eeg(eeg2)

X_sc1 = preprocess_audio(audio1).reshape(-1,1)
X_sc2 = preprocess_audio(audio2).reshape(-1,1)


left_attended = cca_fit(X_sc1, Y_sc1)
left_unattended = cca_fit(X_sc2, Y_sc1)
right_attended = cca_fit(X_sc1, Y_sc2)
right_unattended = cca_fit(X_sc2, Y_sc2)
print(left_attended)
print(left_unattended)
print(right_attended)
print(right_unattended)
dataset_left = np.vstack((left_attended,left_unattended))
dataset_right = np.vstack((right_attended,right_unattended))
labels = [1,0]

model_left = svm_fit(dataset_left ,labels)
model_right = svm_fit(dataset_right ,labels)



if __name__ == "__main__":

    """ PARSE ARGUMENTS """
    parser = argparse.ArgumentParser(description='BCI Workshop example 2')
    parser.add_argument('channels', metavar='N', type=int, nargs='*',
        default=[0, 1, 2, 3],
        help='channel number to use. If not specified, all the channels are used')

    args = parser.parse_args()

    """ CONNECT TO EEG STREAM """

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
   
    """  ACQUIRE DATA """
    Thread(target = play).start()
    Thread(target = acquire_data, args= (inlet, fs)).start()

    time.sleep(61)

    eeg_data = np.load("mydata.npy")
     # Only keep the channel we're interested in
    ch_data = np.array(eeg_data)[:, index_channel]
    eeg = preprocess_eeg(ch_data)
    attended = cca_fit(X_sc1,eeg)
    unattended = cca_fit(X_sc2,eeg)
    data = np.vstack((attended,unattended))
    result1 = model_left.predict(data)
    result2 = model_right.predict(data)

    print(attended)
    print(unattended)
    print(result1)
    print(result2)
              
            
 
