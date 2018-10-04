import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np
import pandas as pd
import os
import re

#plt.style.use('ggplot')

data_dir = '../data'
files = os.listdir(data_dir)
df = pd.DataFrame(columns=['record', 'distance', 'speed'])

def get_record(distance, speed):
    return df['record'][ (df['distance'] == distance) & (df['speed'] == speed) ].tolist()[0]


for f in files:
    if '.wav' not in f:
        continue
    fs, data = wav.read(os.path.join(data_dir, f))
    decoded_classes = [ int(s) for s in re.findall(r'-?\d+', f) ]
    # put the data into the DataFrame
    df = df.append(dict(
        { 'record'   : np.divide(data, float(np.max(data))), 
          'distance' : decoded_classes[0],
          'speed'    : decoded_classes[1] }), ignore_index=True)
    
DISTANCES = sorted(df.distance.unique())
SPEEDS = sorted(df.speed.unique())

EXPECTED_CONTACTS = 100
THRESHOLD = .6 #1844146018536
FRAGMENT_SIZE = 2048
SAMPLES_BETWEEN_CONTACTS = 8192
PLT_DISTANCE = 15
PLT_SPEED = 100

def find_peaks(x, method='last_peaks'):
    above_th = pd.Series(x) > THRESHOLD
    # find every threshold transgression
    above_th = pd.Series(above_th.index[above_th == True].tolist()).astype(int)
    diff = above_th.diff(periods=-1).abs()
    if method == 'first_peaks':
        diff = diff.shift(1).fillna(SAMPLES_BETWEEN_CONTACTS)
    peaks_indices = diff.index[diff >= SAMPLES_BETWEEN_CONTACTS].tolist()
    peaks_indices.append(len(above_th) - 1)
    peaks = above_th[peaks_indices].reset_index(drop=True)
    return peaks

def extract_peaks(d, peaks):
    idx_peaks = pd.Index(peaks)
    contacts = []
    for c_idx in peaks:
        contacts.append(d[c_idx:c_idx + FRAGMENT_SIZE].tolist())
    return contacts

contact_signals = pd.DataFrame()

for i in range(len(df)):
    speed = df.at[i, 'speed']
    distance = df.at[i, 'distance']
    d = pd.Series(df.at[i, 'record'])
    peaks = find_peaks(d.abs())
    if len(peaks) != EXPECTED_CONTACTS:
        print('Unexpected amount of contacts (%d) in set distance %d - speed %d' 
              % (len(peaks), distance, speed))
    contacts = extract_peaks(d, peaks)
    contact_signals = contact_signals.append([[speed, distance] + contacts], ignore_index=True)
    
contact_signals.rename(columns={ 0 : 'speed', 1 : 'distance' }, inplace=True)
last_col = contact_signals.columns[-1]
contact_signals.rename(columns=dict(zip(range(2, last_col), range(0, last_col-2))), inplace=True)
contact_signals.rename(columns={ 101 : 99 }, inplace=True)

# TRANSFORM VIA FFT

FRAMERATE = 44100.
SEGMENT_LEN = 1024

def generate_spectrum(data):
    return signal.welch(data,
               fs=FRAMERATE,
               window='hann',
               nperseg=SEGMENT_LEN,
               nfft=SEGMENT_LEN*2)


psds = pd.DataFrame()
clean_contact_data = contact_signals.iloc[:, :101]

for i in range(len(df)):
    speed = clean_contact_data.at[i, 'speed']
    distance = clean_contact_data.at[i, 'distance']
    contact_data = clean_contact_data.iloc[i, 2:].values.tolist()
    # generate spectra for all every contact of this speed and distance
    f, Pxx = generate_spectrum(contact_data)
    Pxx = 10 * np.log10(Pxx)
    Pxx = [ np.append([distance, speed], pxx) for pxx in Pxx ]
    psds = psds.append(Pxx)

psds.columns = np.append(['distance', 'speed'], np.array(f, dtype=np.float))



################ SWITCH ####################
PLOT_DISTANCES = False


# PLOT PSDs
if PLOT_DISTANCES:
    plt.figure(num=None, figsize=(10, 20), dpi=80)
    plt.subplots_adjust(hspace=0, left=0.1, right=0.95, top=0.95, bottom=0.01)
    counter = 1

    plt.style.use('default')

    for dist in psds.distance.sort_values().unique():    
        dat = psds[psds.distance == dist].iloc[:, 2:].values.tolist()
        average = np.mean(dat, axis=0)
        std_dev = np.std(dat, axis=0)
        ax = plt.subplot(10, 1, counter)
        plt.xscale('log')
        counter += 1
        
        plt.plot(f, average, cmap=plt.get_cmap('plasma') ,label='distance %d cm' % dist)
        plt.fill_between(f, average-std_dev, average+std_dev, color='grey', alpha=.3)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if counter < 10:
            ax.spines['bottom'].set_visible(False)
            ax.get_xaxis().set_visible(False)
        plt.legend()
        
    plt.legend()
    plt.xlabel("Frequency [log(Hz)]")
    plt.ylabel("Frequency power")
    plt.show()
else:
    plt.figure(num=None, figsize=(10, 10), dpi=80)
    plt.subplots_adjust(hspace=0, left=0.1, right=0.95, top=0.95, bottom=0.1)
    counter = 1

    for speed in psds.speed.sort_values().unique():    
        dat = psds[psds.speed == speed].iloc[:, 2:].values.tolist()
        average = np.mean(dat, axis=0)
        std_dev = np.std(dat, axis=0)
        
        ax = plt.subplot(7, 1, counter)
        plt.xscale('log')
        counter += 1
        plt.plot(f, average, label='speed %d' % speed)
        plt.fill_between(f, average-std_dev, average+std_dev, color='grey', alpha=.3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if counter < 7:
            ax.spines['bottom'].set_visible(False)
            ax.get_xaxis().set_visible(False)
        plt.legend()
        
    plt.legend()
    plt.xlabel("Frequency [log(Hz)]")
    plt.ylabel("Frequency power")
    plt.show()