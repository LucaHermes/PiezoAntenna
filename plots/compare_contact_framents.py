import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np
import pandas as pd
import os
import re

plt.style.use('ggplot')

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


#plt.figure(num=None, figsize=(7, 6), dpi=80)
plt.subplot(2, 1, 1)
counter = 1
timeline = np.linspace(0, FRAGMENT_SIZE / 44.100, FRAGMENT_SIZE)
for speed in range(60, 150, 20): 
    data = contact_signals.loc[ (contact_signals['distance'] == PLT_DISTANCE) 
                               & (contact_signals['speed'] == speed) ].iloc[:, 2:99]
    mean_sig = np.mean(data.values.tolist(), axis=1)[0]
    std_sig = np.std(data.values.tolist(), axis=1)[0]    
    #ax = plt.subplot(9, 1, counter)
    plt.plot(timeline, mean_sig, label='Speed %d $^\circ$/s' % speed)
    #plt.fill_between(timeline, mean_sig-std_sig, mean_sig+std_sig, color='grey', alpha=.3)

    plt.ylabel("Amplitude, a.u.")
    plt.xticks([0.0, 10, 20, 30, 40], ' ')
    plt.title('Distance = 15 cm')
    counter += 1
    plt.legend()
plt.subplot(2, 1, 2)
for distance in range(9, 22, 4):
    data = contact_signals.loc[ (contact_signals['distance'] == distance) 
                               & (contact_signals['speed'] == PLT_SPEED) ].iloc[:, 2:99]
    mean_sig = np.mean(data.values.tolist(), axis=1)[0]
    std_sig = np.std(data.values.tolist(), axis=1)[0]
    
    #ax = plt.subplot(9, 1, counter)
    plt.plot(timeline, mean_sig, label='Dist %d cm' % distance)
    #plt.fill_between(timeline, mean_sig-std_sig, mean_sig+std_sig, color='grey', alpha=.3)
    plt.ylabel("Amplitude, a.u.")
    plt.title('Speed = 100 $^\circ$/s')
    plt.xlabel("Time, ms")
    counter += 1
    plt.legend()
plt.show()