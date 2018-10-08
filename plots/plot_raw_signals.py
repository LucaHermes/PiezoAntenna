# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np
import pandas as pd
import os
import re



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
        { 'record'   : np.divide(data, float(np.max(data)))[:220500], 
          'distance' : decoded_classes[0],
          'speed'    : decoded_classes[1] }), ignore_index=True)
    
DISTANCES = sorted(df.distance.unique())
SPEEDS = sorted(df.speed.unique())


# ------------------- FIND PEAKS ----------------------------

EXPECTED_CONTACTS = 100
THRESHOLD = .6 #1844146018536
FRAGMENT_SIZE = 2048
SAMPLES_BETWEEN_CONTACTS = 8192

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
    return peaks, contacts


contact_signals = pd.DataFrame()

for i in range(len(df)):
    speed = df.at[i, 'speed']
    distance = df.at[i, 'distance']
    d = pd.Series(df.at[i, 'record'])
    peaks = find_peaks(d.abs())
    if len(peaks) != EXPECTED_CONTACTS:
        print('Unexpected amount of contacts (%d) in set distance %d - speed %d' 
              % (len(peaks), distance, speed))
    x_ix, contacts = extract_peaks(d, peaks)
    contact_signals = contact_signals.append([[speed, distance, x_ix] + contacts], ignore_index=True)
    
contact_signals.rename(columns={ 0 : 'speed', 1 : 'distance', 2 : 'x_ix' }, inplace=True)
last_col = contact_signals.columns[-1]
contact_signals.rename(columns=dict(zip(range(3, last_col), range(0, last_col-2))), inplace=True)
contact_signals.rename(columns={ 101 : 99 }, inplace=True)
cs = contact_signals.iloc[:, :6]

# -------------------- PLOT DATA ----------------------------

plt.figure(num=None, figsize=(15, 8), dpi=80)
plt.subplots_adjust(wspace=0.1)
plt.style.use('ggplot')

PLT_DISTANCE = 15
PLT_SPEED = 100
plot_at = 1

# plot for varying distance
for distance in range(5, 24, 2):
    # plot for varying speed
    for speed in range(60, 190, 20):
    	plt.subplot(10, 7, plot_at)
        if distance == 23:
            plt.xlabel("Time, sec")
        else:
            plt.xticks(np.arange(0, 5, 0.2), ' ')
        if distance == 5:
            plt.title('%d $^\circ$/s' % speed)
        if speed == 60:
            plt.ylabel('%d cm' % distance, fontsize=15, labelpad=25, rotation=0)
        else:
            plt.yticks([-1, 0, 1], ' ')

        data = get_record(distance, speed)
        #print( cs[cs.distance == distance][cs.speed == speed])
        fragm = cs[cs.distance == distance][cs.speed == speed].iloc[0, 2:]
        x_ix, cf = fragm['x_ix'][1], fragm[1]
        timeline = np.linspace(0, len(data) / 44100., len(data))
        frag_time = np.linspace(x_ix / 44100., (x_ix + len(cf)) / 44100., len(cf))
        plt.plot(frag_time, cf, color='k')
        plt.plot(timeline, data, color='k', linewidth=.5, alpha=.5)
        plt.plot()
        #plt.xlim((2.1, 2.3))
        center_t = frag_time[0] + 0.5*(frag_time[-1] - frag_time[0])
        print(center_t)
        print(frag_time)
        low_t = center_t - 0.08
        hi_t = center_t + 0.08
        plt.xlim((low_t, hi_t))

        plt.ylim((-1, 1))
        plot_at += 1

plt.show()
