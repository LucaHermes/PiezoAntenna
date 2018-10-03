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
        { 'record'   : np.divide(data, float(np.max(data))), 
          'distance' : decoded_classes[0],
          'speed'    : decoded_classes[1] }), ignore_index=True)
    
DISTANCES = sorted(df.distance.unique())
SPEEDS = sorted(df.speed.unique())

plt.figure(num=None, figsize=(15, 8), dpi=80)
plt.subplots_adjust(wspace=0.1)
plt.style.use('ggplot')

PLT_DISTANCE = 15
PLT_SPEED = 100

# plot for varying speed
for speed in range(60, 150, 40):
    data = get_record(PLT_DISTANCE, speed)
    plt.subplot(3, 2, int(speed / 40) * 2)
    timeline = np.linspace(0, len(data) / 44100., len(data))
    plt.plot(timeline, data, color='k', linewidth=.5)
    plt.xlim((20.75, 23.25))
    plt.ylim((-1, 1))
    plt.yticks([-1, -.5, 0, .5, 1], ' ')
    plt.title('Distance %d, speed %d' % (PLT_DISTANCE, speed))
    if speed < 140:
    	plt.xticks(np.linspace(21, 23, 5), ' ')
plt.xlabel("Time [sec]")
    
# plot for varying distance
for distance in [7, 17, 21]:
    data = get_record(distance, PLT_SPEED)
    plt.subplot(3, 2, int(distance / 7) * 2 - 1)
    timeline = np.linspace(0, len(data) / 44100., len(data))
    plt.plot(timeline, data, linewidth=.5, color='b')
    plt.xlim((20.75, 23.25))
    plt.ylim((-1, 1))
    plt.title('Distance %d, speed %d' % (distance, PLT_SPEED))
    plt.ylabel("Amplitude a.u.")
    if distance < 21:
    	plt.xticks(np.linspace(21, 23, 5), ' ')

plt.xlabel("Time [sec]")

plt.show()