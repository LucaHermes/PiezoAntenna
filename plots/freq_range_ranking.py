import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import numpy as np

plt.style.use('ggplot')
fig, ax = plt.subplots()#2, 1, 1)

model_scores = pd.read_csv('../data/analysis/model_scores.csv')
model_scores = model_scores.dropna()

f_hi, f_lo = model_scores.loc[:, 'freqhi_actual'], model_scores.loc[:, 'freqlo_actual']
f_sv = model_scores.loc[:, 'fsv']
r2 = model_scores.loc[:, 'r2']

f_hi_sorted = [ f for _, f in sorted(zip(r2, f_hi)) ]
f_lo_sorted = [ f for _, f in sorted(zip(r2, f_lo)) ]
f_sv_sorted = [ f for _, f in sorted(zip(r2, f_sv)) ]

idx = np.unique(zip(f_lo_sorted, f_hi_sorted), axis=0, return_index=True)[1]
f_hi_clean = [ f_hi_sorted[i] for i in sorted(idx) ]
f_lo_clean = [ f_lo_sorted[i] for i in sorted(idx) ]

PLT_WIDTH = 22050.
PLT_HEIGHT = 50

for i in range(1, PLT_HEIGHT+2):
	hi_frac = f_hi_clean[-i]# / PLT_WIDTH
	lo_frac = f_lo_clean[-i]+.1# / PLT_WIDTH
	print(lo_frac)
	plt.hlines(i, lo_frac, hi_frac, linewidth=5., color=[cm.binary(f_sv_sorted[-i]+.2)])       #, alpha=f_sv_sorted[-i])

plt.ylim((50, 0))
plt.xscale('log')
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xlabel("Frequency, Hz")
plt.xlim((0.1, 22050))
plt.ylabel("Rank")

plt.show()