import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mlp
import numpy as np

plt.style.use('ggplot')
fig, ax = plt.subplots()#2, 1, 1)

model_scores = pd.read_csv('../data/analysis/model_scores.csv')
model_scores = model_scores.dropna()

f_hi, f_lo = model_scores.loc[:, 'freqhi_actual'], model_scores.loc[:, 'freqlo_actual']
f_sv = model_scores.loc[:, 'fsv']
r2 = model_scores.loc[:, 'r2']

f_hi_sorted = [ f for _, f in sorted(zip(r2, f_hi), reverse=True) ]
f_lo_sorted = [ f for _, f in sorted(zip(r2, f_lo), reverse=True) ]
f_sv_sorted = [ f for _, f in sorted(zip(r2, f_sv), reverse=True) ]

flo = []
fhi = []
fsv = []
r2_ = []

for i in range(len(f_lo_sorted)):
	if f_lo_sorted[i] != 0:
		flo.append(f_lo_sorted[i])
		fhi.append(f_hi_sorted[i])
		fsv.append(f_sv_sorted[i])
		r2_.append(sorted(r2, reverse=True)[i])

idx = np.unique(zip(flo, fhi), axis=0, return_index=True)[1]

f_hi_clean = [ fhi[i] for i in sorted(idx) ]
f_lo_clean = [ flo[i] for i in sorted(idx) ]

PLT_WIDTH = 22050.
PLT_HEIGHT = 70
PLOT_95 = True
for i in range(1, PLT_HEIGHT):
	hi_frac = f_hi_clean[i]# / PLT_WIDTH
	lo_frac = f_lo_clean[i]+.1# / PLT_WIDTH
	if r2_[i] < 0.95 and PLOT_95:
		plt.axhline(i, linestyle='--', color='k')
		PLOT_95 = False
	print("append %.3f" % r2_[i])
	plt.hlines(i, lo_frac, hi_frac, linewidth=5., color=[cm.binary(fsv[i]+.1)])       #, alpha=f_sv_sorted[-i])


plt.ylim((PLT_HEIGHT+1, 0))
plt.xscale('log')
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xlabel("Frequency, Hz")
plt.xlim((0.1, 22050))
plt.ylabel("Rank")

ax1 = fig.add_axes([0.91, 0.11, 0.01, 0.4])
cmap = mlp.cm.binary
norm = mlp.colors.Normalize(vmin=0, vmax=100)
cb = mlp.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, ticks=range(10, 105, 10))
cb.set_label('Fraction of support vectors, %')
plt.show()