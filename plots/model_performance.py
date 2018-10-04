import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

#plt.figure(num=None, figsize=(15, 10), dpi=80)
plt.style.use('ggplot')
fig, ax = plt.subplots()#2, 1, 1)

model_scores = pd.read_csv('../data/analysis/model_scores.csv')
model_scores = model_scores.dropna()


############################# remove singular rows of values that only occur once in column freqmid_actual
#model_scores = model_scores[model_scores.groupby('freqmid_actual').freqmid_actual.transform(len) > 1]

f_center = model_scores.loc[:, 'freqmid_actual']
frac_width = (model_scores.loc[:, 'freqhi_actual'] - model_scores.loc[:, 'freqlo_actual']) / f_center
r2 = model_scores.loc[:, 'r2']

# find best performing
sorted_r2 = [ r for _, r in sorted(zip(f_center, r2))]
sorted_frac_width = [ r for _, r in sorted(zip(f_center, frac_width))]
sorted_center = sorted(f_center)
top = []
highest = sorted_r2[0]
xs = []
for i in range(1, len(sorted_center)):
    if sorted_center[i] == sorted_center[i-1]:
        highest = max(highest, sorted_r2[i], sorted_r2[i-1])
    else:
        top.append(highest)
        highest = sorted_r2[i]
top.append(highest)

p = plt.scatter(f_center, r2, marker='.', c=100*frac_width, cmap='rainbow')
plt.plot(np.unique(sorted_center), top, color='k')
plt.axhline(0.95, linestyle='--', color='k', linewidth=0.8)
plt.xscale('log')
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xlabel('Center frequency, Hz')
plt.ylabel(r'Coefficient of determination, $R^2$')

cbar_ax = fig.add_axes([0.91, 0.1, 0.01, 0.5])
cbar = fig.colorbar(p, cax=cbar_ax)
cbar.set_label('Fractional bandwidth, %')




"""

frac_supp_vec = model_scores.loc[:, 'fsv']
frac_supp_vec_sorted = [ r for _, r in sorted(zip(f_center, frac_supp_vec))]
top = []
highest = sorted_r2[0]
f = frac_supp_vec_sorted[0]

for i in range(1, len(sorted_center)):
    if sorted_center[i] == sorted_center[i-1]:
        if highest == max(highest, sorted_r2[i], sorted_r2[i-1]):
            continue
        if  sorted_r2[i] >= sorted_r2[i-1]:
            f = frac_supp_vec_sorted[i]
            highest = sorted_r2[i]
        else:
            f = frac_supp_vec_sorted[i-1]
            highest = sorted_r2[i-1]
    else:
        top.append(f)
        highest = 0
        f = frac_supp_vec_sorted[i]

top.append(highest)


ax = plt.subplot(2, 1, 2)
plt.scatter(f_center, frac_supp_vec, marker='.', c=100*frac_width, cmap='rainbow')
plt.plot(np.unique(sorted_center), top, color='k')
plt.xscale('log')
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())


"""
plt.show()