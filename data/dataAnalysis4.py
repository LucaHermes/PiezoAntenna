import numpy            as np
import pandas           as pd
import scipy.signal     as signal
import scipy.stats      as stats
import scipy.io.wavfile as wav
import argparse
import bz2
from sklearn               import preprocessing
from sklearn.decomposition import PCA
from sklearn               import svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

SAMPLES_BETWEEN_CONTACTS = 4410

# Return <size>-long chunks of data <x> following each peak
# (i.e., value above <threshold>) using one <method>:
# 'first-peak': a chunk starts as soon as a peak is detected and can
#               include several subsequent peaks
# 'last-peak':  a chunk contains only one peak, the last induced by a contact
def contact_chunks(x,threshold,size,method='first-peak'):
    """above_th = x > threshold
    above_th = pd.Series(above_th.index[above_th == True].tolist()).astype(int)
    diff = above_th.diff(periods=-1).abs()
    
    if method == 'first-peak':
        diff = diff.shift(1).fillna(SAMPLES_BETWEEN_CONTACTS)

    peaks_indices = diff.index[diff >= SAMPLES_BETWEEN_CONTACTS].tolist()
    peaks_indices.append(len(above_th) - 1)
    peaks = above_th[peaks_indices].reset_index(drop=True)
    idx_peaks = pd.Index(peaks)
    print("cum contacts\t\t={}".format(len(idx_peaks)))
    contacts = []
    for c_idx in peaks:
        contacts.append(x[c_idx:c_idx + size].tolist())
    return pd.concat({h:x.iloc[i:i+size] for h,i in enumerate(idx_peaks)}, names=['hit'])
    """
    # first peaks have no peaks ahead within the time window
    # last peaks have no following peaks within the time window
    detect = {'first-peak': (lambda w: w[-1] and not w[:-1].any(),      0),'last-peak':  (lambda w: w[0]  and not w[1:].any(),  1-size)}
    # detect all peaks (i.e. threshold-crossing events)
    peaks  = x.abs()>threshold
    # detect only first or last peaks per hit
    #peaks  = peaks.rolling(size).apply(detect[method][0])>0 
    peaks  = peaks.rolling(size).apply(detect[method][0]).shift(detect[method][1])>0
    # indexes of peaks
    ipeaks = np.argwhere(peaks).flatten()
    # segment <x> in contact chunks
    print(ipeaks)
    return pd.concat({h:x.iloc[i:i+size] for h,i in enumerate(ipeaks)}, names=['hit'])

## Process a single wav file (i.e. one contact distance)
def process(wfname):
    dfname      = wfname[:-4]+'.dat.bz2'
    dfcomp      = 'bz2'

    ## Read and decode wav file
    rate,x = wav.read(wfname)
    nch    = 1 if x.ndim==1 else x.shape[1]
    assert(nch == 1)                      # make sure that sound is mono
    dt     = 1./rate
    dtype  = x.dtype
    x      = x/float(np.iinfo(dtype).max) # convert to float on [-1:1]
    x      = pd.Series(x,name='x')
    #x = x.iloc[:50000]
    x.index      *= dt
    x.index.name  = 'time'
    # gain correction:
    # recorded data have very low amplitude, perhaps due to missing preamp???
    gain = 6.25 #5 #100                      # not so good, amplitude levels are not consistent between distances # works with new data
    #gain = 1./np.max(np.abs(x))# not so good, peaks are not consistent between distances
    #gain = 1./(10*np.std(x))    # better, scale intercontact motor noise to similar amplitude between distances
    x *= gain
    # save raw signal
    x.to_frame().to_csv('raw_signal_'+dfname,sep=' ',compression=dfcomp)
    # print info
    print '-'*40
    print '\tWAV file:', wfname
    print 'sampling rate =', rate, 'Hz'
    print 'timestep      =', dt*1000, 'ms'
    print 'dtype         =', dtype
    print 'duration      =', x.index[-1], 'sec'
    print 'sample number =', x.size
    print 'applied gain  =', gain

    ## Spectrogram
    f,t,Sxx = signal.spectrogram(x,fs=rate,scaling='density',mode='psd',
                                 nperseg=1024*2/2, nfft=1024*2) #nperseg=256,nfft=256) #, detrend=False)
    Sxx = 10*np.log10(Sxx) # convert to dB power
    f   = pd.Float64Index(f,name='freq')
    t   = pd.Float64Index(t,name='time')
    Sxx = pd.DataFrame(Sxx,index=f,columns=t)
    # save to dat
    Sxx.index.name = len(Sxx.columns)
    Sxx.to_csv('spectrogram_'+dfname,sep=' ',compression=dfcomp)
    # print info
    print '\tSpectrogram:'
    print 'resolution (freq,time) =', Sxx.shape
    print 'aspect ratio =', Sxx.shape[1]/float(Sxx.shape[0])
    print 'freqstep =', f.to_series().diff().mean(), 'Hz'
    print 'timestep =', t.to_series().diff().mean()*1000, 'ms'
    print 'psd statistics [dB/Hz]:'
    print Sxx.stack().describe([.02,.999])

    ## Contact chunks
    nchunk = 1024*2 #*2 
    x_hit  = contact_chunks(x,threshold=.5,size=nchunk,method='last-peak')
    # save to wav
    dat = x_hit.reset_index(level='hit',drop=True).reindex(x.index,fill_value=0)
    dat = (dat * np.iinfo(dtype).max).astype(dtype).as_matrix()
    wav.write('masked_'+wfname,rate,dat)
    # save to dat
    mask = np.array((nchunk-1)*[1.]+[np.nan])
    dat  = x_hit.groupby('hit').apply(lambda x: x*mask)
    dat.to_frame().to_csv('contact_chunks_'+dfname,sep=' ',na_rep=np.nan,compression=dfcomp)
    # print info
    print '\tContact chunks:'
    print 'size =', nchunk
    print 'duration =', nchunk*dt*1000, 'ms'
    print 'chunk number =', x_hit.index.get_level_values('hit').nunique()
    
    ## Contact PSDs (power spectral densities)
    #print x_hit.reset_index(level='time',drop=True).unstack()
    #print x_hit.to_frame().reindex(level=1)
    #print x_hit #.reindex(index=np.arange(nchunk)*dt,level='time') #,drop=True)
    dat = x_hit.groupby('hit').apply(lambda h: h.reset_index(drop=True)).unstack().as_matrix()
    f ,Pxx  = signal.welch(dat, fs=rate, window='hann', nperseg=nchunk/2, #noverlap=0,
                            nfft=nchunk*1, detrend='constant', return_onesided=True, scaling='density', axis=-1)
    P0  = Pxx[:,f>20].sum(axis=-1).reshape(-1,1) # Reference power (total power of each hit)
    #Pxx = Pxx/P0
    #Pxx = np.sqrt(Pxx)
    #Pxx = 20*np.log10(Pxx)
    Pxx = 10*np.log10(Pxx) # convert to dB power
    ## f2,Pxx2= signal.periodogram(dat, fs=rate, window='hann',
    ##                         nfft=nchunk*1, detrend='constant', return_onesided=True, scaling='density', axis=-1)
    ## Pxx2 = 10*np.log10(Pxx2) # convert to dB power
    #assert np.max(np.abs(Pxx[:,1:]-Pxx2[:,1:]))==0 # except the DC component
    # save to dat
    dat = pd.DataFrame(Pxx,columns=f)
    dat.index.names = [len(dat.columns)]
    dat.to_csv('contact_psd_'+dfname,sep=' ',compression=dfcomp)
    # print info
    print '\tContact PSD:'
    print 'resolution =', f.size
    print 'freqstep =', np.diff(f).mean(), 'Hz'
    print 'reference power (mean,std) =', np.mean(P0), np.std(P0)
    
    dat = dat.stack().to_frame('psd')
    dat.index.names = ['hit','freq']
    return dat

def ci95mean(x):
    return stats.t.interval(0.95, x.count()-1, loc=x.mean(), scale=x.sem())
def ci95std(x):
    df = x.count()-1
    scale = np.sqrt(df)*x.std()
    Xl,Xh = stats.chi.interval(0.95, df)
    return scale/Xh,scale/Xl

def process_aggregate(PSDs):
    ## Mean power spectral densities + standard deviations + 95% confidence intervals
    meanPSDs = PSDs.stack().groupby(['dist','freq']).agg(['mean','std',ci95mean])
    meanPSDs[['ci95meanl','ci95meanh']] = meanPSDs.pop('ci95mean').apply(pd.Series)
    fdat = bz2.BZ2File('mean_contact_psd.dat.bz2','w')
    for dist,dat in meanPSDs.groupby(level='dist'):
        dat.to_csv(fdat,sep=' ')
        print >>fdat
    fdat.close()

    ## Support Vector Regression (using sklearn based on libSVM)
    # reformat dataset
    X = PSDs.drop(0,level='freq') # remove DC components (i.e. freq == 0)
    X = X.unstack(level='freq')
    X.columns = X.columns.droplevel(level=0) # remove 'psd' (does not work by level name!!!)
    y = X.index.get_level_values('dist').to_series()
    # preprocessing (scale to unit normal distrbution: N(0,1))
    ## print X.shape
    ## print np.sqrt((X**2).sum(axis='columns')).shape
    ## print np.sqrt((X**2).sum(axis='columns')).groupby('dist').agg(['mean','std'])
    ## print X.sum(axis='columns').groupby('dist').agg(['mean','std'])
    ## exit()
    X = pd.DataFrame(preprocessing.scale(X), index=X.index, columns=X.columns)
    # split train and test datasets (fairly: using group labels in "stratify")
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=.3,
                                                        stratify=X.index.labels[0])
    assert len( y_test.value_counts().unique())==1 # same sample size in each distance group
    assert len(y_train.value_counts().unique())==1
    
    # systematic variation of spectrum portion in [freqlo:freqhi]
    if True:
        def SVR_perf(finter):
            freqmid,freqw  = finter.name[0],finter.name[1]
            freqlo,freqhi  = freqmid-freqw/2.,freqmid+freqw/2.
            #freqlo,freqhi = finter.name[0],finter.name[1]
            mask  = (freqlo <= X.columns) & (X.columns <= freqhi)
            nfreq = mask.sum()
            if 0<=freqlo and freqhi<=22050 and nfreq>0:
            #if freqlo<freqhi and nfreq>0:
                print finter.name
                est    = svm.SVR(kernel='rbf',epsilon=.5,C=10) #,gamma=1./47)
                #est    = svm.NuSVR(kernel='rbf',nu = .05,C=5) #, gamma=1./4)
                est.fit(X_train.iloc[:,mask], y_train)
                y_pred = est.predict(X_test.iloc[:,mask])
                y_pred = pd.Series(y_pred,index=y_test.index)
                err    = y_pred - y_test
                perf = dict(freqlo_actual  = X.columns[mask][0],
                            freqhi_actual  = X.columns[mask][-1],
                            freqmid_actual = (X.columns[mask][-1]+X.columns[mask][0])/2,
                            freqw_actual   = X.columns[mask][-1]-X.columns[mask][0],
                            nsv  = len(est.support_), # nb of support vectors
                            fsv  = len(est.support_)/float(y_train.count()), # fraction of support vectors
                            r2   = est.score(X_test.iloc[:,mask], y_test),
                            RMSE = np.sqrt(np.mean(err**2)), # Root-mean-square error
                            MAE  = np.mean(np.abs(err)))     # Mean absolute error
            else:
                perf = dict(freqlo_actual=np.nan,freqhi_actual=np.nan,
                            freqmid_actual=np.nan,freqw_actual=np.nan,
                            nsv=np.nan,fsv=np.nan,r2=np.nan,RMSE=np.nan,MAE=np.nan)
            return pd.Series(perf)
        freqs = np.logspace(1,11,num=51,base=2)*10
        #fintervals = pd.MultiIndex.from_product([freqs,freqs],names=['freqlo','freqhi']).to_frame(index=False)
        #scores = fintervals.groupby(['freqlo','freqhi']).apply(SVR_perf)
        fintervals = pd.MultiIndex.from_product([freqs,freqs],names=['freqmid','freqw']).to_frame(index=False)
        scores = fintervals.groupby(['freqmid','freqw']).apply(SVR_perf)
        scores.to_csv('svr_scores.dat',sep=' ',na_rep=np.nan)
        
        finterv_r2max = scores['r2'].idxmax()
        print 'r2_max frequency interval =', finterv_r2max
        print scores.loc[finterv_r2max]
        finterv_fsvmin = scores['fsv'].idxmin()
        print 'fsv_min frequency interval =', finterv_fsvmin
        print scores.loc[finterv_fsvmin]
        
        fbins = np.hstack((-np.inf, (freqs[:-1]+freqs[1:])/2., +np.inf))
        best  = scores.groupby(pd.cut(scores['freqmid_actual'], fbins, labels=False))['r2'].idxmax()
        best  = scores.loc[best] #.sort_values(by=['r2'], ascending=False)
        best['rank'] = best['r2'].rank(ascending=False)
        best.to_csv('svr_best.dat',sep=' ',na_rep=np.nan)
        print best[best['r2']>=.95]['rank'].max()
        print 'interval using the sensor resonance frequency ='
        print best.iloc[-3]
    
    # spectrum portion of fixed width swept across the frequency domain (on a logscale)
    if True:
        #freq_width = opt_finter[1]-opt_finter[0] #140 # Hz
        #freqs = np.logspace(1,11,num=11,base=2)*10
        pred  = {}
        #for finterv in best.index:
        for finterv in [finterv_r2max]: #,finterv_fsvmin]:
            freqmid,freqw = finterv
            freqlo,freqhi = freqmid-freqw/2.,freqmid+freqw/2.
        #for freq in freqs:
        #    freqmid,freqw = freq,freq
        #    freqlo,freqhi = freqmid-freqw/2.,freqmid+freqw/2.
        #for freqlo in freqs:
            #freqhi = freqlo + freq_width
            mask   = (freqlo <= X.columns) & (X.columns <= freqhi)
            nfreq  = mask.sum()
            if nfreq>0:
                freqlo_actual  = X.columns[mask][0]
                freqhi_actual  = X.columns[mask][-1]
                freqmid_actual = (X.columns[mask][-1]+X.columns[mask][0])/2
                freqw_actual   = X.columns[mask][-1]-X.columns[mask][0]
                est    = svm.SVR(kernel='rbf',epsilon=.5,C=10)
                #est    = svm.NuSVR(kernel='rbf',nu = .05,C = 5)
                est.fit(X_train.iloc[:,mask], y_train)
                nsv    = len(est.support_) # nb of support vectors
                r2     = est.score(X_test.iloc[:,mask], y_test)
                print freqmid,freqw, nsv, r2
                assert nsv == scores.loc[finterv]['nsv']
                assert r2  == scores.loc[finterv]['r2']
                y_pred = est.predict(X_test.iloc[:,mask])
                y_pred = pd.Series(y_pred,index=y_test.index)
                y_pred_stats = y_pred.groupby('dist').agg(['mean','std',ci95mean,ci95std])
                y_pred_stats[['ci95meanl','ci95meanh']] = y_pred_stats.pop('ci95mean').apply(pd.Series)
                y_pred_stats[[ 'ci95stdl', 'ci95stdh']] = y_pred_stats.pop( 'ci95std').apply(pd.Series)
                pred[(freqlo_actual,freqhi_actual,freqmid_actual,freqw_actual)] = y_pred_stats
        pred  = pd.concat(pred,names=['freqlo_actual','freqhi_actual','freqmid_actual','freqw_actual'])
        # save results
        i = 0
        fdat  = open('svr_prediction.dat','w')
        for freq,dat in pred.groupby(['freqmid_actual','freqw_actual']):
            i += 1
            dat.to_csv(fdat,sep=' ')
            print >>fdat
        fdat.close()
        print i
    #exit()

    ## Principal Component Analysis
    # preprocessing
    ##X = PSDs.unstack(level='freq') #.as_matrix() #[:,:512]
    ##X.columns = X.columns.droplevel(level=0) # remove 'psd' (for some reason do not work with the level name!!!)
    #X = pd.DataFrame(preprocessing.scale(X), index=X.index, columns=X.columns)
    # apply PCA
    pca = PCA(n_components=3, svd_solver='full',whiten=True)
    #pca = KernelPCA(n_components=10, kernel='rbf') #, gamma=None, degree=3, coef0=1, kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None, remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=1
    Xp = pca.fit_transform(X)
    # save projected data
    Xp = pd.DataFrame(Xp, index=X.index)
    Xp.to_csv('pca_projection.dat.bz2',sep=' ',compression='bz2')
    # save principal axes
    PA = pd.DataFrame(pca.components_,columns=X.columns)
    PA.index.names = [len(PA.columns)]
    PA.to_csv('pca_principal_axes.dat.bz2',sep=' ',compression='bz2')
    print PA
    # print info
    print pca
    print 'Explained variance ratios =', pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_)
    #print 'Singular values =', pca.singular_values_


# Main: process multiple wavfiles (i.e. whole dataset)
parser = argparse.ArgumentParser(description='Process raw data.')
parser.add_argument('wavfile', nargs='+', 
                    help='WAV file containing raw data, i.e. repeated hits at a given distance.')
args = parser.parse_args()

def wfname2dist(wf):
    return float(wf[-6:-4])

PSDs = {wfname2dist(wf):process(wf) for wf in args.wavfile}
PSDs = pd.concat(PSDs,names=['dist'])
process_aggregate(PSDs)

# Notes:
# the raw signal shows a large dynamic range allowing to detect contact events easily by simple thresholding
# run through all files, in option give a range of time (to reduce file sizes)
# once chunk size determined (should be only power of 2, so determine the possibilities, although we will use a Welch with overlap to reduce noise)
# per distance, compute one PSD for each contact event, and the mean PSD + 95 conf interval (should be use logscale on the freq axis)
# then run PCA (with interpolation if we use logscale freq axis)
# from PCA get the eigenvectors
# Replicate exp: with preamp, and reduce inter-contact intervals
