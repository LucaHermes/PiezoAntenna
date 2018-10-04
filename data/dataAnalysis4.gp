set term pdfcairo enhanced color size 12cm,19cm font "Sans,10"
set output 'fig_pca_projection.pdf'
set multiplot layout 2,1
unset key
set tics out nomirror
set xlabel 'PC1'
set ylabel 'PC2'
set zlabel 'PC3'
set xrange [-2.5:2.5]
set yrange [-2.5:2.5]
set zrange [-2.5:2.5]
set cblabel 'Contact distance, cm'
set cbtics 5,2,23

set view equal xyz
set xyplane relative 0
set view 70,60,1.4,1.4
set hidden3d
unset colorbox
splot '< bzcat pca_projection.dat.bz2' u '0':'1':'2':'dist' w p pt 7 ps .5 lc pal z

set border 3
set size ratio -1
set colorbox
plot '< bzcat pca_projection.dat.bz2' u '0':'1':'dist' w p pt 7 ps .5 lc pal z
unset multi
set output
#####
set term pdfcairo enhanced color size 12cm,19cm font "Sans,10"
set output 'fig_pca_axes.pdf'
set multiplot layout 3,1
unset key
set border 3
set tics out nomirror
set xlabel 'Frequency, Hz'
set xrange [20:22050]
set xtics 2
set logscale x
set yrange [-.1:.13]
do for [pc = 0:2] {
    set title 'Coefficients of PC'.(pc+1)
    plot '< bzcat pca_principal_axes.dat.bz2' nonuniform matrix u 1:(($3)/(pc==$2)) w filledcurves y=0 lc -1
}
unset multi
set output

###########################
set term pdfcairo enhanced color size 2*12cm,19cm font "Sans,10"
set output 'fig0.pdf'
set multi layout 3,2
distances = '19 23 9 13 5 7' #'5 7 9 19'
t0 = '410 420 350 390 300 300' #'300 300 350 410'
#distances = '7 13 23' # for fig2bis.pdf
k = .247*1.35

unset key
set border 3
set tics out nomirror
set xlabel 'Time, ms'
set xtics 10
#set xrange [0.3:.5]
#unset xtics

#set lmargin at screen .13
#set rmargin at screen .88

set ylabel 'Amplitude, a.u.' #offset -3,0
set yrange [-1:1]
set ytics 1 #.5
i = 0
do for [dist in distances] {
    #set tmargin at screen .18*1.25+i*k+.06*1.5
    #set bmargin at screen .18*1.25+i*k

    set xrange [word(t0,i+1)+0:word(t0,i+1)+70]
    data = sprintf('distance%02d', 1*dist)
    fdat(prefix) = '< bzcat '.prefix.'_'.data.'.dat.bz2'
    set title sprintf('Contact distance = %d cm', 1*dist) offset 0,-1
    plot fdat('raw_signal')     u (column('time')*1000):'x' w l lc 'gray', \
	 fdat('contact_chunks') u (column('time')*1000):'x' w l lc -1,     \
	 0.5 w l lc 0 dt '-', -0.5 w l lc 0 dt '-'
    i = i + 1
}
unset multi
set output
reset session
#####################################################################


set term pdfcairo enhanced color size 12cm,15cm font "Sans,10"
set output 'fig4bis2.pdf'
set multi layout 2,1 columnsfirst margins .11,.83*0+.96,.07,.98

set tics out nomirror front
unset key

set border 2
#set xlabel 'Spectrum midpoint, Hz'
set xrange [20:22050]
unset xtics #4
set logscale x
set cblabel 'Fractional bandwidth, %'
#set cbrange [20:22050]
set cbtics 50 #2
#set logscale cb
set ylabel 'Coefficient of determination, R^2'
set yrange [-0.2:1]
#set ytics .25
set palette rgb 33,13,10
#set palette defined (20 'gray', 22050 'black')
#set colorbox user origin screen .85,.07 size char 1, screen .2 #.98-.07
set colorbox user origin screen .85,.09 size char 1, screen .2
plot .95 w l lc -1 dt '-', 'svr_best.dat' u 'freqmid_actual':'r2' w l lc -1, 'svr_scores.dat' u 'freqmid_actual':'r2':(100*column('freqw_actual')/column('freqmid_actual')) w p pt 7 ps .3 lc pal z

set border 3
set tics out nomirror
set ylabel 'Fraction of support vectors, %'
set yrange [0:*] #[0:200]
set ytics autofreq
set xlabel 'Center frequency, Hz'
set xrange [20:22050]
set xtics 2
set logscale x
set cblabel 'Fractional bandwidth, Hz'
#set cbrange [20:22050]
#set cbtics 2
#set logscale cb
unset colorbox
plot 'svr_best.dat' u 'freqmid_actual':(column('fsv')*100) w l lc -1, 'svr_scores.dat' u 'freqmid_actual':(column('fsv')*100):(100*column('freqw_actual')/column('freqmid_actual')) w p pt 7 lc pal z ps .3

unset multiplot
set output
reset session

#############
set term pdfcairo enhanced color size 12cm,15cm font "Sans,10"
set output 'fig4bis.pdf'
set multi layout 2,1 columnsfirst margins .11,.83*0+.96,.07,.98

set tics out nomirror front
unset key

set border 2
#set xlabel 'Spectrum midpoint, Hz'
#set xrange [20:22050]
unset xtics #4
#set logscale x
set cblabel 'Bandwidth, Hz'
set cbrange [20:22050]
set cbtics 4 #2
set logscale cb
set ylabel 'Coefficient of determination, R^2'
set yrange [0.7:1]
#set ytics .25
set palette rgb 33,13,10
#set palette defined (20 'gray', 22050 'black')
#set colorbox user origin screen .85,.07 size char 1, screen .2 #.98-.07
set colorbox user origin screen .85,.49 size char 1, screen .2
plot .95 w l lc -1 dt '-', 'svr_best.dat' u (100*column('freqw_actual')/column('freqmid_actual')):'r2' w l lc -1, 'svr_scores.dat' u (100*column('freqw_actual')/column('freqmid_actual')):'r2':'freqw_actual' w p pt 7 ps .3 lc pal z

set border 3
set tics out nomirror
set ylabel 'Fraction of support vectors, %'
set yrange [0:*] #[0:200]
set ytics autofreq
set xlabel 'Fractional bandwidth, %'
#set xrange [20:22050]
#set xtics 2
#set logscale x
set cblabel 'Bandwidth, Hz'
set cbrange [20:22050]
set cbtics 2
set logscale cb
unset colorbox
plot 'svr_best.dat' u (100*column('freqw_actual')/column('freqmid_actual')):(column('fsv')*100) w l lc -1, 'svr_scores.dat' u (100*column('freqw_actual')/column('freqmid_actual')):(column('fsv')*100):'freqw_actual' w p pt 7 lc pal z ps .3

unset multiplot
set output
reset session


###############
set term pdfcairo enhanced color size 12cm,15cm font "Sans,10"
set output 'fig4.pdf'
set multi layout 2,1 columnsfirst margins .11,.83*0+.96,.07,.98

set tics out nomirror front
unset key

set border 2
#set xlabel 'Spectrum midpoint, Hz'
set xrange [20:22050]
unset xtics #4
set logscale x
set cblabel 'Bandwidth, Hz'
set cbrange [20:22050]
set cbtics 4 #2
set logscale cb
set ylabel 'Coefficient of determination, R^2'
set yrange [-0.2:1]
#set ytics .25
set palette rgb 33,13,10
#set palette defined (20 'gray', 22050 'black')
#set colorbox user origin screen .85,.07 size char 1, screen .2 #.98-.07
set colorbox user origin screen .85,.09 size char 1, screen .2
plot .95 w l lc -1 dt '-', 'svr_best.dat' u 'freqmid_actual':'r2' w l lc -1, 'svr_scores.dat' u 'freqmid_actual':'r2':'freqw_actual' w p pt 7 ps .3 lc pal z

set border 3
set tics out nomirror
set ylabel 'Fraction of support vectors, %'
set yrange [0:*] #[0:200]
set ytics autofreq
set xlabel 'Center frequency, Hz'
set xrange [20:22050]
set xtics 2
set logscale x
set cblabel 'Bandwidth, Hz'
set cbrange [20:22050]
set cbtics 2
set logscale cb
unset colorbox
plot 'svr_best.dat' u 'freqmid_actual':(column('fsv')*100) w l lc -1, 'svr_scores.dat' u 'freqmid_actual':(column('fsv')*100):'freqw_actual' w p pt 7 lc pal z ps .3

unset multiplot
set output
reset session

#######################################################################
set term pdfcairo enhanced color size 12cm,19cm font "Sans,10"
set output 'fig3_all_b.pdf'
set multi
set lmargin at screen .11
set rmargin at screen .93
set tmargin at screen .99
set bmargin at screen .4
set tics out nomirror
unset key
set border 0

#set xlabel 'Frequency, Hz'
set xrange [20:22050]
unset xtics #2 #10
set logscale x
set y2label 'Contact distance, cm' offset char 2,0
psdmin = -120; psdmax = -30
#psdmin = -120+20; psdmax = 5 #-30+30
#psdmin = 0; psdmax = .03 #.4
y_offset(dist) = (dist-5.)/2*(psdmax-psdmin)*0.45 #.2 #40
set yrange [psdmin+y_offset(5):psdmax-10*1+y_offset(23)] # -10: power at largest dist is less
set ytics psdmin,30,psdmax out nomirror scale 2
set cblabel 'PSD, dB/Hz' offset char -8,0
set cbrange [psdmin:psdmax]
unset cbtics
set colorbox user origin graph 0,0 size char -.8, first psdmax-psdmin
#set palette defined (psdmin 'red', psdmax 'black')
do for [dist = 5:23:2] { set label dist ''.dist at graph 1.01, first psdmin+y_offset(dist)+10 }
dat(fname) = '< bzcat '.fname.'.bz2'
plot dat('mean_contact_psd.dat') u 'freq':(column('mean')+y_offset(column('dist'))-column('std')):(column('mean')+y_offset(column('dist'))+column('std')) w filledcu lc 'gray', dat('mean_contact_psd.dat') u 'freq':(column('mean')+y_offset(column('dist'))):'mean' w l lc pal z lw 1

set tmargin at screen .35 #.27
set bmargin at screen .06
unset label
unset y2label
set tics out nomirror
set border 3
set xlabel 'Frequency, Hz'
set xtics 2
set ylabel 'Rank'
set yrange [50:1]
set ytics 10,10 format '%g^{th}'
set ytics add ('1^{st}' 1)
set colorbox default
set cblabel 'Fraction of support vectors, %' offset 0,.5
set cbrange [10:100]
set cbtics 10,30 #autofreq
set colorbox horizontal user origin graph .7,1 size graph .3,char -.4
set palette defined (0 'gray', 100 'black')
set arrow 1 from first 20,37.5 to 22050,37.5 nohead dt '-'
plot 'svr_best.dat' u 'freqlo_actual':'rank':'freqw_actual':(0):(100*column('fsv')) w vec nohead lc pal z lw 4
unset multi
set output
reset session

##############################################################################
set term pdfcairo enhanced color size 12cm,6cm font "Sans,10"
set output 'fig5.pdf'
set multi #layout 3,1
unset key
set tics out nomirror
set tmargin at screen .97
set bmargin at screen .17

set size ratio -1
set lmargin at screen .1
set rmargin at screen .5
set border 3
set xlabel 'Contact distance, cm'
set xrange [5-1:23+1]
set xtics 5,2
set ylabel 'Mean estimate, cm'
set yrange [5-1:23+1]
set ytics 5,2
#plot 'svr_prediction.dat' u 'dist':(column('mean')-column('std')):(column('mean')+column('std')) w filledcurves lc 'gray', '' u 'dist':'mean' w l lc -1, x w l lc -1 dt '-'
plot x w l lc 'gray', 'svr_prediction.dat' u 'dist':'mean':'std' w yerrorbars pt 7 ps .4 lc -1

set size noratio
set lmargin at screen .6
set rmargin at screen .97
set tmargin at screen .53
#set bmargin at screen .17
set border 2
#set ylabel 'Precision, cm' offset 1.5,0
set ylabel 'Estimate std. dev., cm' offset 1.5,0
set yrange [*:*]
set ytics .5
#plot 'svr_prediction.dat' u 'dist':'ci95stdl':'ci95stdh' w filledcurves lc 'gray', '' u 'dist':'std' w l lc -1, 0 w l lc -1 dt '-'
plot 0 w l lc 'gray', 'svr_prediction.dat' u 'dist':'std':'ci95stdl':'ci95stdh' w yerrorbars pt 7 ps .4 lc -1

set tmargin at screen .97
set bmargin at screen .61
set border 2
unset xlabel
unset xtics
#set ylabel 'Accuracy, cm'
set ylabel 'Mean error, cm'
set yrange [*:*]
set ytics .5
#plot 'svr_prediction.dat' u 'dist':(column('ci95meanl')-column('dist')):(column('ci95meanh')-column('dist')) w filledcurves lc 'gray', '' u 'dist':(column('mean')-column('dist')) w l lc -1, 0 w l lc -1 dt '-'
plot 0 w l lc 'gray', 'svr_prediction.dat' u 'dist':(column('mean')-column('dist')):(column('ci95meanl')-column('dist')):(column('ci95meanh')-column('dist')) w yerrorbars pt 7 ps .4 lc -1

unset multi
set output
reset session
exit

#######################################################
set term pdfcairo enhanced color size 12cm,19cm font "Sans,10"
set output 'fig2.pdf'
set multi
distances = '5 9 19'
#distances = '7 13 23' # for fig2bis.pdf
k = .247*1.35

unset key
set border 2
set tics out nomirror
set xrange [0:2]
unset xtics

set lmargin at screen .13
set rmargin at screen .88

set ylabel 'Amplitude, a.u.' offset -3,0
set yrange [-1:1]
set ytics 1 #.5
i = 0
do for [dist in distances] {
    set tmargin at screen .18*1.25+i*k+.06*1.5
    set bmargin at screen .18*1.25+i*k

    data = sprintf('distance%02d', 1*dist)
    fdat(prefix) = '< bzcat '.prefix.'_'.data.'.dat.bz2'
    set title sprintf('Contact distance = %d cm', 1*dist) offset 0,-1
    plot fdat('raw_signal')     u 'time':'x' w l lc 'gray', \
	 fdat('contact_chunks') u 'time':'x' w l lc -1,     \
	 0.5 w l lc 0 dt '-', -0.5 w l lc 0 dt '-'
    i = i + 1
}

unset title
set border 3
set xlabel 'Time, s' offset 0,.8
set xtics .5 out nomirror offset .1,.4
set ylabel 'Frequency, Hz' offset 1,0
set yrange [20:22050]
set ytics 4
set logscale y
set cblabel 'PSD, dB/Hz' offset -1,0
set cbrange [-120:-30]
set cbtics 30
set colorbox user origin graph 1.02,0 size char 1, graph 1
#set palette defined (-120 'white', -30 'black')
i = 0
do for [dist in distances] {
    set tmargin at screen .04+i*k+.12*1.35
    set bmargin at screen .04+i*k
    
    data = sprintf('distance%02d', 1*dist)
    fdat(prefix) = '< bzcat '.prefix.'_'.data.'.dat.bz2'
    plot fdat('spectrogram') nonuniform matrix u 1:($2/1):($3) with image
    i = i + 1
    unset colorbox
    unset xlabel
    unset xtics
    set border 2
}


unset multi
set output
reset session

exit

###################################
do for [dist = 5:23:2] { # contact distance [cm]
#    dist = 5
    reset
    data = sprintf('distance%02d', dist)
    fdat(prefix) = '< bzcat '.prefix.'_'.data.'.dat.bz2'
    #data = 'd'.d.'s1m2' # dataset
    #dist = 23-(d-1)*2    # contact distance [cm]
    print data, ' ', dist, ' cm'

    set label 1 data.'   dist = '.dist.' cm' at screen .5,.98 center
    set term pdfcairo enhanced color size 12cm,10cm font "Sans,10"
    set output 'fig1_'.data.'.pdf'
    set multiplot layout 2,1 margin .14,.89,.1,.98 spacing 0.05
    unset key
    set border 2
    unset xtics
    #set xtics out nomirror format ""
    set xrange [0.4*0+2*0:2.15]
    #set xrange [0:40]

    set ylabel 'Amplitude, a.u.'
    set ytics .5 out nomirror
    set yrange [-1:1]
    plot fdat('raw_signal') u 'time':'x' w l lc 'gray' lw .5, fdat('contact_chunks') u 'time':'x' w l lc -1, 0.5 w l lc 0 dt '.', -0.5 w l lc 0 dt '.'

    set border 3
    set xlabel 'Time, s'
    set xtics out nomirror format "% h"
    set ylabel 'Frequency, Hz'
    set ytics 2 out nomirror
    #set yrange [0:22.05]
    set yrange [20:22050]
    set logscale y
    set cblabel 'Power spectral density, dB/Hz'
    #set logscale cb
    set cbrange [-120:-30]
    set colorbox user size 0.01,.5
    #show colorbox
    #set cbrange [1.20249396487e-21*1000 : 5.97533385587e-09*1000]
    #set cbrange [3.5298816942e-15*1000 : 3.47954992048e-12*1000]
    #set cbrange [1.34890887483e-15*1000 : 2.22238125264e-10*1000] #1.09533117796e-11*1000]
    plot fdat('spectrogram') nonuniform matrix u 1:($2/1):($3) with image
    unset multiplot
    set output


    reset
    set label 1 data.'   dist = '.dist.' cm' at screen .5,.98 center
    set term pdfcairo enhanced color size 12cm,10cm font "Sans,10"
    set output 'fig2_'.data.'.pdf'
    set multiplot layout 2,1 margin .11,.89,.1,.98 spacing 0.05
    unset key
    
    set border 3
    set tics out nomirror
    set xlabel 'Frequency, Hz'
    set xrange [20:22050]
    set logscale x
    set xtics 2
    set ylabel 'Power spectral density, dB/Hz'
    set yrange [-130:-30]
    set colorbox user origin 0.9,0.02 size 0.01,.97
    set cblabel 'Contact no.'
    set cbtics 1
    #plot fdat('contact_psd') u (column('freq')/1000):'psd':(column(-1)+1) w l lc pal z lw .5
    plot fdat('contact_psd') nonuniform matrix u ($1/1):3:2 w l pal z lw .5
    
    #plot 'contact_psd_mean_'.data.'.dat' u (column('f')/1000):'cimin':'cimax' w filledcu lc 'gray', '' u (column('f')/1000):'psd' w l lc -1
    
    
    unset multi
    set output
}
