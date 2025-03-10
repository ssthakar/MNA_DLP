# save this as plot_data.gnu
set terminal pdf enhanced size 8,6
set output "unoptimized.pdf"

set grid
set key outside

plot "./tracked_1.dat" using ($1*0.00075) with lines lw 2 lc rgb "black" title "Ground truth", \
     "./tracked_2.dat" using ($1*0.00075) with lines dashtype 2 lw 2 lc rgb "black" title "unoptimized params", \
     "./tracked_3.dat" using ($1*0.00075) with points pt 7 ps 1 lc rgb "black" title "optimized params"
