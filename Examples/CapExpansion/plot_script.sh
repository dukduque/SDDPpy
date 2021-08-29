#!/bin/bash

ptf=/Users/dduque/Dropbox/WORKSPACE/SDDP/Examples/CapExpansion/Output/
exp_file_name=DW_CapExp_N10_m20_n100
python /Users/dduque/Dropbox/WORKSPACE/SDDP/OutputAnalysis/SimulationAnalysis.py  --exp_file=$exp_file_name --path_to_files=$ptf --plot_type=OOS --sampling=ES --cut_type=MC --N=$n --max_time=$tmax --dw_sam=$dw_sam --n=$n