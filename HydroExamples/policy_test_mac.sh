
python ./Hydro_ARx_SP.py --T=24 --R=10 --max_iter=100001 --max_time=30 --sim_iter=1000 --lines_freq=100 --dro_r=0 --lag=1 --N=2 --dynamic_sampling=False --multicut=True
for c in {-2,-1,0,1,2}
do
		for b in {1,5}
        do
			r=$(bc -l <<< "$b*(10^$c)")
			python ./Hydro_ARx_DW_DUAL.py --T=24 --R=10 --max_iter=100001 --max_time=30 --sim_iter=1000 --lines_freq=100 --dro_r=$r --lag=1 --N=2 --dynamic_sampling=True --multicut=True --dynamic_sampling_beta=0.99
			python ./Hydro_ARx_DW_Phlipott.py --T=24 --R=10 --max_iter=100001 --max_time=30 --sim_iter=1000 --lines_freq=100 --dro_r=$r --lag=1 --N=2 --dynamic_sampling=True --multicut=False --dynamic_sampling_beta=0.99
			python ./Hydro_ARx_DW_DUAL.py --T=24 --R=10 --max_iter=100001 --max_time=30 --sim_iter=1000 --lines_freq=100 --dro_r=$r --lag=1 --N=2 --dynamic_sampling=False --multicut=True --dynamic_sampling_beta=0
			python ./Hydro_ARx_DW_Phlipott.py --T=24 --R=10 --max_iter=100001 --max_time=30 --sim_iter=1000 --lines_freq=100 --dro_r=$r --lag=1 --N=2 --dynamic_sampling=False --multicut=False --dynamic_sampling_beta=0
        done
done
