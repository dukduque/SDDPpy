# Hydro Examples

This folder contains the examples for the paper Duque, D. and Morton, D., 2020. Distributionally Robust Stochastic Dual Dynamic Programming. SIAM Journal on Optimization. To run any of the Hydro_ARx_<variant>.py examples, you must pass at least the following arguments:

* --T number of time periods.
* --R number of reservoirs.
* --N Number of scenarios per stage.
* --dro_r radios of the DRO uncertainty set.
* --lag Lag of the auto regressive process.

Risk measures are organized in the following python examples:
* Hydro_ARx_SP.py: Solves SDDP using expectation as the risk measure.
* Hydro_ARx_DW_DUAL.py: Solves a DRO version SDDP using Wasserstein distance over a discrete set of scenarios per stage (Duque and Morton, 2020).
* Hydro_ARx_DW_Phlipott.py: Solves a DRO version SDDP using Wasserstain distance over a discrete set of scenarios per stage, but with the algorithm described in Philpott et al. 2018. Distributionally Robust SDDP. Computational Management Science.
* Hydro_ARx_CW_DUAL.py: Solves a DRO version SDDP using Wasserstein distance over a continues number of scenarios per stage. 

  
The following command line runs SDDP with the risk measure is just expectation, with a number of additional parameters that control the execution of the algorithm. Make sure that the the `path/to/SSDPpy/SDDP/` is in the PYTHONPATH. 

```
python path/to/SDDPpy/HydroExamples/Hydro_ARx_SP.py             
                --T=48,
                --R=10,
                --max_iter=5000,
                --max_time=20000,
                --sim_iter=100,
                --lines_freq=10,
                --dro_r=11,
                --lag=1,
                --N=5,
                --dynamic_sampling=True,
                --multicut=True,
                --dynamic_sampling_beta=0.5,
                --cut_selector=LCS,
                --max_cuts_last_cuts_selector=100
```

To run multiple policies, take a look at `policy_test_mac.sh` as an example. 

