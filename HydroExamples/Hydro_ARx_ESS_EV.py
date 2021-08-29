'''
Created on Jan 3, 2018

@author: dduque
'''
from SDDP import options, LAST_CUTS_SELECTOR, load_algorithm_options,\
    SLACK_BASED_CUT_SELECTOR
from SDDP.SDDP_Alg import SDDP
from SDDP.RiskMeasures import DistRobustWasserstein
from Utils.file_savers import write_object_results
from HydroModel import load_hydro_data, model_builder, random_builder, hydro_path
from InstanceGen.ReservoirChainGen import read_instance, HydroRndInstance #Necessary to unpickle file!


if __name__ == '__main__':

    '''
    Implementation of SDDP solving EV problem.
    --T=12 --R=10 --max_iter=10000 --max_time=1800 --sim_iter=1000 --lines_freq=10 --dro_r=20 --lag=1 --N=10 --dynamic_sampling=False --multicut=True
    '''
    load_algorithm_options()
    T, r_dro, instance_name, out_of_sample_rnd_cont = load_hydro_data('SDDP', 'SP')
    algo = SDDP(T, model_builder, random_builder)
    lbs = algo.run( instance_name=instance_name, ev=False)
    sim_result = algo.simulate_policy(options['sim_iter'], out_of_sample_rnd_cont)
            