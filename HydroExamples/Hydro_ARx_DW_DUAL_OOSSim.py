'''
Created on Sep 13, 2018

@author: dduque
'''
from __init__ import import_SDDP
import_SDDP()

from SDDP import options, LAST_CUTS_SELECTOR, load_algorithm_options,\
    SLACK_BASED_CUT_SELECTOR
from SDDP.SDDP_Alg import SDDP
from SDDP.RiskMeasures import DistRobustWasserstein
from Utils.file_savers import write_object_results
#from HydroModel import load_hydro_data, hydro_path
from HydroValley import load_hydro_data, hydro_path
from InstanceGen.ReservoirChainGen import read_instance, HydroRndInstance  #Necessary to unpickle file!
from SDDP.SDDP_utils import report_stats

if __name__ == '__main__':
    '''
    Implementation of Wasserstein uncertainty set based on taking
    the dual problem of the inner max problem that represents the
    worst-case expectation.
    '''
    load_algorithm_options()
    
    T, model_builder, random_builder, rnd_container_data, rnd_container_oos, r_dro, instance_name, _ = load_hydro_data(
        'DUAL', 'DW')
    #options['cut_selector'] = SLACK_BASED_CUT_SELECTOR#SLACK_BASED_CUT_SELECTOR#LAST_CUTS_SELECTOR
    algo = SDDP(T,
                model_builder,
                random_builder,
                risk_measure=DistRobustWasserstein,
                norm=1,
                radius=r_dro,
                data_random_container=rnd_container_data)
    
    oos_setup = {'when': [10, 30, 60, 120, 180, 300], 'random_container': rnd_container_oos, 'output': []}
    lbs = algo.run(instance_name=instance_name, out_of_sample_setup=oos_setup)
    
    save_path = hydro_path + '/Output/DW_Dual/%s_LBS.pickle' % (instance_name)
    write_object_results(save_path, (algo.instance, lbs))
    
    sim_result = algo.simulate_policy(rnd_container_oos)
    save_path = hydro_path + '/Output/DW_Dual/%s_OOS.pickle' % (instance_name)
    write_object_results(save_path, sim_result)
    report_stats(sim_result.sims_ub)
    
    del (algo)
