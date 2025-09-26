'''
Created on Sep 13, 2018
@author: dduque

Solves a hydro-power generation problem using SDDP. This examples uses the 
expected value as risk measure.
'''
from __init__ import import_SDDP
import_SDDP()

from SDDP import options, LAST_CUTS_SELECTOR, load_algorithm_options,\
    SLACK_BASED_CUT_SELECTOR
from SDDP.SDDP_Alg import SDDP
from Utils.file_savers import write_object_results
from HydroValley import load_hydro_data, hydro_path
from InstanceGen.ReservoirChainGen import read_instance, HydroRndInstance  # Necessary to unpickle file!
from SDDP.SDDP_utils import report_stats

if __name__ == '__main__':
    load_algorithm_options()
    
    T, model_builder, random_builder, rnd_container_data, rnd_container_oos, r_dro, instance_name, _ = load_hydro_data(
        'SP', '')
    
    algo = SDDP(T, model_builder, random_builder, lower_bound=-1E10)
    lbs = algo.run(instance_name=instance_name)
    
    save_path = hydro_path + '/Output/DW_Dual/%s_LBS.pickle' % (instance_name)
    write_object_results(save_path, (algo.instance, lbs))
    
    sim_result = algo.simulate_policy(rnd_container_oos)
    save_path = hydro_path + '/Output/DW_Dual/%s_OOS.pickle' % (instance_name)
    write_object_results(save_path, sim_result)
    report_stats(sim_result.sims_ub)
    del (algo)
