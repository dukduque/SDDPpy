'''
Created on Sep 13, 2018

@author: dduque
'''
#===============================================================================
# import os
# import sys
# #Include all modules of the project when running on shell
# this_path = os.path.dirname(os.path.abspath(__file__))
# parent_path= os.path.abspath(os.path.join(this_path, os.pardir))
# cwd = os.getcwd()
# print(__file__, this_path, parent_path, cwd, os.path.abspath(os.pardir))
# sys.path.append(parent_path)
#===============================================================================
from __init__ import import_SDDP
import_SDDP()

from SDDP import options, LAST_CUTS_SELECTOR, load_algorithm_options,\
    SLACK_BASED_CUT_SELECTOR
from SDDP.SDDP_Alg import SDDP
from SDDP.RiskMeasures import DistRobust, PhilpottInnerDROSolver
from Utils.file_savers import write_object_results
from HydroModel import load_hydro_data, hydro_path
from InstanceGen.ReservoirChainGen import read_instance, HydroRndInstance #Necessary to unpickle file!
from SDDP.SDDP_utils import report_stats



if __name__ == '__main__':
    '''
    Implementation of Wasserstein uncertainty set based on taking
    the dual problem of the inner max problem that represents the
    worst-case expectation.
    '''
    load_algorithm_options()
 
    T, model_builder, random_builder, rnd_container_data, rnd_container_oos, r_dro, instance_name = load_hydro_data('PRIMAL', 'PhiVar')
    #options['cut_selector'] = SLACK_BASED_CUT_SELECTOR#SLACK_BASED_CUT_SELECTOR#LAST_CUTS_SELECTOR
    algo = SDDP(T, model_builder, random_builder, risk_measure = DistRobust, dro_inner_solver=PhilpottInnerDROSolver, set_type = DistRobust.L1_NORM , radius = r_dro, data_random_container=rnd_container_data)
    lbs = algo.run(instance_name=instance_name, dynamic_sampling=options['dynamic_sampling'])                                                              
    
    save_path = hydro_path+'/Output/Phi_Primal/%s_LBS.pickle' %(instance_name)
    write_object_results(save_path, (algo.instance, lbs))
    
    sim_result = algo.simulate_policy(rnd_container_oos)
    save_path = hydro_path+'/Output/Phi_Primal/%s_OOS.pickle' %(instance_name)
    write_object_results(save_path, sim_result)
    report_stats(sim_result.sims_ub)
    del(algo)
    
    
