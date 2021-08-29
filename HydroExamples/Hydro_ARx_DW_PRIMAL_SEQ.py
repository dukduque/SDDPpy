'''
Created on Sep 12, 2018

@author: dduque
'''
from __init__ import import_SDDP, dro_radii
#from HydroExamples import dro_radii, import_SDDP
import_SDDP()

from SDDP import options, LAST_CUTS_SELECTOR, load_algorithm_options,\
    SLACK_BASED_CUT_SELECTOR
from SDDP.SDDP_Alg import SDDP
from SDDP.RiskMeasures import DistRobust, DiscreteWassersteinInnerSolver
from Utils.file_savers import write_object_results
#from HydroModel import load_hydro_data,  hydro_path
from HydroValley import load_hydro_data, hydro_path
from InstanceGen.ReservoirChainGen import read_instance, HydroRndInstance  #Necessary to unpickle file!
from SDDP.SDDP_utils import report_stats

if __name__ == '__main__':
    '''
    Implementation of Wasserstein uncertainty set based on Philpott et al.
    Inner worst-case expectation problem is solved in the backward pass to
    compute the cuts. This approach is regarded as Primal.
    '''
    load_algorithm_options()
    T, model_builder, random_builder, rnd_container_data, rnd_container_oos, r_dro, instance_name, instance_name_gen = load_hydro_data(
        'PRIMAL', 'DW')
    #options['cut_selector'] = SLACK_BASED_CUT_SELECTOR
    algo = SDDP(T,
                model_builder,
                random_builder,
                risk_measure=DistRobust,
                dro_inner_solver=DiscreteWassersteinInnerSolver,
                radius=0,
                norm=1,
                data_random_container=rnd_container_data)
    lbs = algo.run(instance_name=instance_name)
    sim_result = algo.simulate_policy(rnd_container_oos)
    report_stats(sim_result.sims_ub)
    for r_DRO in dro_radii:
        instance_name = instance_name_gen(r_DRO)
        algo.change_dro_radius(r_DRO)
        lbs = algo.run(instance_name=instance_name)
        
        save_path = hydro_path + '/Output/DW_Primal/%s_LBS.pickle' % (instance_name)
        write_object_results(save_path, (algo.instance, lbs))
        
        sim_result = algo.simulate_policy(rnd_container_oos)
        save_path = hydro_path + '/Output/DW_Primal/%s_OOS.pickle' % (instance_name)
        write_object_results(save_path, sim_result)
        report_stats(sim_result.sims_ub)
    
    del (algo)
