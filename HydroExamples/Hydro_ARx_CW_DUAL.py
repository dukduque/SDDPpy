'''
Created on Nov 13, 2019

Run file for DRO model using wasserstein distance and continuous
support for the worst-case probability distribution.

@author: Daniel Duque
'''
from __init__ import import_SDDP
import_SDDP()

from SDDP import options, LAST_CUTS_SELECTOR, load_algorithm_options,\
    SLACK_BASED_CUT_SELECTOR
from SDDP.SDDP_Alg import SDDP
from SDDP.RiskMeasures import DistRobustWassersteinCont
from Utils.file_savers import write_object_results
from HydroModel import load_hydro_data, hydro_path
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
        'DUAL', 'CW')
    
    # Box-type set for the support of the form C\xi <= d (for a box, lb should be negative, e.g, -I\xi<=-lb)
    sup_dim = rnd_container_data.support_dimension
    supp_ctrs = [{'innovations[%i]' % (reservoir): 1} for reservoir in range(sup_dim)]
    supp_ctrs.extend(({'innovations[%i]' % (r_id): -1}) for r_id in range(sup_dim))
    supp_rhs = rnd_container_data.get_noise_ub([f'innovations[{r_id}]' for r_id in range(sup_dim)])
    supp_rhs.extend(rnd_container_data.get_noise_lb([f'innovations[{r_id}]' for r_id in range(sup_dim)], True))
    algo = SDDP(T,
                model_builder,
                random_builder,
                risk_measure=DistRobustWassersteinCont,
                radius=r_dro,
                support_ctrs=supp_ctrs,
                support_rhs=supp_rhs)
    
    lbs = algo.run(instance_name=instance_name)
    
    save_path = hydro_path + '/Output/DW_Dual/%s_LBS.pickle' % (instance_name)
    write_object_results(save_path, (algo.instance, lbs))
    
    sim_result = algo.simulate_policy(rnd_container_oos)
    save_path = hydro_path + '/Output/DW_Dual/%s_OOS.pickle' % (instance_name)
    write_object_results(save_path, sim_result)
    report_stats(sim_result.sims_ub)
    
    del (algo)
