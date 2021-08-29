from gurobipy import GRB
from Utils.argv_parser import sys, parse_args
import numpy as np
import os
import logging
np.set_printoptions(linewidth=90)

sddp_dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()

logger = logging.getLogger('SDDP')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
'''
Logging settings
'''
'''
Type of passes
'''
FORWARD_PASS = 1
BACKWARD_PASS = 2
'''
Status codes 
'''
SP_LOADED = 'Loaded'
SP_OPTIMAL = 'sp_optimal'
SP_INFEASIBLE = 'sp_infeasible'
SP_UNKNOWN = 'sp_unknown'
SP_UNBOUNDED = 'sp_unbounded'
'''
tolerances
'''
ZERO_TOL = 1E-9
FEASIBILITY_TOL = 1E-6
SDDP_OPT_TOL = 1E-3
'''
Cut selector
'''
LAST_CUTS_SELECTOR = 'LCS'
SLACK_BASED_CUT_SELECTOR = 'SBCS'
'''
Algorithm options
'''
options = {}
options['max_iter'] = 1000
options['max_time'] = 900
options['sim_iter'] = 1000
options['outputlevel'] = 2
options['lines_freq'] = 1
options['n_sample_paths'] = 1
options['grb_threads'] = 1
options['multicut'] = False
options['expected_value_problem'] = False
options['in_sample_ub'] = 200
options['opt_tol'] = 1E-4
options['dynamic_sampling'] = False
options['dynamic_sampling_beta'] = 0.5
options['max_cuts_last_cuts_selector'] = 50
options['slack_cut_selector'] = 1E-4
options['slack_num_iters_cut_selector'] = 200
options['max_cuts_slack_based'] = options['max_cuts_last_cuts_selector']
options['max_stage_with_oracle'] = 10
options['max_iters_oracle_ini'] = 10
options['cut_selector'] = None


def gurobiStatusCodeToStr(intstatus):
    if intstatus == GRB.LOADED:
        return SP_LOADED
    elif intstatus == GRB.OPTIMAL:
        return SP_OPTIMAL
    elif intstatus == GRB.INFEASIBLE:
        return SP_INFEASIBLE
    elif intstatus == GRB.UNBOUNDED:
        return SP_UNBOUNDED
    else:
        return SP_UNKNOWN


def alg_options():
    return options


def load_algorithm_options():
    argv = sys.argv
    _, kwargs = parse_args(argv[1:])
    for kword in kwargs:
        if kword in options:
            options[kword] = kwargs[kword]
            logger.info('Loading %s' % (kword))
        else:
            if len(kword) > 5:
                logger.warning('Parameter %s is not in the algorithm options. Check algorithm parameters wiki.' %
                               (kword))
    if 'lines_freq' not in kwargs:
        options['lines_freq'] = int(options['max_iter'] / 10)


def check_options_concitency():
    '''
    Check consistency of the algorithm options. 
    '''
    if options['expected_value_problem']:
        assert options['multicut'] == False, 'Expected value problem assumes single cut algoirthm'


class not_optimal_sp(Exception):
    def __init__(self, msg):
        super(not_optimal_sp, self).__init__(msg)
        self.msg = msg
