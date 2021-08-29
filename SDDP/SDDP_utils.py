'''
Created on Jun 13, 2018

@author: dduque
'''
import numpy as np
from SDDP import logger as sddp_log

def print_model(m):
    '''
    Print out a gurobi model
    '''
    print(m.getConstrs())
    for c in m.getConstrs(): print(c.ConstrName, m.getRow(c) , '  ', c.Sense, '  ', c.RHS)
    for v in m.getVars(): print(v.varname, ' '  , v.lb , '  ---  ', v.ub)
    
    

def report_stats(out_of_sample):
    '''
    Report percentiles and other descriptive statistics of a list of out-of-sample outcomes
    Args:
        out_of_sample (list of float): out-of sample results
    '''
    oos = np.array(out_of_sample)
    p20 = np.percentile(oos,20)
    p80 = np.percentile(oos,80)
    p10 = np.percentile(oos,10)
    p90 = np.percentile(oos,90)
    p5 = np.percentile(oos,5)
    p95 = np.percentile(oos,95)
    o_median = np.median(oos)
    o_mean = np.mean(oos)
    o_sd = np.std(oos)
    
    sddp_log.info('Mean:   \t%10.2f' %(o_mean))
    sddp_log.info('Median: \t%10.2f' %(o_median))
    sddp_log.info('SD:     \t%10.2f' %(o_sd))
    sddp_log.info('20-80: \t(%10.2f, %10.2f)' %(p20,p80))
    sddp_log.info('10-90: \t(%10.2f, %10.2f)' %(p10,p90))
    sddp_log.info('05-95: \t(%10.2f, %10.2f)' %(p5,p95))
