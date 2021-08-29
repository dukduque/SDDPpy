'''
Created on Jan 3, 2018

@author: dduque
'''
import csv
import SDDP
import logging
import numpy as np
from SDDP.RandomnessHandler import RandomContainer, StageRandomVector, AR1_depedency
from SDDP.SDPP_Alg import SDDP
from SDDP import logger as sddp_log

from Utils.argv_parser import sys,parse_args
from gurobipy import Model, GRB, quicksum
from InstanceGen.ReservoirChainGen import read_instance, HydroRndInstance
from HydroExamples import *
'''
Global variables to store instance data
'''
hydro_instance = read_instance()
T = None
nr = None
Rmatrix = None
RHSnoise = None
initial_inflow = None 
valley_chain = None
prices = None

def random_builder():
    rc = RandomContainer()
    rndVectors = []
    for t in range(0,T):
        rv_t = StageRandomVector(t)
        rc.append(rv_t)
        for (i,r) in enumerate(valley_chain):
            if t>0:
                re = rv_t.addRandomElement('innovations[%i]' %(i), r.inflows)
            else:
                re = rv_t.addRandomElement('innovations[%i]' %(i), [0.0])
            rndVectors.append(rv_t)
    return rc

def model_builder(stage):
    '''
    Builds a particular instance of a multistage problem
    '''
    m = Model('Hydrovalley')

    #Reservoir level
    reservoir_level = m.addVars(nr, 
                                lb = [r.min for r in valley_chain], 
                                ub = [r.max for r in valley_chain], 
                                obj = 0,
                                vtype=GRB.CONTINUOUS, 
                                name='reservoir_level')
    reservoir_level0 = m.addVars(nr, 
                                lb = 0, 
                                ub = 0, 
                                obj = 0,
                                vtype=GRB.CONTINUOUS, 
                                name='reservoir_level0')
    inflow = m.addVars(nr,lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='inflow')
    inflow0 = m.addVars(nr, lb=0, ub=0, obj=0, vtype=GRB.CONTINUOUS, name='inflow0')
    
    
    outflow = m.addVars(nr, lb=0, obj=0, vtype=GRB.CONTINUOUS, name='outflow')
    spill = m.addVars(nr, lb=0, obj=0, vtype=GRB.CONTINUOUS, name='spill')
    innovations = m.addVars(nr,  obj=0, vtype=GRB.CONTINUOUS, name='innovations')
    pour = m.addVars(nr, lb=0, obj=0, vtype=GRB.CONTINUOUS, name='pour')
    generation = m.addVar(lb=0, obj=0, vtype=GRB.CONTINUOUS, name='generation')
    dispatch = m.addVars([(ri,tf)  for (ri,r) in enumerate(valley_chain) for tf in range(0,len(r.turbine.flowknots))],
                        lb=0,ub=1,obj=0,vtype= GRB.CONTINUOUS,name='dispatch')
    if stage == 0:
        for v in reservoir_level0:
            reservoir_level0[v].lb = valley_chain[v].initial
            reservoir_level0[v].ub = valley_chain[v].initial
        for v in inflow0:
            inflow0[v].lb = 0
            inflow0[v].ub = 0
            inflow[v].lb = initial_inflow[v]
            inflow[v].ub = initial_inflow[v]
            
    m.update()
            
    in_state = [v.VarName for v in reservoir_level0.values()]
    in_state.extend((v.VarName for v in inflow0.values()))
    out_state = [v.VarName for v in reservoir_level.values()]
    out_state.extend((v.VarName for v in inflow.values()))
    rhs_vars = [v.VarName for v in innovations.values()]
    
    #Constraints
    #AR1 model as a constraint
    R_t = Rmatrix[stage] #For lag 1 only!
    m.addConstrs((inflow[i] - sum(R_t[1][i][j]*inflow0[j]  for j in range(0,len(valley_chain)) if j in R_t[1][i]) == innovations[i]    for i in range(0,len(valley_chain)) ), 'AR1')
    #Balance constraints
    m.addConstr(reservoir_level[0] ==  reservoir_level0[0] + inflow[0] - outflow[0] - spill[0] + pour[0], 'balance[0]')
    m.addConstrs((reservoir_level[i] ==  reservoir_level0[i] + inflow[i] - outflow[i] - spill[i] + pour[i] + outflow[i-1] + spill[i-1]     for i in range(1,nr)), 'balance')
          
    #Generation
    m.addConstr(generation==quicksum(r.turbine.powerknots[level] * dispatch[i,level] for (i,r) in enumerate(valley_chain) for level in range(0,len(r.turbine.flowknots))), 'generationCtr')

    # Flow out
    for (i,r) in enumerate(valley_chain):
        m.addConstr(outflow[i] == quicksum(r.turbine.flowknots[level] * dispatch[i, level] for level in range(len(r.turbine.flowknots))), 'outflowCtr[%i]' %(i))
    
    #Dispatched
    for (i,r) in enumerate(valley_chain):
        m.addConstr(quicksum(dispatch[i, level] for level in range(len(r.turbine.flowknots)))<= 1, 'dispatchCtr[%i]' %(i))
    #Objective
    objfun = -prices[stage]*generation + quicksum(0*r.spill_cost*spill[i] for (i,r) in enumerate(valley_chain)) + quicksum(r.spill_cost*pour[i] for (i,r) in enumerate(valley_chain))
    m.setObjective(objfun, GRB.MINIMIZE)
    m.update()

    return m, in_state, out_state, rhs_vars

if __name__ == '__main__':
    sddp_log.addHandler(logging.FileHandler("HydroAR1_ESS.log", mode='w'))
    argv = sys.argv
    positional_args,kwargs = parse_args(argv[1:])
    if 'R' in kwargs:
        nr = kwargs['R']
    if 'T' in kwargs:
        T = kwargs['T']
    if 'max_iter' in kwargs:
        SDDP.options['max_iter'] = kwargs['max_iter']
        SDDP.options['lines_freq'] = int(SDDP.options['max_iter']/10)
    if 'sim_iter' in kwargs:
        SDDP.options['sim_iter'] = kwargs['sim_iter']
    

    
    for nr in [50]:#,10,50,100,500,1000]:
        instance_name = "Hydro_R%i_AR1_T%i_I%i_ESS" % (nr, T, SDDP.options['max_iter'])
        Rmatrix = hydro_instance.ar_matrices
        RHSnoise = hydro_instance.RHS_noise[0:nr]
        initial_inflow = hydro_instance.inital_inflows[0:nr]
        valley_chain = [
                Reservoir(0, 200, 20, Turbine([50, 60, 70], [55, 65, 70]), 1000, x) for x in RHSnoise
                ]
        prices = [1+round(np.sin(0.8*x),2) for x in range(0,T)]
        
        
        algo = SDDP(T, model_builder, random_builder)
        algo.run( instance_name=instance_name)
        algo.simulate_policy(SDDP.options['sim_iter'])
        del(algo)
    
    
    sddp_log.addHandler(logging.FileHandler("HydroAR1_ESS.log"))
    
