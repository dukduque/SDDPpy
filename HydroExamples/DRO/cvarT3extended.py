'''
Created on Nov 6, 2017

@author: dduque
'''
import numpy as np

from gurobipy import *
from statsmodels.genmod._tweedie_compound_poisson import _alpha

'''
Example of a hydrovalley
'''

class Turbine:
    def __init__(self): 
        self.flowknots = None
        self.powerknot = None
class Reservoir:
    global_res_id = 0 
    def __init__(self, minl, maxl, ini, turbine, spill_cost, inflows, normNoise):
        self.id = self.global_res_id
        self.min = minl
        self.max = maxl
        self.initial = ini
        self.turbine = turbine
        self.spill_cost = spill_cost
        self.inflows = inflows
        self.normNoise = normNoise
        self.global_res_id +=1

class HV_Instance:
    def __init__(self):
        self.reservoirs = []        

def buildExtendedModel(hvInstance, _lambda, _alpha):
    m = Model()
    
    numReservoirs = len(hvInstance.reservoirs)
    
    #First stage variables
    gen_quantitiy = m.addVar(lb=0, vtype= GRB.CONTINUOUS, 'gen1')
    level1 = {}
    spill1 = {}
    outflow1 = {}
    dispatch1 = {}
    
    for r in hvInstance:
        level1[r.id] = m.addVar(lb=r.min, ub=r.max, vtype=GRB.CONTINUOUS, 'level1%i' %(r.id))
        spill1[r.id] = m.addVar(vtype=GRB.CONTINUOUS, 'spill1%i' %(r.id))
        outflow1[r.id] = m.addVar(vtype=GRB.CONTINUOUS, 'outflow1%i' %(r.id))
        for (i,l) in enumerate(r.turbine.flowknots):
            dispatch1[r.id,i] = m.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, 'dispatch1%i%i' %(r.id,i))
            
            
        
  
        
    
    
    
    
    


