'''
Created on Sep 21, 2018

@author: dduque

Based on Oscar Dowson - MilkPowder
Dowson et al. (2017). A multi-stage stochastic optimization model of a dairy farm. 
'''

import json
import os 
import sys
import numpy
from SDDP.RandomnessHandler import RandomContainer, StageRandomVector
from gurobipy import Model, GRB
from SDDP.SDDP_utils import print_model
from SDDP.SDDP_Alg import SDDP
from SDDP import options, load_algorithm_options
from SDDP.RiskMeasures import DistRobust, DiscreteWassersteinInnerSolver,\
    DistRobustWasserstein
milk_path = os.path.dirname(os.path.realpath(__file__))

parameters = json.load(open('%s/model.parameters.json' % (milk_path)))
O = parameters["niwa_data"]
T = None
print(parameters["maximum_milk_production"])

def random_builder():
    #evapotranspiration  rainfall
    #O = parameters["niwa_data"]
    rc = RandomContainer()
    for w in range(0, T):
        rv_t = StageRandomVector(w)
        rc.append(rv_t)
        if w > 0:
            rv_t.addRandomElement('evapotranspiration' , [ow['evapotranspiration'] for ow in O[w]])
            rv_t.addRandomElement('rainfall' ,  [ow['rainfall'] for ow in O[w]])
        else:
            rv_t.addRandomElement('evapotranspiration' , [numpy.mean([ow['evapotranspiration'] for ow in O[w]])])
            rv_t.addRandomElement('rainfall' ,  [numpy.mean([ow['rainfall'] for ow in O[w]])])

    rc.preprocess_randomness()
    return rc

def random_builder_oos(data_OOS):
    #evapotranspiration  rainfall
    rc = RandomContainer()
    for w in range(0, T):
        rv_t = StageRandomVector(w)
        rc.append(rv_t)
        if w > 0:
            rv_t.addRandomElement('evapotranspiration' , [ow['evapotranspiration'] for ow in data_OOS[w]])
            rv_t.addRandomElement('rainfall' ,  [ow['rainfall'] for ow in data_OOS[w]])
        else:
            rv_t.addRandomElement('evapotranspiration' , [numpy.mean([ow['evapotranspiration'] for ow in data_OOS[w]])])
            rv_t.addRandomElement('rainfall' ,  [numpy.mean([ow['rainfall'] for ow in data_OOS[w]])])

    rc.preprocess_randomness()
    return rc


def model_builder(stage):
    
    m = Model('MilkPowder_%i' % (stage))
    # Parameters
    O = parameters["niwa_data"]
    
    
    Pm = parameters["maximum_pasture_cover"]  # maximum pasture-cover
    Pn = parameters["number_of_pasture_cuts"]  # number of pasture growth curve cuts
    gm = parameters["maximum_growth_rate"]  # pasture growth curve coefficient
    beta = parameters["harvesting_efficiency"]  # efficiency of harvesting pasture-cover into supplement
    np = parameters["pasture_energy_density"]  # net energy content of pasture (MJ/kgDM)
    ns = parameters["supplement_energy_density"]  # net energy content of supplement (MJ/kgDM)
    # index of soil fertility estimated from average seasonal pasture growth
    kappa = parameters["soil_fertility"]  # actual growth was too low
    
    # pasture growth as a function of pasture cover
    def g(p, gmax=gm, pmax=Pm):
        return 4 * gmax / pmax * p * (1 - p / pmax)
    
    # derivative of g(p) w.r.t. pasture cover
    def dgdt(p, gmax=gm, pmax=Pm):
        return 4* gmax / pmax * (1 - 2 * p / pmax)
    
    # States P, Q, W, C, M
    P = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='P')  # pasture cover (kgDM/Ha)
    Q = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Q')  # supplement storage (kgDM)
    W = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='W')  # soil moisture (mm)
    C = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='C')  # number of cows milking
    M = m.addVar(lb=-parameters["maximum_milk_production"], ub=-parameters["maximum_milk_production"], vtype=GRB.CONTINUOUS, name="M")
    # State in previous stage
    P0 = m.addVar(lb=parameters["initial_pasture_cover"], ub=parameters["initial_pasture_cover"], vtype=GRB.CONTINUOUS, name='P0')
    Q0 = m.addVar(lb=parameters["initial_storage"], ub=parameters["initial_storage"], vtype=GRB.CONTINUOUS, name='Q0')
    W0 = m.addVar(lb=parameters["initial_soil_moisture"], ub=parameters["initial_soil_moisture"], vtype=GRB.CONTINUOUS, name='W0')
    C0 = m.addVar(lb=parameters["stocking_rate"], ub=parameters["stocking_rate"], vtype=GRB.CONTINUOUS, name='C0')  # number of cows milking
    M0 = m.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, name="M0")
    
    # Variables
    b = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='b')  # quantity of supplement to feed (kgDM)
    h = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='h')  # quantity of supplement to harvest (kgDM/Ha)
    i = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='i')  # irrigate farm (mm/Ha)
    fs = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='fs')  # feed herd stored pasture (kgDM)
    fp = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='fp')  # feed herd pasture (kgDM)
    ev = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='ev')  # evapotranspiration rate
    gr = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='gr')  # potential growth
    mlk = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='mlk')  # milk production (MJ)
    milk = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='milk')  # milk production (kgMS)
    delta = m.addVars([1, 2], lb=0, vtype=GRB.CONTINUOUS)  # penalties
    m.update()
    
    #RHS Noise - assumed to be year one of the data for the the first stage (week)
    evatra_ini = numpy.mean([ow['evapotranspiration'] for ow in O[stage]])
    rainfall_ini = numpy.mean([ow['rainfall'] for ow in O[stage]])

    evatra = m.addVar(lb=evatra_ini, ub=evatra_ini, vtype=GRB.CONTINUOUS, name='evapotranspiration')
    rainfall = m.addVar(lb=rainfall_ini, ub=rainfall_ini, vtype=GRB.CONTINUOUS, name='rainfall')
    
    # Expression for later usege:
    energy_req = parameters["stocking_rate"] * (parameters["energy_for_pregnancy"][stage] + parameters["energy_for_maintenance"] + parameters["energy_for_bcs_dry"][stage]) + \
            C0 * (parameters["energy_for_bcs_milking"][stage] - parameters["energy_for_bcs_dry"][stage])
    
    # Constraints
    m.addConstr((P <= P0 + 7 * gr - h - fp), 'Pasture Cover')
    m.addConstr((Q <= Q0 + beta * h - fs), 'Supplement Storage')
    m.addConstr((Q <= Q0 + beta * h - fs), 'Supplement Storage')
    m.addConstr((C <= C0), 'CowsAmount')
    m.addConstr((W <= parameters["maximum_soil_moisture"]), 'Soil Limit')
    m.addConstr((milk <= mlk / parameters["energy_content_of_milk"][stage]), 'milkRel')
    m.addConstr((M <= M0 + milk), 'milk_BALANCE')
    m.addConstr((np * (fp + fs) + ns * b >= energy_req + mlk), 'EnergyBalance')
    m.addConstr((mlk <= parameters["max_milk_energy"][stage] * C0), 'maximum milk')
    m.addConstr((milk >= parameters["min_milk_production"] * C0), 'minimum milk')
    m.addConstr((gr <= kappa[stage] * ev / 7), 'pasture growth constraints1')
    for pbar in numpy.linspace(0,Pm,Pn):#Pn
        m.addConstr((gr <= g(pbar) + dgdt(pbar) * (P0 - pbar + 1e-2)), 'growth_aprox%f' %(pbar))
    m.addConstr((i <= parameters["maximum_irrigation"]), 'max_irrigation')
    
    m.addConstr((ev <= evatra), 'evapotranspiration_ctr')
    m.addConstr((W <= W0 - ev + rainfall + i), 'soil-mosture-balance')
    
    if stage >= parameters["maximum_lactation"]:
        m.addConstr((C<=0), name='dry_off') # dry off by end of week 44 (end of may)
    
    m.update()
    
    
    
    # a maximum rate of supplementation - for example due to FEI limits
    cow_per_day = parameters["stocking_rate"] * 7
    m.addConstr((delta[1]>=0 ), 'max_rate_d1_1')
    m.addConstr((delta[1] >= cow_per_day * (0.00 + 0.25 * (b / cow_per_day - 3)) ), 'max_rate_d1_2')
    m.addConstr((delta[1] >= cow_per_day * (0.25 + 0.75 * (b / cow_per_day - 4)) ), 'max_rate_d1_3')
    m.addConstr((delta[1] >= cow_per_day * (0.75 + 1.00 * (b / cow_per_day - 5)) ), 'max_rate_d1_4')
    price  = 6 #parameters["prices"][stage][price]
    objfunc = parameters["supplement_price"] * b + parameters["cost_irrigation"] * i + parameters["harvest_cost"] * h  + delta[1] + 100 * delta[2] - 0.0001*W 
    if stage == 51: #LAST WEEK
        objfunc = objfunc  - M * price
        m.addConstr((P + delta[2] >= parameters["final_pasture_cover"]), 'minimum_pasture_cover')
    m.setObjective(objfunc, GRB.MINIMIZE)
    m.update()
    
    in_states = [var.VarName for var in [P0, Q0, W0, C0, M0]]
    out_states = [var.VarName for var in [P, Q, W, C, M]]
    rhs_vars = [evatra.VarName, rainfall.VarName] 
    return m, in_states, out_states, rhs_vars 

    #print_model(m)

if __name__ == '__main__':
    load_algorithm_options()
    T=52
    N_train = 20
    N_test = len(parameters["niwa_data"][0]) - N_train
    W = len(parameters["niwa_data"])
    Y = len(parameters["niwa_data"][0])
    O = [[parameters["niwa_data"][w][y] for y in range(N_train)]for w in range(W)]
    data_OOS  = [[parameters["niwa_data"][w][y] for y in range(N_train, Y)]for w in range(W)]
    algo = SDDP(T, model_builder, random_builder,  risk_measure =DistRobust, dro_solver = DiscreteWassersteinInnerSolver, dro_solver_params = {'norm': 1 , 'radius':0})
    #algo = SDDP(T, model_builder, random_builder,  risk_measure =DistRobustWasserstein,  radius = 10)
    lbs = algo.run(instance_name='MILK_52')
    out_of_sample_rnd_cont = random_builder_oos(data_OOS)
    sim_result = algo.simulate_policy(5000, out_of_sample_rnd_cont)