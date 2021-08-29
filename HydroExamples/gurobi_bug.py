'''
Created on May 20, 2019

@author: dduque
'''

from gurobipy import GRB, Model, tupledict, read
import os 
import sys
hydro_path = os.path.dirname(os.path.realpath(__file__))
parent_path= os.path.abspath(os.path.join(hydro_path, os.pardir))
sys.path.append(parent_path)
os.chdir(hydro_path)

def gv(m,n):
    var = m.getVarByName(n)
    print(var)
    return var


'''
Read model at iteration 3 and its basis. 
'''
m3 = read('./sp0_iter3.mps')
m3.read('./sp0_iter3.bas')
m3.optimize()
#Print solutions 
for v in m3.getVars():
    print(v.VarName, v.X, v.RC, v.LB, v.UB)

#Manually add the cut
#Add (benders) cut
cut_exp = 80 * gv(m3,'reservoir_level[0]') + 25 * gv(m3,'reservoir_level[1]') + 52.88232369018938 * gv(m3,'inflow[0,1]') + 8.310453412619006 * gv(m3,'inflow[1,1]') + 1.0 * gv(m3,'oracle[0][0]') 
print(cut_exp)
m3.addConstr(cut_exp>= 23918.67191367184, 'cut4')
m3.optimize()
#print sols with the cut
for v in m3.getVars():
    print(v.VarName, v.X, v.RC, v.LB, v.UB)
    
####
## SAME X is obtained, (wrong) end of the method!
####

# Reset the model with the constraint and optimize
#Now we get a new X.  
print('Reseting problem ')
m3.reset()
m3.optimize()
for v in m3.getVars():
    print(v.VarName, v.X, v.RC, v.LB, v.UB)



print('\n Test 2 ')
m4 = read('./sp0_iter4.mps')
m4.read('./sp0_iter3.bas')
m4.optimize()
print(m4.getVars())


'''
Read the model at iter 4, but fix one variable
to the solution of the model at iter 3. 
'''
print('\n Test 3 ')
m4 = read('./sp0_iter4.mps')
v = gv(m4,'reservoir_level[0]')
v.LB = 6.5685528470991684e+01
v.UB = 6.5685528470991684e+01
m4.optimize()
print(m4.getVars())

'''
Read the model at iter 4, read the basis from model at iteration 3 (which is an ifeasible basis)
and  fix one variable to the solution of the model at iter 3. 
'''
print('\n Test 4 ')
m4 = read('./sp0_iter4.mps')
m4.read('./sp0_iter3.bas')
v = gv(m4,'reservoir_level[1]')
v.LB = 30
v.UB = 30
m4.optimize()
print(m4.getVars())

