'''
Created on Jun 27, 2018

@author: dduque, based on Knitro example

Auxiliary modul to solve the nonlinear problem associated to the worst-case 
expectation in DRO.
'''
"""
Auxiliary modul to solve the nonlinear problem associated to the worst-case 
expectation in DRO.
"""

import sys
sys.path.append('/Applications/knitro-11.0.1-z-MacOS-64/examples/Python/')

#from knitro import *
#from knitroNumPy import *
import numpy as np



def solve_worst_case_expectation(outcomes_org,outcomes_des,branches,costs,distances,r_w):
    #outcomes_org,outcomes_des,scenarios,branches,costs,distances
    #===========================================================================
    # outcomes_org = 10
    # outcomes_des = 30
    # scenarios = 1000
    # r_w = 100
    # #branches = np.array([[3,2,1,2,1,2,3,4,5],[1,2,3,4,4,3,2,1,1],[3,3,3,1,2,2,3,2,1]])
    # #costs = np.array([100,200,140,400,101,302,432,900,1020])
    # branches = np.random.randint(0,high=6,size=(outcomes_des,scenarios))
    # costs = np.random.randint(50,high=500,size=scenarios)
    # #branches = np.array([[3,1],[1,6],[3,0]])
    # #costs = np.array([100,150])
    # 
    # distances = np.random.randint(1,high=10,size=(outcomes_org,outcomes_des))
    # if outcomes_org == outcomes_des:
    #     distances = np.sqrt(distances.transpose().dot(distances))
    # for i in range(len(distances)):
    #     distances[i,i] = 0 
    #===========================================================================
    #distances = np.array([[0,2,5],[2,0,3],[5,3,0]])
    #distances = np.array([[0,2,5],[2,0,3]])
    
    #----------------------------------------------------------------
    #   METHOD evaluateFC
    #----------------------------------------------------------------
    def evaluateFC (x, c):
        # print('used')
        p = x[0:outcomes_des].reshape((outcomes_des, 1))
        z = x[outcomes_des:].reshape((outcomes_org, outcomes_des))
        obj = np.sum(costs * np.product(p ** branches, axis=0)) / np.sum(np.product(p ** branches, axis=0))
        ctr_index = 0
        for o in range(len(z)):
            c[ctr_index] = z[o].sum()
            ctr_index += 1
        for d in range(len(z[0])):
            c[ctr_index] = z[:, d].sum() - p[d, 0]
            ctr_index += 1
           
        c[ctr_index] = np.sum(distances * z)
        # print('x: ',x,'obj: ', obj, '\n  ctr: ', c)
        return obj
    
    def build_problem(n,m):
        xIx = np.arange(n)
        pIx = xIx[0:outcomes_des].reshape((outcomes_des,1))
        zIx = xIx[outcomes_des:].reshape((outcomes_org,outcomes_des))
        jacIxConstr = np.array([],np.int32)
        jacIxVar = np.array([],np.int32)
        jac = np.array([])
        ctr_index = 0 
        for o in range(outcomes_org):
            jacIxConstr = np.hstack((jacIxConstr,[ctr_index]*outcomes_des))
            jacIxVar = np.hstack((jacIxVar,zIx[o])) 
            jac = np.hstack((jac,[1]*outcomes_des)) 
            ctr_index += 1
        for d in range(outcomes_des):
            jacIxConstr = np.hstack((jacIxConstr,[ctr_index]*(outcomes_org+1)))
            jacIxVar = np.hstack((jacIxVar,zIx[:,d],pIx[d]))
            jac = np.hstack((jac,[1]*(outcomes_org),[-1]))
            ctr_index += 1
        jacIxConstr = np.hstack((jacIxConstr,[ctr_index]*np.size(zIx)))
        jacIxVar = np.hstack((jacIxVar,zIx.flatten()))
        jac = np.hstack((jac,distances.flatten()))
        return  jacIxConstr , jacIxVar , jac
    #----------------------------------------------------------------
    #   METHOD evaluateGA
    #----------------------------------------------------------------
    def evaluateGA (x, objGrad):
        '''
        Evaluates the gradient gradient
        '''
        p = x[0:outcomes_des].reshape((outcomes_des,1))
        print(x[0:outcomes_des])
        for i in range(len(p)):
            pc = p.copy()
            pc[i] = 1
            no_i = np.squeeze(branches[i]*costs*np.product(pc**branches, axis=0))
            #print('noi ', no_i)
            wi_i = np.zeros_like(no_i)
            wi_i[branches[i]>0] = p[i]**(branches[i,branches[i]>0]-1)
            #print('wio ', wi_i)
            objGrad[i] = np.sum(no_i*wi_i)
            #print(no_i*wi_i)
        #print( objGrad[0:outcomes_des])
    
    #----------------------------------------------------------------
    #   METHOD evaluateH
    #----------------------------------------------------------------
    def evaluateH(x, lambda_, sigma, hess):
        pass
        #==============================================================================
        #     hess[0] = sigma * ( (-400.0 * x[1]) + (1200.0 * x[0]*x[0]) + 2.0)
        #     hess[1] = (sigma * (-400.0 * x[0])) + lambda_[0]
        #     hess[2] = (sigma * 200.0) + (lambda_[1] * 2.0)
        #==============================================================================
    
    
    #----------------------------------------------------------------
    #   MAIN METHOD FOR TESTING
    #----------------------------------------------------------------
    
    '''
    Define problem input
    '''
    n = outcomes_des + outcomes_org*outcomes_des
    objGoal = KTR_OBJGOAL_MAXIMIZE
    objType = KTR_OBJTYPE_GENERAL;
    bndsLo = np.array ([0 for i in range(n)])
    bndsUp = np.array ([1 for i in range(n)])
    m = outcomes_org + outcomes_des + 1
    cType = np.array ([ KTR_CONTYPE_LINEAR for j in range(m)], np.int64)
    cBndsLo = np.array ([1.0/outcomes_org for j in range(outcomes_org)] + [0 for j in range(outcomes_des)] + [0])
    cBndsUp = np.array ([1.0/outcomes_org for j in range(outcomes_org)] + [0 for j in range(outcomes_des)] + [r_w])
    
    jacIxConstr, jacIxVar, ctrJac = build_problem(n,m)
    
    #jacIxConstr = np.array ([ 0, 0, 0, 1, 1, 1, 2, 2, 2, 3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,6,6,6,6,6], np.int32)
    #jacIxVar    = np.array ([ 3, 4, 5, 6, 7, 8, 9,10,11, 3,6,9,0,4,7,10,1,5,8,11,2,3,4,5,6,7,8,9,10,11], np.int32)
    hessRow = np.array ([0,0,0,1,1,1,2,2,2], np.int32)
    hessCol = np.array ([0,1,2,0,1,2,0,1,2], np.int32)
    
    xInit = np.array([1/outcomes_des]*n)
    #xInit = np.array([0.2,0.2,0.6,0.2,0,1/3-0.2,0,0.2,1/3-0.2,0,0,1/3])
    xInit = None#np.zeros(n)+1
    
    #---- SETUP AND RUN KNITRO TO SOLVE THE PROBLEM.
    
    #---- CREATE A NEW KNITRO SOLVER INSTANCE.
    kc = KTR_new()
    if kc == None:
        raise RuntimeError ("Failed to find a Knitro license.")
    
    #---- DEMONSTRATE HOW TO SET KNITRO PARAMETERS.
    if KTR_set_int_param_by_name(kc, 'outlev', 0):
        raise RuntimeError ("Error setting parameter 'outlev'")
    if KTR_set_int_param_by_name(kc, "hessopt", 2):
       raise RuntimeError ("Error setting parameter 'hessopt'")
    if KTR_set_int_param_by_name(kc, "multistart", 0):
       raise RuntimeError ("Error setting parameter 'multistart'")
    if KTR_set_int_param_by_name(kc, "gradopt", 2   ):
        raise RuntimeError ("Error setting parameter gradopt")
    if KTR_set_double_param_by_name(kc, "feastol", 1.0E-6):
        raise RuntimeError ("Error setting parameter 'feastol'")
    if KTR_set_double_param_by_name(kc, "ftol", 1.0E-3):
        raise RuntimeError ("Error setting parameter 'feastol'")
    if KTR_set_int_param_by_name(kc, "ftol_iters", 10):
        raise RuntimeError ("Error setting parameter 'feastol'")
    
    
    KTR_set_int_param_by_name(kc, "derivcheck", 0)
    
    KTR_set_double_param_by_name(kc, "derivcheck_tol", 1E-5)
    
    KTR_set_int_param_by_name(kc, "cg_precond", 0)
    KTR_set_int_param_by_name(kc, "algorithm", 0)
    
    #------------------------------------------------------------------ 
    #     FUNCTION callbackEvalFC
    #------------------------------------------------------------------
     # # The signature of this function matches KTR_callback in knitro.h.
     #  Only "obj" and "c" are modified.
     # #
    def callbackEvalFC (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, obj, c, objGrad, jac, hessian, hessVector, userParams):
        if evalRequestCode == KTR_RC_EVALFC:
            obj[0] = evaluateFC(x, c)
            return 0
        else:
            return KTR_RC_CALLBACK_ERR
    
    #------------------------------------------------------------------
    #     FUNCTION callbackEvalGA
    #------------------------------------------------------------------
     ## The signature of this function matches KTR_callback in knitro.h.
     #  Only "objGrad" and "jac" are modified.
     ##
    def callbackEvalGA (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, obj, c, objGrad, jac, hessian, hessVector, userParams):
        if evalRequestCode == KTR_RC_EVALGA:
            jac = ctrJac.copy()
            evaluateGA(x, objGrad)
            return 0
        else:
            return KTR_RC_CALLBACK_ERR
    
    #------------------------------------------------------------------
    #     FUNCTION callbackEvalH
    #------------------------------------------------------------------
     ## The signature of this function matches KTR_callback in knitro.h.
     #  Only "hessian" or "hessVector" is modified.
     ##
    def callbackEvalH (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, obj, c, objGrad, jac, hessian, hessVector, userParams):
        if evalRequestCode == KTR_RC_EVALH:
            evaluateH(x, lambda_, 1.0, hessian)
            return 0
        elif evalRequestCode == KTR_RC_EVALH_NO_F:
            evaluateH(x, lambda_, 0.0, hessian)
            return 0
        else:
            return KTR_RC_CALLBACK_ERR
    
    
    def cb_check_comp_grad(evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, obj, c, objGrad, jac, hessian, hessVector, userParams):
        print('my own callback')
        if evalRequestCode == KTR_RC_NEWPOINT:
            print('my own callback')
            
            
            
    #---- REGISTER THE CALLBACK FUNCTIONS THAT PERFORM PROBLEM EVALUATION.
    #---- THE HESSIAN CALLBACK ONLY NEEDS TO BE REGISTERED FOR SPECIFIC
    #---- HESSIAN OPTIONS (E.G., IT IS NOT REGISTERED IF THE OPTION FOR
    #---- BFGS HESSIAN APPROXIMATIONS IS SELECTED).
    if KTR_set_func_callback(kc, callbackEvalFC):
        raise RuntimeError ("Error registering function callback.")
    if KTR_set_grad_callback(kc, callbackEvalGA):
        raise RuntimeError ("Error registering gradient callback.")
    if KTR_set_newpt_callback(kc, cb_check_comp_grad):
        raise RuntimeError ("Error registering newpt callback.")
    #==============================================================================
    # if KTR_set_hess_callback(kc, callbackEvalH):
    #     raise RuntimeError ("Error registering hessian callback.")
    #==============================================================================
    
    #---- INITIALIZE KNITRO WITH THE PROBLEM DEFINITION.
    ret = KTR_init_problem (kc, n, objGoal, objType, bndsLo, bndsUp,
                                    cType, cBndsLo, cBndsUp,
                                    jacIxVar, jacIxConstr,
                                    hessRow, hessCol,
                                    xInit, None)
    if ret:
        raise RuntimeError ("Error initializing the problem, Knitro status = %d" % ret)
    
    #---- SOLVE THE PROBLEM.
    #----
    #---- RETURN STATUS CODES ARE DEFINED IN "knitro.h" AND DESCRIBED
    #---- IN THE KNITRO MANUAL.
    x       = np.zeros (n)
    lambda_ = np.zeros (m + n)
    obj     = np.array ([0.0])
    nStatus = KTR_solve (kc, x, lambda_, 0, obj,
                             None, None, None, None, None, None)
    
    print
    print
    if nStatus not in [0,-101,-102,-103]:
        print (obj, x[0:outcomes_des], nStatus)
        raise RuntimeError ("Knitro failed to solve the problem, final status = %d" % nStatus)
    else:
        #---- AN EXAMPLE OF OBTAINING SOLUTION INFORMATION.
        print ("Knitro>fea.viol,KKT_viol= %e %e" %(KTR_get_abs_feas_error (kc), KTR_get_abs_opt_error (kc) ) )
        print (obj, x[0:outcomes_des], nStatus)
        my_grad = [0]*n
        ret1 = KTR_get_objgrad_values(kc,my_grad )
        my_jac = [0]*len(jacIxConstr)
        ret2= KTR_get_jacobian_values(kc,my_jac)
        #print('')
        #print(ret1, ret2, my_grad)
        #print(np.array(my_jac))
        return obj, x[0:outcomes_des], nStatus
    #---- BE CERTAIN THE NATIVE OBJECT INSTANCE IS DESTROYED.
    KTR_free(kc)

#+++++++++++++++++++ End of source file +++++++++++++++++++++++++++++
if __name__ == '__main__':
    np.set_printoptions(precision=15,threshold=np.nan)
    np.random.seed(0)
    outcomes_org = 5
    outcomes_des = 5
    scenarios = 1000
    r_w = 1
    #branches = np.array([[3,2,1,2,1,2,3,4,5],[1,2,3,4,4,3,2,1,1],[3,3,3,1,2,2,3,2,1]])
    #costs = np.array([100,200,140,400,101,302,432,900,1020])
    branches = np.random.randint(0,high=6,size=(outcomes_des,scenarios))
    costs = np.random.randint(50,high=500,size=scenarios)
    #branches = np.array([[3,1],[1,6],[3,0]])
    #costs = np.array([100,150])
    
    distances = np.random.randint(1,high=10,size=(outcomes_org,outcomes_des))
    if outcomes_org == outcomes_des:
        distances = np.sqrt(distances.transpose().dot(distances))
    for i in range(np.minimum(len(distances),len(distances[0]))):
        distances[i,i] = 0 
    solve_worst_case_expectation(outcomes_org,outcomes_des,branches,costs,distances,r_w)
