#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:13:29 2018

@author: dduque
"""

from gurobipy import *
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from SDDP.RiskMeasures import DistRobustWassersteinCont
newsvendro_path = os.path.dirname(os.path.realpath(__file__))
from OutputAnalysis.SimulationAnalysis import SimResult, plot_sim_results



def create_nv_model(N,Xi,K,a,b,L,C,d,dro_radius):
    '''
    Creates a model of the news vendor where the loss function
    if determined by the affine pieces in a[k] and b[k] 
    '''
    model = Model('ContWassersteinNewsVendor')
    model.params.OutputFlag = 0 
    model.params.Solver = 1
    
    #Decision Variables
    x = model.addVar(lb=0,ub=GRB.INFINITY,obj=0,vtype=GRB.CONTINUOUS, name='x')
    #x = model.addVar(lb=0,ub=0,obj=0,vtype=GRB.CONTINUOUS, name='x')
    s = model.addVars(N,lb=-GRB.INFINITY,ub=GRB.INFINITY,obj=[1/len(N) for _ in range(n)], vtype=GRB.CONTINUOUS, name='s')
    lam = model.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,obj=dro_radius,vtype=GRB.CONTINUOUS,name='lambda_var')
    gam = tupledict()
    for i in N:
        gam_i = model.addVars(K[i],[i],L,lb=0,ub=GRB.INFINITY,obj=0, vtype=GRB.CONTINUOUS, name='gamma_var')
        gam.update(gam_i)
    model.update()
    piece_ctrs = {}
    for i in N:
        for k in K[i]:
            piece_ctrs[(k,i)] = model.addConstr(rhs=(b[k,i] + a[k,i]*Xi[i] + quicksum((d[l]-C[l]*Xi[i])*gam[k,i,l] for l in L)), 
                                                sense=GRB.GREATER_EQUAL, 
                                                lhs=s[i], 
                                                name='piece[%i,%i]' %(k,i))
    norm_ctrs = {}
    for i in N:
        for k in K[i]:
            linexp = quicksum(C[l]*gam[k,i,l] for l in L) - a[k,i]
            norm_ctrs[(k, i, -1)] = model.addConstr(lhs=lam - linexp, sense=GRB.GREATER_EQUAL, rhs=0, name='InfNormNeg[%i,%i]' % (k, i))  # RESTA
            norm_ctrs[(k, i, 1)] = model.addConstr(lhs=lam + linexp, sense=GRB.GREATER_EQUAL, rhs=0, name='InfNormPos[%i,%i]' % (k, i))  # SUMA     
    model.update()
    
    return model, x, s, lam, gam, piece_ctrs, norm_ctrs


def gen_new_piece(model, x, s, lam,  gam, piece_ctrs, norm_ctrs, N,Xi,K,a,b,L,C,d , loss_func):
    '''
    Generates new pieces of the loss function and adds them in the model.
    It also generates new variables \gamma associated with the new pieces. 
    '''
    x_k = x.X
    for i in N:
        k = len(K[i])
        K[i].append(k)
        h_k_i = loss_func(x_k, Xi[i], order=0)
        grad = loss_func(x_k, Xi[i], order =1) #[grad_x, grad_Xi]
        a[(k,i)]=grad[1] #Grad of Xi
        b[(k,i)]=h_k_i+grad[0]*(x-x_k)-grad[1]*Xi[i]
        new_gam = model.addVars([k],[i],L,lb=0,ub=GRB.INFINITY,obj=0, vtype=GRB.CONTINUOUS, name='gamma_var')
        model.update()
        for ng in new_gam:
            if ng in gam:
                raise 'repeated key'
            
        gam.update(new_gam)
        piece_ctrs[(k,i)] = model.addConstr(rhs=(b[k,i] + a[k,i]*Xi[i] + quicksum((d[l]-C[l]*Xi[i])*gam[k,i,l] for l in L)), 
                                                sense=GRB.GREATER_EQUAL, 
                                                lhs=s[i], 
                                                name='piece[%i,%i]' %(k,i))
        linexp = quicksum(C[l]*gam[k,i,l] for l in L) - a[k,i]
        norm_ctrs[(k, i, -1)] = model.addConstr(lhs=lam - linexp, sense=GRB.GREATER_EQUAL, rhs=0, name='InfNormNeg[%i,%i]' % (k, i))  # RESTA
        norm_ctrs[(k, i, 1)] = model.addConstr(lhs=lam + linexp, sense=GRB.GREATER_EQUAL, rhs=0, name='InfNormPos[%i,%i]' % (k, i))  # SUMA  
    
    print('Gen piece Nominal')   
    model.update()

def gen_new_piece_on_wcd(model, x, s, lam,  gam, piece_ctrs, norm_ctrs, N,Xi,K,a,b,L,C,d , loss_func, new_support, support_map):
    '''
    Generates new pieces of the loss function and adds them in the model.
    It also generates new variables \gamma associated with the new pieces. 
    '''
    
    x_k = x.X
    N_counter = np.zeros(len(N))
    for i in N:
        for i_hat in range(len(new_support)):
            #i = support_map[i_hat] #Original support point
            k = len(K[i])
            k = str(k)+'wc' + str(N_counter[i])
            N_counter[i] += 1 
            K[i].append(k)
            #New piece is computed wrt to the new support point
            h_k_i = loss_func(x_k, new_support[i_hat], order=0)
            grad = loss_func(x_k, new_support[i_hat], order =1) #[grad_x, grad_Xi]
            a[(k,i)]=grad[1] #Grad of Xi
            b[(k,i)]=h_k_i+grad[0]*(x-x_k)-grad[1]*new_support[i_hat]
            new_gam = model.addVars([k],[i],L,lb=0,ub=GRB.INFINITY,obj=0, vtype=GRB.CONTINUOUS, name='gamma_var')
            model.update()
            for ng in new_gam:
                if ng in gam:
                    raise 'repeated key'
            gam.update(new_gam)
           
            piece_ctrs[(k,i)] = model.addConstr(rhs=(b[k,i] + a[k,i]*Xi[i] + quicksum((d[l]-C[l]*Xi[i])*gam[k,i,l] for l in L)), 
                                                    sense=GRB.GREATER_EQUAL, 
                                                    lhs=s[i], 
                                                    name='piece[%s,%i]' %(k,i))
            linexp = quicksum(C[l]*gam[k,i,l] for l in L) - a[k,i]
            norm_ctrs[(k, i, -1)] = model.addConstr(lhs=lam - linexp, sense=GRB.GREATER_EQUAL, rhs=0, name='InfNormNeg[%s,%i]' % (k, i))  # RESTA
            norm_ctrs[(k, i, 1)] = model.addConstr(lhs=lam + linexp, sense=GRB.GREATER_EQUAL, rhs=0, name='InfNormPos[%s,%i]' % (k, i))  # SUMA  
    print('Gen piece WC')   
    model.update()

def solve_model(model, x, s, lam,  gam, piece_ctrs, norm_ctrs, N, K , Xi):
    model.optimize()
    if model.status == GRB.OPTIMAL:
        print(model.ObjVal, x.X, 'lam: ', lam.X, [s[i].X for i in N])
        #=======================================================================
        # for c in model.getConstrs():
        #     if c.Pi  > 1E-8:
        #         print(c.ConstrName, c.Pi)
        #=======================================================================
        print('orig supp' , Xi)
        new_support = []
        pmf = []
        org_supp_id = {}
        n_wc_sup = 0
        for (k,i) in piece_ctrs:
            if piece_ctrs[(k,i)].Pi > 1E-8:
                new_atom = Xi[i]  + (norm_ctrs[(k,i,1)].Pi - norm_ctrs[(k,i,-1)].Pi)/piece_ctrs[(k,i)].Pi
                new_support.append(new_atom)
                pmf.append(piece_ctrs[(k,i)].Pi)
                if n_wc_sup not in org_supp_id:
                    org_supp_id[n_wc_sup] = set()
                org_supp_id[n_wc_sup].add(i)
                n_wc_sup +=1
        new_support = np.array(new_support)
        pmf = np.array(pmf)
        org_supp_keys = np.arange(len(pmf))
        supp_argsort = np.argsort(new_support) 
        pmf = pmf[supp_argsort]
        org_supp_keys = org_supp_keys[supp_argsort]
        new_support.sort()

        for i in range(len(new_support)-1, 0, -1):
            if np.abs(new_support[i] -new_support[i-1])<1E-8:
                new_support = np.delete(new_support, obj=i)
                rep_prob = pmf[i]
                pmf =np.delete(pmf,obj =i)
                pmf[i-1] += rep_prob 
                temp_key = org_supp_keys[i]
                temp_list = org_supp_id[temp_key].copy()
                org_supp_keys = np.delete(org_supp_keys, obj = i)
                new_key = org_supp_keys[i-1]
                org_supp_id[new_key].update(temp_list)
                org_supp_id.pop(temp_key)
                           
        support_mapping  = [sorted(list(org_supp_id[k])) for k in org_supp_keys]
        print('new supp ', new_support)
        print('new pmf' , pmf)
        print('supp mapping' , support_mapping)
        if np.abs(pmf.sum()-1)>1E-8:
            raise 'PMF not right'
        return  new_support , pmf, support_mapping
    
    
def nv_cutting(N,Xi,K,a,b,L,C,d,dro_radius, loss_func):
    '''
    Solves DRO NV for the loss function given by pieces in a and b
    '''
    model, x, s, lam, gam, piece_ctrs, norm_ctrs = create_nv_model(N, Xi, K, a, b, L, C, d, dro_radius)
    new_supp, new_pmf, new_supp_map = solve_model(model, x, s, lam, gam, piece_ctrs, norm_ctrs, N, K , Xi)
    
    for iter in range(23):
        assert len(model.getConstrs()) == len(piece_ctrs) + len(norm_ctrs)
        gen_new_piece(model, x, s, lam, gam, piece_ctrs, norm_ctrs, N, Xi, K, a, b, L, C, d, loss_func)
        new_supp, new_pmf, new_supp_map = solve_model(model, x, s, lam, gam, piece_ctrs, norm_ctrs, N, K , Xi)
        gen_new_piece_on_wcd(model, x, s, lam, gam, piece_ctrs, norm_ctrs, N, Xi, K, a, b, L, C, d, loss_func, new_supp,new_supp_map)
        new_supp, new_pmf, new_supp_map = solve_model(model, x, s, lam, gam, piece_ctrs, norm_ctrs, N, K , Xi)
    return new_supp, new_pmf, x.X
    


def print_model(model):
    print(model.getObjective())
    for c in model.getConstrs():
        print(c.ConstrName, model.getRow(c), c.Sense, c.RHS)
      

def solve_cont_wasserstain(N,K,L, xi_n, C, d, r, c, dro_radius):
    
    model = Model('ContWassersteinNewsVendor')
    model.params.OutputFlag = 0 
    model.params.Solver = 1
    x = model.addVar(lb=0,ub=GRB.INFINITY,obj=0,vtype=GRB.CONTINUOUS, name='x')
    s = model.addVars(N,lb=-GRB.INFINITY,ub=GRB.INFINITY,obj=[1/len(N) for _ in range(n)], vtype=GRB.CONTINUOUS, name='s')
    lam = model.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,obj=dro_radius,vtype=GRB.CONTINUOUS,name='lambda_var')
    gam = model.addVars(K,N,L,lb=0,ub=GRB.INFINITY,obj=0, vtype=GRB.CONTINUOUS, name='gamma_var')
    model.update()
    piece_ctrs = {}
    #Piece 0:
    for i in N:
        piece_ctrs[(0,i)] = model.addConstr(rhs=((c-r)*x+quicksum((d[l]-C[l]*xi_n[i])*gam[0,i,l] for l in L)), sense=GRB.GREATER_EQUAL, lhs=s[i], name='piece[0,%i]' %i)
    #Piece 1:
    for i in N:
        piece_ctrs[(1,i)] = model.addConstr(rhs=(c*x-r*xi_n[i]+quicksum((d[l]-C[l]*xi_n[i])*gam[1,i,l] for l in L)), sense=GRB.GREATER_EQUAL, lhs=s[i],name='piece[1,%i]' %i)
    
    norm_ctrs = {}
    for k in K:
        for i in N:
            linexp = LinExpr()
            if k == 0:
                linexp = quicksum(C[l]*gam[k,i,l] for l in L)
            else:
                linexp = quicksum(C[l]*gam[k,i,l] for l in L) + r
            norm_ctrs[(k,i,-1)] = model.addConstr(lhs=lam-linexp, sense=GRB.GREATER_EQUAL, rhs=0,name='InfNormNeg[%i,%i]' %(k,i)) #RESTA
            norm_ctrs[(k,i, 1)] =model.addConstr(lhs=lam+linexp, sense=GRB.GREATER_EQUAL, rhs=0,name='InfNormPos[%i,%i]' %(k,i)) #SUMA
    model.update()
    
    model.optimize()
   
    if model.status == GRB.OPTIMAL:
        print(model.ObjVal, x.X, 'Lambda: ', lam.X, [s[i].X for i in N])
        for c in model.getConstrs():
            if c.Pi  > 1E-8:
                print(c.ConstrName, c.Pi)
        print('orig supp' , xi_n)
        new_support = []
        pmf = []
        for (k,i) in piece_ctrs:
            if piece_ctrs[(k,i)].Pi > 1E-8:
                new_atom = xi_n[i]  + (norm_ctrs[(k,i,1)].Pi - norm_ctrs[(k,i,-1)].Pi)/piece_ctrs[(k,i)].Pi
                new_support.append(new_atom)
                pmf.append(piece_ctrs[(k,i)].Pi)
        new_support = np.array(new_support)
        pmf = np.array(pmf)
        supp_argsort = np.argsort(new_support) 
        pmf = pmf[supp_argsort]
        new_support.sort()

        for i in range(len(new_support)-1, 0, -1):
            if np.abs(new_support[i] -new_support[i-1])<1E-8:
                new_support = np.delete(new_support, obj=i)
                rep_prob = pmf[i]
                pmf =np.delete(pmf,obj =i)
                pmf[i-1] += rep_prob                
        print('new supp ', new_support)
        print('new pmf' , pmf)
        return x.X, new_support , pmf
    else:
        model.computeIIS()
        model.write("NewsVendorInf.ilp")
        os.system("open -a TextEdit filename Users/dduque/Desktop/NewsVendorInf.ilp")
        

      
def create_obj(c,rho):
    '''
    Creates a penalty function of the form
    f(x,d) = cx + h(x,d)
    h(x,d) = {0         if d<=x
              (d-x)^2   if d>x
    '''
    def news_vendor_obj(x, demand, order=0):
        if order == 0:  
            if demand<=x:
                return c*x
            else:
                return  rho*np.power(np.maximum(0,demand-x),2) + c*x
        elif order == 1:
            if demand<=x:
                return np.array([c,0])
            else:
                grad_x = -2*rho*(demand-x) + c
                grad_d = 2*rho*(demand-x)
            return np.array([grad_x,grad_d])
    return news_vendor_obj

def create_obj2(c,r):
    #y=\max\left(\left(c-p_3\right)q,cq-\left(p_1x\ -40\right),cq-\left(p_2x\ -20\right),cq-p_3x\ \right)
    def news_vendor_obj(x, xi_n):
        return np.maximum((c-r[0])*x, np.maximum(c*x-r[0]*xi_n, np.maximum(c*x-r[1]*xi_n +20, c*x-r[2]*xi_n +40)))
    return news_vendor_obj


def test_out_of_sample(x_star, test_set, obj_fun, instance):
    objs = obj_fun(x_star,test_set)
    sim_res = SimResult(instance,objs)
    x_bar = np.mean(objs)
    std  = np.std(objs)
    if instance == None:
        print('%10s %10.4f %10.4f %10.4f' %('EV_Policy', x_star, x_bar, std))
    else:
        print('%10.4f %10.4f %10.4f %10.4f' %(instance['risk_measure_params']['radius'], x_star, x_bar, std))
    return sim_res

    
    
if __name__ == '__main__':
    np.random.seed(0)
    C = [-1,1]
    
    rho = 1
    c = 2 #cost
    L = [0,1]
    news_vendor_fun = create_obj(c,rho)
   
    #density = np.append(np.append(np.random.lognormal(0,1,size=1000) ,np.random.normal(10,1,size=2000)), np.random.normal(15,2,size=1000))
    #density = np.abs(np.random.normal(50,25,size=2000))
    density = np.append(np.abs(np.random.binomial(100,0.02,size=20000)), np.abs(np.random.binomial(200,0.01,size=20000)))
    
    #===========================================================================
    # n, bins, patches = plt.hist(density, 100, normed=1, facecolor='green', alpha=0.75)
    # y = mlab.normpdf( bins, density.mean(), density.std())
    # l = plt.plot(bins, y, 'r--' , linewidth=1)
    # plt.show()
    #===========================================================================
    oos_sim = np.random.choice(density, size=1000, replace=False)
    #===========================================================================
    # n, bins, patches = plt.hist(oos_sim, 100, normed=1, facecolor='green', alpha=0.75)
    # y = mlab.normpdf( bins, oos_sim.mean(), oos_sim.std())
    # l = plt.plot(bins, y, 'r--' , linewidth=1)
    # plt.show()
    #===========================================================================
    xi_n = np.array([])
    for n in [5]:
        print('Solving n=',n)
        extra_poits = n- len(xi_n)

        #np.array([10,18,50,56])
        xi_n = np.append(xi_n , np.random.choice(density, size=extra_poits, replace=False))
        heightsE,binsE = np.histogram(xi_n,bins=np.arange(xi_n.min(),xi_n.max()+2))
        heightsE = heightsE/sum(heightsE)
        
        xi_n.sort()
        #ev_policy = xi_n[int(quantile*n) - 1] 
        #oos_dem_bar = news_vendor_fun(ev_policy,oos_sim).mean()
        #test_out_of_sample(ev_policy, oos_sim, news_vendor_fun, None)
        N = [i for i in range(n)]
        d = [-xi_n.min()*0.5, xi_n.max()*1.2]
        mean_n = xi_n.mean()
        std_n = xi_n.std()
        #d = [-np.maximum(0,mean_n-3*std_n), mean_n+3*std_n]
        print(d)
        K = [[0] for _ in N]
        a = {(0,i):0 for i in N}
        b = {(0,i):0 for i in N}
        
        sim_results = []
        #for dro_radius in [100]:
        for dro_radius in [1]:
            K = [[0] for _ in N]
            a = {(0,i):0 for i in N}
            b = {(0,i):0 for i in N}
            instance_name = 'NewVendor_N%i_CW' %(n)
            instance = {'risk_measure_params':{}}
            
            instance['risk_measure_params']['radius'] = dro_radius
            instance['risk_measure']= DistRobustWassersteinCont
            
            supp, pmf, x_rob = nv_cutting(N,xi_n,K,a,b,L,C,d,dro_radius, news_vendor_fun)
            
           
            data_worst_case = np.random.choice(supp, size=1000000, p=pmf)
            #heights,bins = np.histogram(data_worst_case,bins=int(len(set(xi_n))+1))
            #===================================================================
            # wc_bins = np.array(list(set(data_worst_case).union(set(xi_n))))
            # wc_bins.sort()
            # heights,bins = np.histogram(data_worst_case,bins=np.arange(wc_bins.min(),wc_bins.max()+2))
            # heights = heights/sum(heights)
            #===================================================================
            bin_width = (max(binsE) - min(binsE))/len(binsE)
            plt.bar(binsE[:-1],heightsE,width=bin_width, color="blue", alpha=1)
            #plt.bar(bins[:-1],heights,width=(max(bins) - min(bins))/len(bins), color="red", alpha=0.7)
            plt.bar(supp,pmf,width=bin_width, color="red", alpha=0.7)
            plt.show()
        #=======================================================================
        #     plt.bar(supp,pmf, color="red", alpha=0.5)
        #     plt.show()
        #     sim_res  = test_out_of_sample(x_star,oos_sim,news_vendor_fun,instance)
        #     sim_results.append(sim_res)
        # 
        #=======================================================================
        #=======================================================================
        # plot_sim_results(sim_results, newsvendro_path+'/Output/%s.pdf' %(instance_name), n, excel_file = False)
        #=======================================================================
