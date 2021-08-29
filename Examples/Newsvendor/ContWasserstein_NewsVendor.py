"""
Created on Fri Jul 13 16:13:29 2018

Implementation of DRO reformulation for the news vendor problem.
The reformulation is the one in Mohajerin and kuhn 2018 paper
in Math. Programming.
@author: dduque
"""

from gurobipy import Model, GRB, quicksum, LinExpr

import os
import sys
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

# Import SDDPpy function for output
sys.path.append("/Users/dduque/Dropbox/WORKSPACE/SDDP/")
from SDDP.RiskMeasures import DistRobustWassersteinCont
from SDDP.SDDP_utils import report_stats
from OutputAnalysis.SimulationAnalysis import SimResult, plot_sim_results
news_vendor_path = os.path.dirname(os.path.realpath(__file__))


def solve_cont_wasserstain(N, xi_n, K, r, c, b, L, C, d, dro_r, norm='inf'):
    """
        Defines a DRO version of the news vendor problem where
        the cost function is piece-wise convex in x (first stage)
        and the demand (i.e., max of affine funcions as in the paper).
        The cost function for a given value of x and demand realization
        xi is given by
        f(x,xi) = max{(c-r[0])x, cx - r[0]xi} #Ignorin other pieces
        Args:
            N (list): support indices
            xi_n (list): empirical support
            K (list): pieces indices
            r (list): coeff of xi on every piece (neg. of the sell cost)
            c (float): coeff of x on every piece (buy cost)
            b (list): intercepts of the pieces
            L (list): support constraint indices
            C (list): lhs coefficients of the constraints defining the
                support of the one-dimensional random variable
            d (list): rhs of the constraints for the support
            dro_r (float): radius of the dro model
            norm (str or int): Dual norm to be used. If inf, then
                the Wassersteain metric is defined under the L1-norm.
                If 2, then the metric is defined under the L2-norm.
        Returns:
            x_star (double): optimal solution to the NV problem
            new_support (ndarray): support of the worst-case distribution
            new_pmf (ndarray): pmf of the worst-case distribution
    """
    model = Model('ContWassersteinNewsVendor')
    model.params.OutputFlag = 0
    model.params.NumericFocus = 3
    model.params.QCPDual = 1
    x = model.addVar(lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='x')
    s = model.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=[1 / len(N) for _ in N], vtype=GRB.CONTINUOUS, name='s')
    lam = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=dro_r, vtype=GRB.CONTINUOUS, name='lambda_var')
    gam = model.addVars(K, N, L, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='gamma_var')
    model.update()
    piece_ctrs = {}
    for i in N:
        for k in K:
            piece_ctrs[(k, i)] = model.addConstr(rhs=((c[k] * x + r[k] * xi_n[i] + b[k]) + quicksum(
                (d[l] - C[l] * xi_n[i]) * gam[k, i, l] for l in L)),
                                                 sense=GRB.GREATER_EQUAL,
                                                 lhs=s[i],
                                                 name=f'piece[{k},{i}]')
    
    norm_ctrs = {}
    norm_aux_ctrs = {}
    for k in K:
        for i in N:
            aux_var_k_i = model.addVar(lb=-GRB.INFINITY,
                                       ub=GRB.INFINITY,
                                       obj=0,
                                       vtype=GRB.CONTINUOUS,
                                       name=f'norm_aux[{k},{i}')
            # Extra constraint modeling the linear expresion inside the norm
            # dual variable of this constraint is q_ik
            norm_aux_ctrs[(k, i)] = model.addConstr(lhs=aux_var_k_i,
                                                    sense=GRB.EQUAL,
                                                    rhs=quicksum(C[l] * gam[k, i, l] for l in L) - r[k],
                                                    name=f'lin_norm_dual_q[{k},{i}]')
            
            if norm == 'inf':
                norm_ctrs[(k, i, -1)] = model.addConstr(lhs=lam - aux_var_k_i,
                                                        sense=GRB.GREATER_EQUAL,
                                                        rhs=0,
                                                        name='L_inf_norm_n[%i,%i]' % (k, i))
                norm_ctrs[(k, i, 1)] = model.addConstr(lhs=lam + aux_var_k_i,
                                                       sense=GRB.GREATER_EQUAL,
                                                       rhs=0,
                                                       name='L_inf_norm_p[%i,%i]' % (k, i))
            elif norm == 2:
                norm_ctrs[(k, i, 1)] = model.addConstr(lhs=lam - aux_var_k_i * aux_var_k_i,
                                                       sense=GRB.GREATER_EQUAL,
                                                       rhs=0,
                                                       name='L2Norm[%i,%i]' % (k, i))
            else:
                raise f'L-{norm} is not defined as a valid dual norm.'
    
    model.update()
    model.write('news_vendor_dro.lp')
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        #print(model.ObjVal, x.X, lam.X, [s[i].X for i in N])
        print(f"f* = {model.ObjVal}")
        # for c in model.getConstrs():
        #     if c.Pi > 1E-9:
        #         #pass
        #         print(c.ConstrName, c.Pi)
        # try:
        #     for c in model.getQConstrs():
        #         print(c.QCName, c.QCPi)
        # except:
        #     pass
        # for v in model.getVars():
        #     if v.X > 1E-6:
        #         print(v.VarName, v.X)
        #print('orig supp', xi_n)
        new_support = []
        pmf = []
        for (k, i) in piece_ctrs:
            if piece_ctrs[(k, i)].Pi > 1E-7:
                new_atom = xi_n[i]
                new_atom = xi_n[i] - (norm_aux_ctrs[(k, i)].Pi) / piece_ctrs[(k, i)].Pi
                new_support.append(new_atom)
                pmf.append(piece_ctrs[(k, i)].Pi)
        new_support = np.array(new_support)
        pmf = np.array(pmf)
        supp_argsort = np.argsort(new_support)
        pmf = pmf[supp_argsort]
        pmf = pmf / pmf.sum()  # In the QP case, we might have numerical error
        new_support.sort()
        
        for i in range(len(new_support) - 1, 0, -1):
            if np.abs(new_support[i] - new_support[i - 1]) < 1E-8:
                new_support = np.delete(new_support, obj=i)
                rep_prob = pmf[i]
                pmf = np.delete(pmf, obj=i)
                pmf[i - 1] += rep_prob
        # print('new supp ', new_support)
        # print('new pmf', pmf)
        return x.X, new_support, pmf
    else:
        model.computeIIS()
        model.write("NewsVendorInf.ilp")
        os.system("open -a TextEdit filename Users/dduque/Desktop/NewsVendorInf.ilp")


def create_obj(K, c, r, b):
    '''
        Creates a function as the maximum of the pieces
        defined by c, r, and b:
            max_k {c_k x + r_k xi + b_k}
    '''
    
    def news_vendor_obj(x, xi_n):
        return np.max([c[k] * x + r[k] * xi_n + b[k] for k in K], axis=0)
    
    return news_vendor_obj


def test_out_of_sample(x_star, test_set, obj_fun, instance):
    objs = obj_fun(x_star, test_set)
    sim_res = SimResult(instance, objs)
    x_bar = np.mean(objs)
    std = np.std(objs)
    # if instance == None:
    #     print('%10s %10.4f %10.4f %10.4f' % ('EV_Policy', x_star, x_bar, std))
    # else:
    #     print('%10.4f %10.4f %10.4f %10.4f' % (instance['risk_measure_params']['radius'], x_star, x_bar, std))
    return sim_res


if __name__ == '__main__':
    np.random.seed(0)
    
    K = [0, 1, 2]  # [0, 1, 2, 3]
    r = [0, -3, -5, -9]  # selling prices to induces multiple pieces
    c = [1 + r[1], 1, 1, 1]
    b = [0, 0, 20, 40]
    
    news_vendor_fun = create_obj(K, c, r, b)
    # density = np.append(np.append(np.random.lognormal(0, 1, size=1000), np.random.normal(10, 1, size=2000)),
    #                     np.random.normal(15, 2, size=1000))
    # neg_supp = np.random.lognormal(2, 1, size=100000)
    # density = neg_supp[neg_supp > 0]
    density = np.abs(np.random.binomial(200, 0.1, size=20000))
    density_bins = np.arange(0, 40)
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
    for n in [20]:
        print('Solving n=', n)
        extra_points = n - len(xi_n)
        
        xi_n = np.append(xi_n, np.random.choice(density, size=extra_points, replace=False))
        heights_n, bins_n = np.histogram(xi_n, bins=density_bins)
        heights_n = heights_n / sum(heights_n)
        xi_n.sort()
        
        #ev_policy = xi_n[int(quantile*n) - 1]
        #oos_dem_bar = news_vendor_fun(ev_policy,oos_sim).mean()
        #test_out_of_sample(ev_policy, oos_sim, news_vendor_fun, None)
        N = [i for i in range(n)]
        L = [0, 1]
        C = [-1, 1]
        d = [-xi_n.min() * 0.0, xi_n.max() * 1.5]
        
        sim_results = []
        sp_sim = None  # Non-DRO simulations
        #for dro_radius in [100]:
        
        # for dro_radius in ([0, 0.5] + list(np.arange(1, 10, 0.1)) +
        #                    [10, 11, 12, 13, 14, 15, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250]):
        for dro_radius in [2]:
            print(f'DRO radius: {dro_radius}')
            plt.bar(bins_n[:-1], heights_n, width=(max(bins_n) - min(bins_n)) / len(bins_n), color="blue", alpha=0.8)
            instance_name = 'NewVendor_N%i_CW' % (n)
            instance = {'risk_measure_params': {}}
            
            instance['risk_measure_params']['radius'] = dro_radius
            instance['risk_measure'] = DistRobustWassersteinCont
            x_star, supp, pmf = solve_cont_wasserstain(N, xi_n, K, r, c, b, L, C, d, dro_radius, norm=2)
            print(supp, pmf)
            data_worst_case = np.random.choice(supp, size=1000000, p=pmf)
            heights, bins = np.histogram(data_worst_case, bins=len(bins_n))
            heights = heights / sum(heights)
            plt.bar(bins[:-1], heights, width=(max(bins_n) - min(bins_n)) / len(bins_n), color="red", alpha=0.5)
            #plt.bar(supp,pmf, color="red", alpha=0.5)
            
            sim_res = test_out_of_sample(x_star, oos_sim, news_vendor_fun, instance)
            sim_results.append(sim_res)
            print('x* = ', x_star)
            report_stats(sim_res.sims_ub)
            if dro_radius == 0:
                sp_sim = sim_res
            #plt.show()
        # plot_sim_results(sp_sim,
        #                  sim_results,
        #                  news_vendor_path + '/Output/%s.pdf' % (instance_name),
        #                  n,
        #                  excel_file=False)
