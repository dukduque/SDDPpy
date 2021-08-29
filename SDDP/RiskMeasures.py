'''
Created on Jan 11, 2018

Modified on May 22, 2019:
    Deep change in the cut gradient and intercept computation.

@author: dduque
'''

from abc import ABC, abstractmethod
from gurobipy import *
import SDDP as cs
import numpy as np
from SDDP import ZERO_TOL, FEASIBILITY_TOL
from SDDP import logger as sddp_log


class AbstractRiskMeasure(ABC):
    '''
    Abstract representation of what a risk measure should do in SDDP
    '''
    
    @abstractmethod
    def __init__(self):
        '''
        
        Attr:
            _current_cut_gradient (dict): local copy of the current
                cut gradient (used again when computing the.
        '''
        self.next_stage_rnd_vec = None
        self._current_cut_gradient = None
    
    @abstractmethod
    def compute_cut_gradient(self, sp, sp_next, srv, soo, spfs, cut_id, p=None):
        '''
        Computes cut(s) gradient(s) for both multi cut and single cut
        implementations of the algorithm.
        
        Args:
            sp (StageProblem): current subproblem where the cut will be added.
            sp_nex (StageProblem): subproblem for the next stage.
            srv (StageRandomVector): Random vector of the next stage
            soo (List of dict): A list of outputs of all the subproblems descendants (subs outputs outcomes).
            spfs (dict (str-float)): Values for the states of the current stage computed
                in the forward pass.
            cut_id (int): Numeric id of the cut being created (iteration in SDDP)
            p (ndarray): Optional, vector of probabilities.
        Returns:
            pi_bar (list[dict]): Expected value of the duals. For the single cut algorithm
                the list contains just one element.
            cut_gradient_coeff(list[dict]):
        '''
        multicut = sp.multicut
        cut_gradient_coeff = None
        if multicut == False:  #single cut
            probs = srv.p if type(p) == type(None) else p
            cut_gradient_coeff = [{}]
            for v_out in sp.out_state:
                cut_gradient_coeff[0][v_out] = sum(probs[i] * soo[i]['RC'][v_out] for (i, o) in enumerate(srv.outcomes))
        else:  #Multicut
            cut_gradient_coeff = [{} for _ in range(len(soo))]
            for (i, outcome) in enumerate(soo):
                cut_gradient_coeff[i] = outcome['RC']
        
        self._current_cut_gradient = cut_gradient_coeff
        return cut_gradient_coeff
    
    @abstractmethod
    def compute_cut_intercept(self, sp, sp_next, srv, soo, spfs, cut_id, p=None):
        '''
        Computes cut intercept(s).
        
        Args: (as in cut gradient method)
        Returns:
            cut_intercepts (list of float): List of intercep(s) of the cut(s).
        '''
        cut_gradient_coeff = self._current_cut_gradient
        cut_intercepts = None
        if not sp.multicut:
            probs = srv.p if p is None else p
            cut_intercepts = [
                np.sum(probs[i] * soo[i]['objval']
                       for (i, o) in enumerate(srv.outcomes)) - np.sum(spfs[vn] * cut_gradient_coeff[0][vn]
                                                                       for vn in sp.out_state)
            ]
        else:
            cut_intercepts = [0 for _ in srv.outcomes]
            for (i, o) in enumerate(srv.outcomes):
                cut_intercepts[i] = soo[i]['objval'] - sum(spfs[vn] * cut_gradient_coeff[i][vn] for vn in sp.out_state)
        return cut_intercepts
    
    @abstractmethod
    def update_cut_intercept(self):
        raise ('Method not implemented in abstract class')
    
    @abstractmethod
    def modify_stage_problem(self, *args):
        raise ('Method not implemented in abstract class')
    
    @abstractmethod
    def forward_pass_updates(self, *args, **kwargs):
        'Default is False for sub resolve and 0 for constraint violations'
        return False, 0
    
    @abstractmethod
    def forward_prob_update(self, *args, **kwags):
        '''
        Only needed for dynamic sampling
        '''
        return None


class Expectation(AbstractRiskMeasure):
    '''
    Implements an expected value risk measure
    '''
    
    def __init__(self):
        super().__init__()
    
    def compute_cut_gradient(self, sp, sp_next, srv, soo, spfs, cut_id):
        '''
            See abstract class method
        '''
        return super().compute_cut_gradient(sp, sp_next, srv, soo, spfs, cut_id)
    
    def compute_cut_intercept(self, sp, sp_next, srv, soo, spfs, cut_id):
        '''
            See abstract class method
        '''
        return super().compute_cut_intercept(sp, sp_next, srv, soo, spfs, cut_id)
    
    def update_cut_intercept(self):
        pass
    
    def modify_stage_problem(self, sp, model, n_outcomes):
        model.setObjective(sp.cx + sp.oracle.sum() / len(sp.oracle))
    
    def forward_pass_updates(self, *args, **kwargs):
        'Default is False for sub resolve and 0 for constraint violations'
        return False, 0
    
    def forward_prob_update(self, *args):
        pass


'''
Distance function for Wasserstein metric
'''


def norm_fun(xi_o, xi_d, n):
    '''
    Computes the norm of the difference of the vectors given as parameters
    '''
    return np.linalg.norm(xi_o - xi_d, n)


def mod_chi2(xi_o, xi_d, n):
    if (xi_o == xi_d).all():
        return 0
    else:
        return 1


class DistRobustWasserstein(AbstractRiskMeasure):
    '''
    Distributional uncertainty set defined by the Wasserstein metric
    Attributes:
        norm (int): norm degree to compute the distance between two random vectors
        radius (float): length of the uncertainty set based on Wasserstein distance
        primal_dus (int): If different from 'ALL', specifies the number of combinations
            to be considered for every outcome in the origin side of the transportation problem.
        data_random_container (RandomContainer): contains the randomness coming form data. It might be
            the same as the random container used in the algorithm when the number of data points equal
            the number of outcomes being considered. 
        dro_ctrs (dic of GRBCtr): dictionary storing the constraints whose dual variables are the transportation plan.  
    '''
    
    def __init__(self, norm=1, radius=1, primal_dus='ALL', dist_func=norm_fun, data_random_container=None):
        super().__init__()
        self.norm = norm
        self.radius = radius
        self.primal_dus = primal_dus
        self.dist_func = dist_func
        if self.primal_dus != 'ALL':
            assert isinstance(self.primal_dus, int) , 'Primal_DUS parameters should specify an integer'+ \
            'value to determine the number of constraints to be added in the primal representation of the DUS'
        self.data_random_container = data_random_container
        self.dro_ctrs = {}
        self.gamma = None
    
    def compute_cut_gradient(self, sp, sp_next, srv, soo, spfs, cut_id):
        '''
            See abstrac method. This risk measure does not support the
            single cut implmentation.
        '''
        if sp.multicut == False:  #single cut
            raise 'Risk measure does not support single cut'
        else:  #Multicut
            return super().compute_cut_gradient(sp, sp_next, srv, soo, spfs, cut_id)
    
    def compute_cut_intercept(self, sp, sp_next, srv, soo, spfs, cut_id):
        '''
            See abstrac method. This risk measure does not support the
            single cut implmentation.
        '''
        if sp.multicut == False:  #single cut
            raise 'Risk measure does not support single cut'
        else:  #Multicut
            return super().compute_cut_intercept(sp, sp_next, srv, soo, spfs, cut_id)
    
    def update_cut_intercept(self):
        pass
    
    def modify_stage_problem(self, sp, model, next_stage_rnd_vector):
        '''
        Modify the stage problem model to incorporate the DRO
        risk measure as dual variables of the inner problem.
        
        Args:
            sp (StageProblem): Stage problem to modify.
            model (GRBModel): Model object associate to the stage.
            next_stage_rnd_vector(StageRandomVector): random vector for the following stage.
            
        '''
        assert sp.multicut, 'This risk measure implementation is only compatible with multicut setting.'
        t = sp.stage
        if sp._last_stage:
            return
        
        nsrv_org = next_stage_rnd_vector  #Origin points in the support for the transport problem
        if self.data_random_container != None:
            nsrv_org = self.data_random_container[t + 1]
        
        nsrv_des = next_stage_rnd_vector  #Destination points in the support for the transport problem
        
        n_outcomes_org = nsrv_org.outcomes_dim
        n_outcomes_des = nsrv_des.outcomes_dim
        
        #print(n_outcomes_org , '   ---   ' ,n_outcomes_des)
        #lambda_var =  model.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='lambda[%i]' %(t))
        gamma_var = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='gamma[%i]' % (t))
        nu_var = model.addVars(n_outcomes_org,
                               lb=-GRB.INFINITY,
                               ub=GRB.INFINITY,
                               vtype=GRB.CONTINUOUS,
                               name='nu[%i]' % (t))
        model.update()
        #Update objective function
        new_objective = sp.cx + self.radius * gamma_var + quicksum(nsrv_org.p[i] * nu_var[i]
                                                                   for i in range(n_outcomes_org))
        model.setObjective(new_objective, GRB.MINIMIZE)
        
        #Add extra constraints associated to the primal representation of the uncertainty set
        for i in range(n_outcomes_org):
            #WARNING: This implementation assumes that the distance between outcomes is a valid metric
            xi_i = nsrv_org.get_sorted_outcome(i)
            for j in range(n_outcomes_des):
                xi_j = nsrv_des.get_sorted_outcome(j)
                d_ij = self.dist_func(xi_i, xi_j, self.norm)
                crt = model.addConstr((d_ij * gamma_var + nu_var[i] - sp.oracle[j] >= 0), 'dro_dual_ctr[%i%i]' % (i, j))
                self.dro_ctrs[(i, j)] = crt
            model.update()
        
        self.gamma = gamma_var
    
    def forward_pass_updates(self, *args, **kwargs):
        return False, 0
    
    def modify_param(self, **kwags):
        for kw_name in kwags:
            if kw_name in self.__dict__:
                self.__dict__[kw_name] = kwags[kw_name]
                if kw_name == 'radius' and self.gamma != None:
                    self.gamma.obj = kwags[kw_name]
            
            else:
                raise 'Parameter %s is not defined in risk measure %s' % (kw_name, type(self))
    
    def forward_prob_update(self, t, sp, next_sp, fp_out_states, sample_path, rnd_container):
        '''Updates the probability distribution for the descendants of the current stage.
        The update sets the probability use to sample new outcomes to the worst case prob
        distribution for the particular sample path being explored.
        Args:
            t (int): current stage (just solved in forward pass)
            rnd_container (RandomContainer): object that contains all the randomness in the problem.
            
        This method modifies the probabilities of the random vector for stage t+1.
        '''
        if t == len(rnd_container.stage_vectors) - 1:
            #Last stage needs no update
            return
        desc_rnd_vec = rnd_container[t + 1]  #Descendant outcomes
        p_w = np.zeros(desc_rnd_vec.outcomes_dim)
        for (i, j) in self.dro_ctrs:
            p_w[j] += self.dro_ctrs[(i, j)].Pi
        p_w = np.around(np.abs(p_w), 8)
        p_w = p_w / p_w.sum()
        #Mix with nominal distribution
        ds_beta = cs.options['dynamic_sampling_beta']
        p_w = ds_beta * p_w + (1 - ds_beta) * desc_rnd_vec.p_copy
        desc_rnd_vec.modifyOutcomesProbabilities(p_w)
    
    def define_scenario_tree_uncertainty_set(self, stage, outcome, model, srv, phi, branch_name):
        '''
        Modifies the model given as a parameter to incorporate the uncertainty set
        for a given node in a scenario tree.
        Args:
            stage (int): stage of the associate scenario tree at which a constraint on 
                its descendants is being added. 
            model (GRBModel): gubori model for the upper bound. 
            srv (StageRandomVector): random elements of the descendants.
            phi (list of GRBVariables): decision variables representing the worst-case distribution
                for the branch of the ScenarioTreeNode that invoked this method.
        '''
        nsrv_org = srv  #Origin points in the support for the transport problem
        if self.data_random_container != None:
            nsrv_org = self.data_random_container[stage + 1]
        
        nsrv_des = srv  #Destination points in the support for the transport problem
        n_org = nsrv_org.outcomes_dim
        n_des = nsrv_des.outcomes_dim
        
        z = model.addVars(n_org, n_des, lb=0, up=1, obj=0, vtype=GRB.CONTINUOUS, name='z_%i%s' % (stage, branch_name))
        model.update()
        model.addConstrs((z.sum(o, '*') == nsrv_org.p_copy[o] * quicksum(phi) for o in range(n_org)),
                         name='org_%i%s' % (stage, branch_name))
        model.addConstrs((z.sum('*', d) == phi[d] for d in range(n_des)), name='des_%i%s' % (stage, branch_name))
        
        DUS_exp = LinExpr()
        for i in range(n_org):
            #WARNING: This implementation assumes that the distance between outcomes is a valid metric
            xi_i = nsrv_org.get_sorted_outcome(i)
            for j in range(n_des):
                xi_j = nsrv_des.get_sorted_outcome(j)
                d_ij = self.dist_func(xi_i, xi_j, self.norm)
                DUS_exp.addTerms(d_ij, z[i, j])
        
        model.addConstr(lhs=DUS_exp,
                        sense=GRB.LESS_EQUAL,
                        rhs=self.radius * quicksum(phi),
                        name='dus_%i%s' % (stage, branch_name))
        model.update()
    
    def orthogonal_proj_uncertainty_set(self, p_k, d_k, srv, stage):
        '''
        Performs an orthogonal projection onto the uncertainty set.
        TODO: rewrite after deciding what this is actually going to do. 
        Args:
            p_trial(ndarray): trial point 
        '''
        model = Model('DUS_hit_and_run')
        model.params.OutputFlag = 0
        nsrv_org = srv  #Origin points in the support for the transport problem
        if self.data_random_container != None:
            nsrv_org = self.data_random_container[stage + 1]
        
        nsrv_des = srv  #Destination points in the support for the transport problem
        n_org = nsrv_org.outcomes_dim
        n_des = nsrv_des.outcomes_dim
        
        z = model.addVars(n_org, n_des, lb=0, up=1, obj=0, vtype=GRB.CONTINUOUS, name='z')
        d_range = model.addVar(lb=0, obj=1, vtype=GRB.CONTINUOUS, name='d_range')
        model.update()
        model.addConstrs((z.sum(o, '*') == nsrv_org.p_copy[o] for o in range(n_org)), name='org')
        model.addConstrs((z.sum('*', d) == p_k[d] + d_range * d_k[d] for d in range(n_des)), name='des')
        
        DUS_exp = LinExpr()
        for i in range(n_org):
            xi_i = nsrv_org.get_sorted_outcome(i)
            for j in range(n_des):
                xi_j = nsrv_des.get_sorted_outcome(j)
                d_ij = self.dist_func(xi_i, xi_j, self.norm)
                DUS_exp.addTerms(d_ij, z[i, j])
        
        model.addConstr(lhs=DUS_exp, sense=GRB.LESS_EQUAL, rhs=self.radius, name='dus')
        model.update()
        
        model.ModelSense = GRB.MAXIMIZE
        model.update()
        model.optimize()
        if model.status == GRB.OPTIMAL:
            return d_range.X
        else:
            raise 'Trial point or direction are invalid, opt problem not optimal'
    
    def get_dus_params(self, srv, stage):
        nsrv_org = srv  #Origin points in the support for the transport problem
        if self.data_random_container != None:
            nsrv_org = self.data_random_container[stage + 1]
        
        nsrv_des = srv  #Destination points in the support for the transport problem
        n_org = nsrv_org.outcomes_dim
        n_des = nsrv_des.outcomes_dim
        
        d_ij = np.zeros((n_org, n_des))
        for i in range(n_org):
            xi_i = nsrv_org.get_sorted_outcome(i)
            for j in range(n_des):
                xi_j = nsrv_des.get_sorted_outcome(j)
                d_ij[i, j] = self.dist_func(xi_i, xi_j, self.norm)
        
        return n_org, n_des, d_ij, self.radius


class DistRobustWassersteinCont(AbstractRiskMeasure):
    '''
    Distributional uncertainty set defined by the Wasserstein metric with continuous support.
    
    This class implements the single-level reformulation in:
    Data-driven distributionally robust optimization using the Wasserstein metric:
    performance guarantees and tractable reformulations. 
    Mohajerin Estahani and Kuhn, 2017
    
    Attributes:
        norm (int): norm degree to compute the distance between two random vectors
        radius (float): length of the uncertainty set based on Wasserstein distance
        support_ctrs(list of dict): Constraints that define the support of the random vector. 
            Each element of the list corresponds to a constraint represented with a dictionary
            that has as key the name of the random element and as value the corresponding ctr coefficient.
            The constraion is assume to be C xi <= d. 
             
        support_rhs (list of float): RHS of the constraints defining the uncertainty support in R^m (m is the dimension of the random vector)
        dro_ctrs (dic of GRBCtr): dictionary storing the constraints whose dual variables are the transportation plan.  
    '''
    
    def __init__(self, norm=1, radius=1, support_ctrs=None, support_rhs=None):
        super().__init__()
        self.norm = norm
        self.radius = radius
        
        #verify support constraints consistency
        assert len(support_ctrs) == len(support_rhs), 'Number of ctrs needs to be equal to the number of RHSs.'
        self.support_ctrs = support_ctrs
        self.support_rhs = support_rhs
        self.dro_ctrs = {}
        
        #===========================================
        #Additional notation as in Mohajerin and Kuhn, 2018 (model 18b)
        self.lambda_var = None
        self._aux_norm_ctrs = {}  # Stores constraints modeling the norm
        self._support_slack = None
        self._support_slack_computed = False
        self._pi = None
    
    def compute_cut_gradient(self, sp, sp_next, srv, soo, spfs, cut_id):
        '''
            See abstract method documentation for cut gradients.
            
            Besides the cut gradient and intercept computation, one needs
            to create addition variables to model the dual of the worst-case
            expectation problem. 
        '''
        multicut = sp.multicut
        pi = None
        cut_gradient_coeff = None
        
        if multicut == False:  #single cut
            raise 'Risk measure does not support single cut'
        else:  #Multicut
            #Gen and store regular cut gradients (w.r.t x)
            cut_gradient_coeff = super().compute_cut_gradient(sp, sp_next, srv, soo, spfs, cut_id)
            
            pi = [{} for _ in srv.outcomes]
            for ctr in sp_next.ctrsForDuals:
                for (i, o) in enumerate(srv.outcomes):
                    pi[i][ctr] = soo[i]['duals'][ctr]
            
            self._pi = pi
            
            #Create new variables for the reformulation \lambda \in R^l (l number of constraints that define the support)
            vec_dim = len(srv.elements)
            outcomes_dim = len(srv.outcomes)
            supp_ctr_dim = len(self.support_ctrs)
            m = sp.model
            # gamma_k: dual variables used in equation 18b of Mohajerin and Kuhn, 2018.
            gamma_k = m.addVars(outcomes_dim,
                                supp_ctr_dim,
                                lb=0,
                                ub=GRB.INFINITY,
                                obj=0,
                                vtype=GRB.CONTINUOUS,
                                name='gamma[%i]' % (cut_id))
            m.update()
            
            if self._support_slack_computed == False:
                #Computes the coefficients for gamma in the cut (only computed once)
                self._support_slack = np.array([[
                    self.support_rhs[sc_ix] - sum(o[ele] * sup_ctr[ele] for ele in sup_ctr)
                    for (sc_ix, sup_ctr) in enumerate(self.support_ctrs)
                ] for o in srv.outcomes])
                self._support_slack = np.around(self._support_slack, 10)
                self._support_slack_computed = True
                assert (self._support_slack >=
                        0).all(), 'Uncertainty set is infeasible for the training data set. ' + self._support_slack[
                            self._support_slack < 0]
            
            #Adding coefficients of NEW gamma variables to the cut
            #Adding extra constraints that relates gradients of \xi (\pi) with gamma
            for (i, o) in enumerate(srv.outcomes):
                for supp_ctr_ix in range(supp_ctr_dim):
                    v_name = gamma_k[i, supp_ctr_ix].VarName
                    if v_name in cut_gradient_coeff[i]:
                        cut_gradient_coeff[i][v_name] += self._support_slack[i][supp_ctr_ix]
                    else:
                        cut_gradient_coeff[i][v_name] = self._support_slack[i][supp_ctr_ix]
                pi_i_k = pi[i]
                aux_norm_var = m.addVars(srv.elements.keys(),
                                         lb=-GRB.INFINITY,
                                         ub=GRB.INFINITY,
                                         obj=0,
                                         vtype=GRB.CONTINUOUS,
                                         name=f'aux_norm[{cut_id},{i}]')
                for rhs_ctr_name in sp_next.ctrRHSvName:
                    rnd_ele_name = sp_next.ctrRHSvName[rhs_ctr_name]
                    self._aux_norm_ctrs[(cut_id, i, rnd_ele_name)] = m.addConstr(
                        lhs=aux_norm_var[rnd_ele_name],
                        sense=GRB.EQUAL,
                        rhs=quicksum(gamma_k[i, supp_ctr_ix] * (self.support_ctrs[supp_ctr_ix].get(rnd_ele_name, 0))
                                     for supp_ctr_ix in range(supp_ctr_dim)) - pi_i_k[rhs_ctr_name],
                        name=f'aux_norm_{cut_id}_{i}_{rnd_ele_name}')
                # ctr: \lambda >= || C^T gamma - dual ||_* from DRO reformulation
                if self.norm == 1:  # Then dual norm is the L-inf norm
                    for (rnd_ele, var) in aux_norm_var.items():
                        m.addConstr(lhs=self.lambda_var,
                                    sense=GRB.GREATER_EQUAL,
                                    rhs=var,
                                    name='normCtr_%i_%i_%s_neg' % (cut_id, i, rnd_ele))
                        m.addConstr(lhs=self.lambda_var,
                                    sense=GRB.GREATER_EQUAL,
                                    rhs=-var,
                                    name='normCtr_%i_%i_%s_pos' % (cut_id, i, rnd_ele))
                
                elif self.norm == 2:  # Then dual norm is the L2 norm
                    q_exp = self.lambda_var * self.lambda_var - quicksum(v * v for f in aux_norm_var.values())
                    m.addConstr(lhs=q_exp,
                                sense=GRB.GREATER_EQUAL,
                                rhs=0,
                                name='normCtr_%i_%i_%s_pos' % (cut_id, i, rnd_ele_name))
                else:
                    raise f'L{self.norm} is nor supported.'
            
            m.update()
        self._current_cut_gradient = cut_gradient_coeff
        return cut_gradient_coeff
    
    def compute_cut_intercept(self, sp, sp_next, srv, soo, spfs, cut_id):
        '''
            See abstrac method. This risk measure does not support the
            single cut implmentation.
        '''
        if sp.multicut == False:  #single cut
            raise 'Risk measure does not support single cut'
        else:  #Multicut
            return super().compute_cut_intercept(sp, sp_next, srv, soo, spfs, cut_id)
    
    def update_cut_intercept(self):
        raise 'update_cut_intercept is not available for this risk measure'
    
    def modify_stage_problem(self, sp, model, next_stage_rnd_vector):
        '''
        Modify the stage problem model to incorporate the DRO
        risk measure as dual variables of the inner problem.
        
        Args:
            sp (StageProblem): Stage problem to modify.
            model (GRBModel): Model object associate to the stage.
            next_stage_rnd_vector(StageRandomVector): random vector for the following stage.
            
        '''
        assert sp.multicut, 'This risk measure implementation is only compatible with multicut setting.'
        t = sp.stage
        if sp._last_stage:
            return
        
        nsrv_org = next_stage_rnd_vector  #Origin points in the support for the transport problem
        #nsrv_des #Destination is a continuum
        
        n_outcomes_org = nsrv_org.outcomes_dim
        
        #Using the same notation as in Kuhn paper
        #lambda_var =  model.addVar(lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='lambda[%i]' %(t))
        lambda_var = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='lambda[%i]' % (t))
        self.lambda_var = lambda_var
        #nu_var =  model.addVars(n_outcomes_org,  lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='nu[%i]' %(t))
        # Oracle variables are used in placed of nu_var
        
        model.update()
        #Update objective function
        new_objective = sp.cx + self.radius * lambda_var + quicksum(nsrv_org.p[i] * sp.oracle[i]
                                                                    for i in range(n_outcomes_org))
        model.setObjective(new_objective, GRB.MINIMIZE)
        
        #Extra constraints are added on demand as cuts get generated
    
    def forward_pass_updates(self, *args, **kwargs):
        return False, 0
    
    def forward_prob_update(self, t, sp, next_sp, fp_out_states, sample_path, rnd_container):
        '''
            Updates the probability distribution for the descendants of the current stage.
            The update sets the probability use to sample new outcomes to the worst case prob
            distribution for the particular sample path being explored.
            Args:
                t (int): current stage (just solved in forward pass)
                rnd_container (RandomContainer): object that contains all the randomness in the problem.
                
            This method modifies the probabilities of the random vector for stage t+1.
        '''
        pass
        #TODO: Implement dynamic sampling considering that the support changes
    
    def forward_prob_update_WassCont(self, stage, sp, rnd_cont):
        '''
            Computes the worst-case probability distribution given the
            solution to a particular stage problem. The worst-case 
            distribution corresponds to the next stage. The support is
            computed using the dual variables of the stage problem that
            correspond to constrains modeling the norm in Mohajerin and
            Kuhn (2018) paper.
            Args:
                stage (int): stage index
                sp (StageProblem): stage problem solved in the forward pass
                rnd_cont (RandomContainer): random container of the problem
            Return:
                new_support (list of ndarray): new support
                new_pmf (ndarray): pmf of the new support
            Note:
                The worst-case distribution is stored as an attribute of the
                corresponding random vector in the random container.
        '''
        tup_ind, duals_vars = sp.cut_pool.get_non_zero_duals()
        new_support = []
        new_pmf = []
        support_map = []  # Indices of the outcome that originated a new support point
        if stage == len(rnd_cont.stage_vectors) or len(duals_vars) == 0:
            return None, None
        
        for (i, tup) in enumerate(tup_ind):
            # print(tup, duals_vars[i])
            k = tup[0]
            out_i = tup[1]
            new_pmf.append(duals_vars[i])
            new_supp_point = rnd_cont[stage + 1].outcomes[out_i].copy()
            for ele in rnd_cont[stage + 1].elements:
                delta_change = self._aux_norm_ctrs[(k, out_i, ele)].Pi
                new_supp_point[ele] = new_supp_point[ele] - delta_change / duals_vars[i]
            new_support.append(new_supp_point)
            support_map.append(set([out_i]))
        
        # Support points might be repeated, so they are merged if needed
        ix_1 = len(new_support) - 1
        while ix_1 > 0:
            ix_2 = ix_1 - 1
            while ix_2 >= 0:
                if np.sqrt(np.sum(
                    (new_support[ix_1][e] - new_support[ix_2][e])**2 for e in new_support[ix_1])) < cs.ZERO_TOL:
                    new_support.pop(ix_1)  # Erase duplicate support
                    new_pmf[ix_2] += new_pmf.pop(ix_1)  # Combined probabilites
                    support_map[ix_2].update(support_map.pop(ix_1))
                    break
                ix_2 -= 1
            ix_1 -= 1
        
        rnd_cont[stage + 1].worst_case_dist = {'support': new_support, 'pmf': new_pmf, 'map': support_map}
        return new_support, new_pmf


class DistRobustDuality(AbstractRiskMeasure):
    INF_NORM = 'inf_norm'
    L1_NORM = 'L1_norm'
    L2_NORM = 'L2_norm'
    '''
    Attributes:
        dro_ctrs (dic of GRBCtr): dictionary storing the constraints whose dual variables relates to worst-case probs.  
    '''
    
    def __init__(self, set_type=L1_NORM, radius=0, nominal=None, cutting_planes=False, data_random_container=None):
        
        super().__init__()
        self.cutting_planes_approx = cutting_planes
        self.set_type = set_type
        self.radius = radius
        self.q = nominal
        self.data_random_container = data_random_container
        self.dro_ctrs = {}
        if self.cutting_planes_approx == True:
            self.cuts_handler = None
            self.cut_index = 0
    
    def compute_cut_gradient(self, sp, sp_next, srv, soo, spfs, cut_id):
        '''
            See abstrac method. This risk measure does not support the
            single cut implmentation.
        '''
        if sp.multicut == False:  #single cut
            raise 'Risk measure does not support single cut'
        else:  #Multicut
            return super().compute_cut_gradient(sp, sp_next, srv, soo, spfs, cut_id)
    
    def compute_cut_intercept(self, sp, sp_next, srv, soo, spfs, cut_id):
        '''
            See abstrac method. This risk measure does not support the
            single cut implmentation.
        '''
        if sp.multicut == False:  #single cut
            raise 'Risk measure does not support single cut'
        else:  #Multicut
            return super().compute_cut_intercept(sp, sp_next, srv, soo, spfs, cut_id)
    
    def update_cut_intercept(self):
        raise 'Cut intercepts should not be updated in this risk measure.'
    
    def modify_stage_problem(self, sp, model, next_stage_rnd_vector):
        '''
        Modify the stage problem model to incorporate the DRO
        risk measure as dual variables of the inner problem.
        
        Args:
            sp (StageProblem): Stage problem to modify.
            model (GRBModel): Model object associate to the stage.
            n_outcomes(int): Number of outcomes for the following stage.
            
        '''
        assert sp.multicut, 'This risk measure implementation is only compatible with multicut setting.'
        t = sp.stage
        if sp._last_stage:
            return
        n_outcomes = len(next_stage_rnd_vector.outcomes)
        assert n_outcomes == len(
            self.data_random_container[t + 1].outcomes
        ), 'Inconsistent data was passed to the risk measure. Verify that the random builder function provided to build the SDDP model uses the same random container object.'
        
        if type(self.q) == type(None):
            self.q = next_stage_rnd_vector.p_copy
        set_type = self.set_type
        if set_type in [DistRobustDuality.L1_NORM, DistRobustDuality.L2_NORM]:
            r = self.radius
            q = self.q
            
            lambda_var = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='lambda[%i]' % (t))
            gamma_var = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='gamma[%i]' % (t))
            nu_var = model.addVars(n_outcomes,
                                   lb=-GRB.INFINITY,
                                   ub=GRB.INFINITY,
                                   vtype=GRB.CONTINUOUS,
                                   name='nu[%i]' % (t))
            model.update()
            #Update objective function
            new_objective = sp.cx + lambda_var + r * gamma_var - quicksum(q[i] * nu_var[i] for i in range(n_outcomes))
            model.setObjective(new_objective, GRB.MINIMIZE)
            self.dro_ctrs = model.addConstrs((lambda_var - nu_var[i] - sp.oracle[i] >= 0 for i in range(n_outcomes)),
                                             'dro_dual_ctr')
            
            if self.cutting_planes_approx == False:
                
                if set_type == DistRobustDuality.L2_NORM:
                    model.params.QCPDual = 1
                    model.params.BarQCPConvTol = 1E-8
                    model.addConstr(
                        quicksum(nu_var[i] * nu_var[i] for i in range(n_outcomes)) <= gamma_var * gamma_var,
                        'norm_dro_ctr')
                elif set_type == DistRobustDuality.L1_NORM:
                    model.addConstrs((nu_var[i] + gamma_var >= 0 for i in range(n_outcomes)), 'norm_dro_ctr_pos')
                    model.addConstrs((-nu_var[i] + gamma_var >= 0 for i in range(n_outcomes)), 'norm_dro_ctr_neg')
            else:  #Using cutting planes
                #Add max norm to bound
                model.addConstrs((nu_var[i] + gamma_var >= 0 for i in range(n_outcomes)), 'norm_dro_ctr_pos')
                model.addConstrs((-nu_var[i] + gamma_var >= 0 for i in range(n_outcomes)), 'norm_dro_ctr_neg')
                
                if set_type == DistRobustDuality.L2_NORM:
                    
                    def g(x, order):
                        n = len(x)
                        gamma = x[-1]
                        if order == 0:
                            g_val = np.sqrt(sum(x[i]**2 for i in range(n - 1))) - gamma
                            return g_val
                        elif order == 1:
                            norm_v = np.sqrt(sum(x[i]**2 for i in range(n - 1)))
                            g_grad = [x[i] / norm_v for i in range(n)]
                            g_grad[-1] = -1
                            return g_grad
                        else:
                            raise 'Order of the function is either 0 or 1.'
                    
                    var_names = [nu_var[v].VarName for v in nu_var]
                    var_names.append(gamma_var.VarName)
                    self.cuts_handler = DRO_CuttingPlanes([g], var_names)
                elif set_type == DistRobustDuality.L1_NORM:
                    self.cutting_planes_approx = False  # There is nothing to approximate
            
            model.update()
        else:
            raise 'Set different from L1 and L2 norm are not supported'
    
    def forward_pass_updates(self, sp, **kwargs):
        '''
        Runs updates associated to the risk measure during the forward pass
        and after the model is optimized.
        
        Return:
            Tuple with the following information:
                A boolean flag indicating if the subproblem needs to be re-solved
                A numeric value for the constraint violation
        '''
        
        tol = kwargs['fea_tol']
        if self.cutting_planes_approx == True and sp._last_stage == False:
            cph = self.cuts_handler
            var_vals = [sp.model.getVarByName(vn).X for vn in cph.dual_set_var]
            
            fea, ctr, vio = cph.check_dro_feasibility(var_vals, tol)
            if fea == False:
                model_vars = [sp.model.getVarByName(vn) for vn in cph.dual_set_var]
                cut_lhs = cph.refine_set(var_vals, model_vars, ctr, vio)
                #print(sp.stage, '__>' , vio, cut_lhs)
                sp.model.addConstr(cut_lhs, GRB.LESS_EQUAL, 0, 'dus_cut_%i' % (self.cut_index))
                self.cut_index = self.cut_index + 1
                resolve = False
                if vio > tol:
                    resolve = True
                return resolve, vio
            return False, vio
        return False, 0
    
    def forward_prob_update(self, t, sp, next_sp, fp_out_states, sample_path, rnd_container):
        '''
        Updates the probability distribution for the descendants of the current stage.
        The update sets the probability use to sample new outcomes to the worst case prob
        distribution for the particular sample path being explored.
        Args:
            t (int): current stage (just solved in forward pass)
            rnd_container (RandomContainer): object that contains all the randomness in the problem.
            
        This method modifies the probabilities of the random vector for stage t+1.
        '''
        if t == len(rnd_container.stage_vectors) - 1:
            #Last stage needs no update
            return
        desc_rnd_vec = rnd_container[t + 1]  #Descendant outcomes
        p_w = np.zeros(desc_rnd_vec.outcomes_dim)
        for i in self.dro_ctrs:
            p_w[i] += self.dro_ctrs[i].Pi
        p_w = np.around(np.abs(p_w), 8)
        p_w = p_w / p_w.sum()
        #Mix with nominal distribution
        ds_beta = cs.options['dynamic_sampling_beta']
        p_w = ds_beta * p_w + (1 - ds_beta) * desc_rnd_vec.p_copy
        desc_rnd_vec.modifyOutcomesProbabilities(p_w)


class DRO_CuttingPlanes():
    '''
    Implements helper function to do an outer approximation of the 
    of the uncertainty set in the DRO setting. In particular, it approximates 
    the dual of a given uncertainty set
    
    Attrs:
        dual_dro_set (list of func): A list of functions that define the dual uncertatny set.    
            Each function has the signature:
                def g(vars_values, order):
                    vars_values: a point to evalute the function
                    order:  0 return the function evaluation
                            1 returns a subgradient in the same format as the input
            The set is assumed to be g_i(x,0) <= 0 \forall i
        dual_set_var (collection of str): A list of place holders for the variables that define the set
    '''
    
    def __init__(self, dual_dro_set, dual_dro_var):
        self.dual_dro_set = dual_dro_set
        self.dual_set_var = dual_dro_var
    
    def check_dro_feasibility(self, var_values, tolerance):
        '''
        Check feasibility of the set
        Args:
            var_values (collection of floats): Current value of the variables.
            tolerance (float): Feasibility tolerance.
        Return:
            status (bool): True is current point is feasible, False o.w.
            most_violated (int): index of the most volated inequality
            max_violation (float): violation of the constraint (value of the function)
        '''
        assert tolerance >= 0, 'Feasibility tolerance must be positive.'
        assert type(var_values) == type(self.dual_set_var), 'Invalid format of the variables'
        most_violated = None
        max_violation = tolerance
        for (i, f) in enumerate(self.dual_dro_set):
            f_val = f(var_values, 0)
            if f_val > max_violation:
                max_violation = f_val
                most_violated = i
        
        if most_violated != None:
            return False, most_violated, max_violation
        else:
            return True, None, 0
    
    def refine_set(self, var_values, vars, g_func, g_val):
        '''
        Generate a valid inequality that refines the set
        Args:
            var_values (collection of floats): Current value of the variables.
            vars (collection of GRBVar): Variables of the model.
            g_func (func): Function associated to the most violated inequality.
            g_val (float): current value of the function
        Return:
            cut (LinExpr): A linear expression of the cut. 
        '''
        
        sub_gradient = self.dual_dro_set[g_func](var_values, 1)
        
        cut_intercept = 0  #  For 2 norm the intercept is zero! g_val - quicksum(sub_gradient[i]*var_values[i] for i in range(len(vars)))
        cut_lhs = quicksum(sub_gradient[i] * vars[i] for i in range(len(vars))) + cut_intercept
        return cut_lhs


class DistRobust(AbstractRiskMeasure):
    '''
        Distributionally robust risk measure based on primal formulation (Philpott et al.). 
    '''
    
    #Constants to identify the type of distance being used.
    INF_NORM = 'inf_norm'
    L1_NORM = 'L1_norm'
    L2_NORM = 'L2_norm'
    D_Wasserstein = 'Discrete_Wasserstein'
    
    def __init__(self, dro_inner_solver, **dro_solver_params):
        '''
        inner_solver (dro_inner_solver): a class that computes the worst distribution on the 
            backward pass of the algorithm given descendants objective values. In particular
            it solves the inner max problem of a particular stage to where it belongs.
        '''
        super().__init__()
        self.inner_solver = dro_inner_solver(**dro_solver_params)
        self.inner_solver_class = dro_inner_solver
        self.inner_solver_param = dro_solver_params
        
        #For multi-cut version
        self.global_oracle = None
        
        self._wors_case_dist = None
        self._static_dist = True
    
    def compute_cut_gradient(self, sp, sp_next, srv, soo, spfs, cut_id):
        '''
        Computes expected dual variables for the single cut version
        and then the gradient.
        
        Args:
            sp (StageProblem): current subproblem where the cut will be added.
            sp_nex (StageProblem): subproblem for the next stage.
            srv (StageRandomVector): Random vector of the next stage
            soo (List of dict): A list of outputs of all the subproblems descendants.
            spfs (dict (str-float)): Values for the states of the current stage computed 
                in the forward pass.
            cut_id (int): Numeric id of the cut being created (iteration in SDDP)
        Return 
            pi_bar (list[dict]): Expected value of the duals. For the single cut algorithm
                the list contains just one element.
            cut_gradient_coeff(list[dict]):
        '''
        #Solve inner max problem on the side
        zs = np.array([soo[i]['objval'] for i in range(len(srv.outcomes))])
        p = self.inner_solver.compute_worst_case_distribution(zs)
        self._wors_case_dist = p
        
        #Get cut_gradient
        cut_gradient_coeff = super().compute_cut_gradient(sp, sp_next, srv, soo, spfs, cut_id, p)
        
        #Add extra constraints for multicut implementation if needed.
        if sp.multicut:
            lhs_exp = self.global_oracle - quicksum(p[i] * sp.oracle[i] for i in sp.oracle)
            try:
                if -lhs_exp.getValue() > FEASIBILITY_TOL:
                    sp.model.addConstr(lhs=lhs_exp,sense=GRB.GREATER_EQUAL,rhs=0,name='unicut[%i]' % (cut_id))
            except AttributeError:
                sp.model.addConstr(lhs=lhs_exp,sense=GRB.GREATER_EQUAL,rhs=0,name='unicut[%i]' % (cut_id))

            
        
        self._current_cut_gradient = cut_gradient_coeff
        return cut_gradient_coeff
    
    def compute_cut_intercept(self, sp, sp_next, srv, soo, spfs, cut_id):
        '''
        Computes cut intercept(s) 
        
        Args: (as in cut gradient method)
        Returns:
            cut_intercepts (list of int): List of intercep(s) of the cut(s).
        '''
        p = self._wors_case_dist
        return super().compute_cut_intercept(sp, sp_next, srv, soo, spfs, cut_id, p)
    
    def update_cut_intercept(self):
        pass
    
    def modify_stage_problem(self, sp, model, next_stage_rnd_vector):
        '''
        Modify the stage problem to accommodate additional variables and constraints
        defined in the DRO appoach
        '''
        self.inner_solver.build_model(t=sp.stage, next_stage_rnd_vec=next_stage_rnd_vector)
        if sp.multicut == False:
            model.setObjective(sp.cx + sp.oracle.sum())
        else:
            self.global_oracle = model.addVar(lb=-GRB.INFINITY,
                                              ub=GRB.INFINITY,
                                              vtype=GRB.CONTINUOUS,
                                              name='GlobalOracle[%i]' % (sp.stage))
            model.addConstr(lhs=self.global_oracle - quicksum(sp.oracle[i] for i in sp.oracle) / len(sp.oracle),
                            sense=GRB.GREATER_EQUAL,
                            rhs=0,
                            name='unicut[ini]')
            model.setObjective(sp.cx + self.global_oracle)
    
    def forward_pass_updates(self, *args, **kwargs):
        'Default is False for sub resolve and 0 for constraint violations'
        return False, 0
    
    def forward_prob_update(self, t, sp, next_sp, forward_out_states, sample_path, rnd_container):
        '''
            Computes a proxy of the worst case distribution by solving the inner max problem
            with the value of the oracles as objective function.
        '''
        if next_sp == None:
            #No change in probabilities
            return
        assert sp.model.status == GRB.OPTIMAL, 'Previous stage problem is assume to be solved.'
        next_rnd_vector = rnd_container[t + 1]  #Descendant outcomes
        omega_t = next_rnd_vector.getOutcomes(sample_path, ev=False)
        zs = np.zeros(len(omega_t))
        if sp.multicut:
            for (k, oracle) in sp.oracle.items():
                zs[k] = oracle.X
        else:
            for (i, outcome) in enumerate(omega_t):
                next_sp_output = next_sp.solve(in_state_vals=forward_out_states[t - 1],
                                               random_realization=outcome,
                                               forwardpass=False,
                                               random_container=rnd_container,
                                               sample_path=sample_path)
                zs[i] = next_sp_output['objval']
        
        p_worst = self.inner_solver.compute_worst_case_distribution(zs)
        ds_beta = cs.options['dynamic_sampling_beta']
        p_w = ds_beta * p_worst + (1 - ds_beta) * next_rnd_vector.p_copy
        next_rnd_vector.modifyOutcomesProbabilities(p_w)
    
    def modify_param(self, **kwags):
        for kw_name in kwags:
            if kw_name in self.inner_solver_param:
                self.inner_solver_param[kw_name] = kwags[kw_name]
                self.inner_solver.modify_inner_solver(**{kw_name: kwags[kw_name]})
            else:
                raise 'Parameter %s is not defined in risk measure %s' % (kw_name, type(self))


class DistRobusInnerSolver(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def compute_worst_case_distribution(self):
        pass


class InnerDROSolverX2(DistRobusInnerSolver):
    def __init__(self, nominal_p, DUS_radius, set_type):
        self.nominal_p = nominal_p
        self.uncertanty_radius = DUS_radius
    
    def compute_worst_case_distribution(self, outcomes_objs):
        '''
        Implements Philpott et al 2017 algorithm to compute the worst case probability distribution. 
        '''
        m = len(outcomes_objs)
        
        r = self.uncertanty_radius
        sorted_inidices = sorted(range(m), key=lambda k: outcomes_objs[k])
        for k in range(0, m - 1):
            p = np.zeros(m)
            z_bar = sum(outcomes_objs[sorted_inidices[i]] for i in range(k, m)) / (m - k)
            s = np.sqrt(sum(outcomes_objs[sorted_inidices[i]]**2 - z_bar**2 for i in range(k, m)) / (m - k))
            
            broke = False
            for i in range(k, m):
                normalized = (outcomes_objs[sorted_inidices[i]] - z_bar) / s
                if np.isnan(normalized):
                    broke = True
                    break
                p[sorted_inidices[i]] = (1 / (m - k)) + np.sqrt((r**2) * (m - k) - k / m) * normalized / (m - k)
            if p[sorted_inidices[k]] >= 0 and broke == False:
                assert np.abs(sum(p) - 1) <= cs.ZERO_TOL, 'Sum of scenario probs exceeds 1.'
                return p
        
        p = np.zeros(m)
        p[sorted_inidices[-1]] = 1
        return p


class PhilpottInnerDROSolver(DistRobusInnerSolver):
    def __init__(self, radius, set_type, data_random_container):
        self.data_random_container = data_random_container
        self.uncertanty_radius = radius
        self._one_time_warning = True
        self.set_type = set_type
        self.nominal_p = None
    
    def build_model(self, **kwargs):
        t = kwargs['t']
        nsrv = kwargs['next_stage_rnd_vec']
        assert len(self.data_random_container[t + 1].outcomes) == len(nsrv.outcomes)
        q = nsrv.p_copy
        DUS_radius = self.uncertanty_radius
        set_type = self.set_type
        m = Model('DRO_solver')
        m.params.OutputFlag = 0
        #m.params.FeasibilityTol = 1E-9
        p = m.addVars(len(q), lb=0, ub=1, obj=1, vtype=GRB.CONTINUOUS, name='p')
        m.update()
        m.addConstr(p.sum(), GRB.EQUAL, 1, 'prob')
        if set_type == DistRobustDuality.L2_NORM:
            m.addConstr(quicksum((p[i] - q[i]) * (p[i] - q[i]) for i in p), GRB.LESS_EQUAL, DUS_radius**2, 'DUS')
        elif set_type == DistRobustDuality.L1_NORM:
            d_plus = m.addVars(len(q), lb=0, ub=1, obj=0, vtype=GRB.CONTINUOUS, name='dp')
            d_minus = m.addVars(len(q), lb=0, ub=1, obj=0, vtype=GRB.CONTINUOUS, name='dm')
            m.addConstr(quicksum(d_plus[i] + d_minus[i] for i in p), GRB.LESS_EQUAL, DUS_radius, 'DUS')
            m.addConstrs((d_plus[i] - d_minus[i] == p[i] - q[i] for i in p), 'ABS_liner')
        else:
            raise 'Not supported norm for this solver.'
        m.setObjective(m.getObjective(), GRB.MAXIMIZE)
        m.update()
        self.model = m
        self.p_var = p
        self.nominal_p = q
    
    def compute_worst_case_distribution(self, outcomes_objs):
        '''
        Compute the worst-case probability distribution for a particular stage
        given the objective function value of the descendent nodes (as many as outcomes)
        Args:
            outcomes_objs (ndarray): vector of objectives values of the next stage.
        Returns:
            new_p (ndarray): array with the worse case probabilities in DRO
        '''
        vars = self.p_var
        for (i, z) in enumerate(outcomes_objs):
            vars[i].obj = z
        self.model.update()
        self.model.optimize()
        
        if self._one_time_warning:
            self._one_time_warning = False
            #sddp_log.warning('DRO inner problem not implemented!')
        assert len(outcomes_objs) == len(self.nominal_p)
        new_p = np.array([vars[i].X for i in range(len(vars))])
        return new_p


class DiscreteWassersteinInnerSolver(DistRobusInnerSolver):
    '''
    Implements the linear program to obtain the worst-case
    distribution in the discrete setting, i.e., there is a finite
    number of support points that will ship probability mass to 
    a finite (potentially larger) set of support points.
    '''
    
    def __init__(self, norm=1, radius=1, dist_func=norm_fun, data_random_container=None):
        self.norm = norm
        self.radius = radius
        self.dist_func = dist_func
        self.org_rnd_cnt = data_random_container
        self.model = None
        self.worst_p = None
    
    def build_model(self, t, next_stage_rnd_vec):
        '''
        Constructs the model that will be used to 
        compute the worst case probabilities.
        
        Args:
            t (int): stage id for which the model is the inner problem
            next_stage_rnd_vec (StageRandomVector): random vector for  stage t + 1
        '''
        srv = next_stage_rnd_vec
        nsrv_org = srv  #Origin points in the support for the transport problem
        if self.org_rnd_cnt != None:
            nsrv_org = self.org_rnd_cnt[t + 1]
        
        nsrv_des = srv  #Destination points in the support for the transport problem
        n_org = nsrv_org.outcomes_dim
        n_des = nsrv_des.outcomes_dim
        
        model = Model()
        model.params.OutputFlag = 0
        p = model.addVars(n_des, lb=0, ub=1, obj=0, vtype=GRB.CONTINUOUS, name='p[t_%i]' % (t))
        z = model.addVars(n_org, n_des, lb=0, ub=1, obj=0, vtype=GRB.CONTINUOUS, name='z[t_%i]' % (t))
        model.update()
        model.addConstrs((z.sum(o, '*') == nsrv_org.p_copy[o] for o in range(n_org)), name='org[t_%i]' % (t))
        model.addConstrs((z.sum('*', d) == p[d] for d in range(n_des)), name='des[t_%i]' % (t))
        
        DUS_exp = LinExpr()
        for i in range(n_org):
            xi_i = nsrv_org.get_sorted_outcome(i)
            for j in range(n_des):
                xi_j = nsrv_des.get_sorted_outcome(j)
                d_ij = self.dist_func(xi_i, xi_j, self.norm)
                DUS_exp.addTerms(d_ij, z[i, j])
        model.addConstr(lhs=DUS_exp, sense=GRB.LESS_EQUAL, rhs=self.radius, name='dro_radius')
        model.ModelSense = GRB.MAXIMIZE
        model.update()
        
        #Save the model an the variables reference
        self.model = model
        self.worst_p = p
    
    def compute_worst_case_distribution(self, outcomes_objs):
        '''
        Compute the worst-case probability distribution for a particular stage
        given the objective function value of the descendent nodes (as many as outcomes)
        Args:
            outcomes_objs (ndarray): vector of objectives values of the next stage.
        Returns:
            new_p (ndarray): array with the worse case probabilities in DRO
        '''
        assert len(outcomes_objs) == len(self.worst_p)
        for (i, out_obj) in enumerate(outcomes_objs):
            self.worst_p[i].Obj = out_obj
        self.model.update()
        self.model.optimize()
        
        if self.model.status == GRB.OPTIMAL:
            new_p = np.array([self.worst_p[i].X for i in range(len(outcomes_objs))])
            return new_p
        else:
            raise 'Discrete Wasserstein model was not optimal.'
    
    def modify_inner_solver(self, radius=None):
        if self.model != None:
            self.model.getConstrByName('dro_radius').RHS = radius
            self.model.update()


class ContinuousWassersteinInnerSolver(DistRobusInnerSolver):
    '''
    .
    '''
