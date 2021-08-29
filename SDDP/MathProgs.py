'''
Created on Nov 17, 2017

@author: dduque
'''
from gurobipy import GRB, Model, tupledict, read
from SDDP.CutManagament import Cut
from SDDP.CutManagament import CutPool
import SDDP
from SDDP import *
from time import time
import scipy.sparse as sp
from SDDP.RiskMeasures import Expectation


class StageProblem():
    '''
    Class to represent a general stage problem
    
    Attributes:
        model (GRB Model): Model of the stage
        cuts (CutPool): Object storing cuts
    '''
    
    def __init__(self,
                 stage,
                 model_builder,
                 next_stage_rnd_vector,
                 lower_bound,
                 last_stage=False,
                 risk_measure=Expectation(),
                 multicut=False):
        '''
        Constructor
        
        Args:
            stage(int): Stage of the model
            model_builder(func): A function that build the stage model (GRBModel)
            next_stage_rnd_vector (StageRandomVector): Random vector that characterizes realizations in the next stage.
            lower_bound (float): A valid lower bound on the problem objective. 
            last_stage (bool): Boolean flag indicating if this stage problem is the last one.
            risk_measure (AbstractRiskMeasure): A risk measure object to form the cuts.
            multicut (bool): Boolean flag indicating weather to use multicut or single cut.
        '''
        self.stage = stage
        self._last_stage = last_stage
        self.model_stats = MathProgStats()
        model_setup = model_builder(stage)
        if len(model_setup) == 4:
            model, in_states, out_states, rhs_vars = model_setup
            out_in_map = None
        else:
            model, in_states, out_states, rhs_vars, out_in_map = model_setup
        
        self.model = model
        self.risk_measure = risk_measure
        self.lower_bounds = []
        self._lower_bound_on = True
        self.cut_pool = CutPool(self)
        self.multicut = multicut
        
        self.states_map = {}
        self.out_in_map = out_in_map
        self.in_state = [x for x in in_states]  # Names of incoming states
        self.out_state = [x for x in out_states]  # Names of outgoing states
        self.in_state_var = {x: model.getVarByName(x) for x in in_states}
        self.out_state_var = {x: model.getVarByName(x) for x in out_states}
        self.gen_states_map(self.in_state)
        self.rhs_vars = [x for x in rhs_vars]
        self.rhs_vars_var = {x: model.getVarByName(x) for x in rhs_vars}
        
        # Optimizer parameters
        self.model.params.OutputFlag = 0  # No optimization output
        self.model.params.DualReductions = 0  # Give definite info on status (when inf or unbounded)
        self.model.params.Threads = SDDP.options['grb_threads']
        self.model.params.Method = 1
        #self.model.params.NumericFocus = 3
        #self.model.params.PreDual = 0
        #self.model.params.Presolve = 0
        #self.model.params.Crossover = 0
        #self.model.params.CrossoverBasis = 1
        #self.model.params.NormAdjust = 2
        #self.model.params.ObjScale = 1
        #self.model.params.Quad = 1
        #self.model.params.ScaleFlag = 0
        #self.model.params.SimplexPricing = 3
        #self.model.params.FeasibilityTol = 1E-9
        
        # Add oracle var(s) and include it in the objective
        self.cx = self.model.getObjective()  # Get objective before adding oracle variable
        if last_stage == False:
            self.n_outmes = next_stage_rnd_vector.outcomes_dim
            num_outcomes = next_stage_rnd_vector.outcomes_dim
            if multicut == False:  # Gen single cut variable
                self.oracle = self.model.addVars(1, lb=lower_bound, vtype=GRB.CONTINUOUS, name='oracle[%i]' % (stage))
                self.lower_bounds = [lower_bound]
                #self.orcale_bound  = self.model.addConstr((self.oracle[0]>=lower_bound), "oracle_bound")
            else:
                if num_outcomes == 0:
                    raise 'Multicut algorithm requires to define the number of outcomes in advance.'
                self.oracle = self.model.addVars(num_outcomes,
                                                 lb=lower_bound,
                                                 vtype=GRB.CONTINUOUS,
                                                 name='oracle[%i]' % (stage))
                self.lower_bounds = [lower_bound for _ in range(num_outcomes)]
            self.model.update()
            risk_measure.modify_stage_problem(self, self.model, next_stage_rnd_vector)
        
        #Construct dictionaries of (constraints,variables) key where duals are needed
        self.ctrsForDuals = set()  #Set of names
        self.ctrsForDualsRef = {}  #Dict containing the reference to the constraints
        self.ctrInStateMatrix = {}
        for vname in self.in_state:
            var = self.model.getVarByName(vname)
            col = self.model.getCol(var)
            for j in range(0, col.size()):
                ctr = col.getConstr(j)
                ctr_coeff = col.getCoeff(j)
                self.ctrsForDuals.add(ctr.ConstrName)
                self.ctrsForDualsRef[ctr.ConstrName] = ctr
                self.ctrInStateMatrix[ctr.ConstrName, vname] = -ctr_coeff
        
        self.ctrRHSvName = {}
        for vname in self.rhs_vars:
            var = self.model.getVarByName(vname)
            col = self.model.getCol(var)
            #assert col.size() == 1, "RHS noise is not well defined"
            for c_index in range(col.size()):
                ctr = col.getConstr(c_index)
                assert ctr.ConstrName not in self.ctrRHSvName, "Duplicated RHS noise variable in the a single constraint."
                self.ctrRHSvName[ctr.ConstrName] = vname
                self.ctrsForDuals.add(ctr.ConstrName)
                self.ctrsForDualsRef[ctr.ConstrName] = ctr
    
    def lower_bounding_solve(self, in_state_lb, in_state_ub, random_realization=None):
        '''
        Computes a lower bound on stage problem t to set lower bounds on the
        oracle(s) of stage t-1. To do this, the state variables of stage t-1
        are allowed to change in the optimization problem of stage t. 
        
        Args:
            in_state_lb (dict): A dictionary with the lower bound for the incoming 
                state variables of stage problem t.
            in_state_ub (dict): A dictionary with the upper bound for the incoming 
                state variables of stage problem t.
            random_realization: A dictionary with the realization of the random vector
                for which the lower bound is going to be computed.
        '''
        assert self.stage > 0, "Initial lower bound computation is for stages t=2,...,T"
        
        #Setup stage problem t to get a lower bound
        for (in_name, in_var) in self.in_state_var.items():
            in_var.lb = in_state_lb[in_name]
            in_var.ub = in_state_ub[in_name]
        assert len(random_realization) == len(
            self.rhs_vars), "In random vector has different cardinality than expected %i" % (
                len(random_realization) - len(self.rhs_vars)) + random_realization + " " + str(self.rhs_vars)
        for rr in random_realization:
            self.rhs_vars_var[rr].lb = random_realization[rr]
            self.rhs_vars_var[rr].ub = random_realization[rr]
        
        self.model.update()
        self.model.optimize()
        
        #Cases handling
        status = self.model.status
        if status == GRB.UNBOUNDED:
            raise 'Original problem has unbounded objective for the given bounds on the states.'
        elif status == GRB.INFEASIBLE:
            raise 'Original problem is infeasible.'
        elif status == GRB.OPTIMAL:
            output = {}
            output['objval'] = self.model.ObjVal
            return output
        else:
            raise 'Problem fail in initial lower bounding phase'
    
    def solve(self,
              in_state_vals=None,
              random_realization=None,
              forwardpass=False,
              random_container=None,
              sample_path=None,
              num_cuts=0):
        '''
        Solves a stage problem given the state variables of the previous stage
        Args:
            in_state_vals (dict of real): dictionary containing the values of the state variables
                in the previous stage. They are referred as in_state for this model.
            random_realization (dic of real): dictionary containing the realization values to be solved.
                the key is the name of the variable that models the random value as a placeholder.
        
        Output:
            output (dict of objects): a dictionary with output information
                'status': Optimization model status
                'duals': dictionary of the dual variables associated to previous 
                        stage cuts.
                'out_state': Value of the state variables at the end of the sate
                        (input for the next stage).
                'cputime': float time solving the problem
                'lptipe': float value of the time solving the lp only
                'cutupdatetime' float value of the time updating the cuts.
        '''
        tnow = time()
        cutupdatetime = 0
        setuptime = 0
        
        if self.stage > 0:
            setuptime = time()
            assert len(in_state_vals) == len(self.in_state), "In state vector has different cardinality than expected"
            for in_s_name in in_state_vals:
                in_s_val = in_state_vals[in_s_name]
                self.states_map[in_s_name].lb = in_s_val
                self.states_map[in_s_name].ub = in_s_val
            
            assert len(random_realization) == len(
                self.rhs_vars), "In random vector has different cardinality than expected %i" % (
                    len(random_realization) - len(self.rhs_vars)) + random_realization + " " + str(self.rhs_vars)
            for rr in random_realization:
                self.rhs_vars_var[rr].lb = random_realization[rr]
                self.rhs_vars_var[rr].ub = random_realization[rr]
            self.model.update()
            setuptime = time() - setuptime
            
            cutupdatetime = time()
            self.update_cut_pool(random_container, random_realization)
            cutupdatetime = time() - cutupdatetime
            self.model.update()
        '''
        Solve model and handle possible outcomes
        '''
        lp_time = time()
        self.model.update()
        self.model.optimize()
        if self.model.status != GRB.OPTIMAL:
            self.model.reset()
            self.model.optimize()
        lp_time = time() - lp_time
        '''
        Retrieve solution
        '''
        data_mgt_time = time()
        output = {}
        status = gurobiStatusCodeToStr(self.model.status)
        output['status'] = status
        if status == SP_OPTIMAL:
            output['objval'] = self.model.ObjVal
            if forwardpass == True:
                resolves = 0
                resolve, violation = self.risk_measure.forward_pass_updates(self, fea_tol=1E-2)
                while resolve and resolves < 2:
                    #print('Pass %i: resolving %i for violation %f' %(num_cuts, self.stage,violation))
                    lp_time_resolve = time()
                    self.model.optimize()
                    lp_time += time() - lp_time_resolve
                    resolve, violation = self.risk_measure.forward_pass_updates(self, fea_tol=1E-2)
                    resolves += 1
                
                output['risk_measure_info'] = violation
                
                output['out_state'] = {}
                for vname in self.out_state_var:
                    val = self.out_state_var[vname].X
                    v_lb = self.out_state_var[vname].lb
                    v_ub = self.out_state_var[vname].ub
                    if val < v_lb:
                        val = v_lb
                    elif val > v_ub:
                        val = v_ub
                    output['out_state'][vname] = val
            else:
                output['duals'] = {cname: self.ctrsForDualsRef[cname].Pi for cname in self.ctrsForDuals}
                output['RC'] = {
                    self.states_map[v_in].VarName: self.model.getVarByName(v_in).RC
                    for v_in in self.in_state
                }
                output['dual_obj_rhs_noice'] = sum(self.rhs_vars_var[self.ctrRHSvName[ctr_name]].UB *
                                                   output['duals'][ctr_name] for ctr_name in self.ctrRHSvName)
                if random_container[self.stage].is_independent == False:
                    #Cut duals are only used to recompute the cut
                    output['cut_duals'] = {cut.name: cut.ctrRef.Pi for cut in self.cut_pool if cut.is_active}
            
            # Update statistics
            self.model_stats.add_simplex_iter_entr(self.model.IterCount)
        
        else:
            pass  # raise 'Model is not optimal, status: %i' % (self.model.status)
            
            #if self.stage == 0:
            #self.print_theta()
        
        output['lptime'] = lp_time
        output['cutupdatetime'] = cutupdatetime
        output['setuptime'] = setuptime
        data_mgt_time = time() - data_mgt_time
        output['datamanagement'] = data_mgt_time
        output['cputime'] = time() - tnow
        
        return output
    
    def print_stage_res_summary(self):
        strout = ''
        #=======================================================================
        # for v in self.model.getVars():
        #     if 'generation' in v.varname or 'inflow[1' in v.varname or 'reservoir_level[1' in v.varname:
        #         strout = strout + '%20s:%10.3f;' %(v.varname, v.X)
        #=======================================================================
        strout = '======== Stage %s %.3f, HydroGen: %10.2f,  Thermal Gen %10.2f =========\n' % (
            self.stage, self.cx.getValue(), self.model.getVarByName('generation').X,
            self.model.getVarByName('thermal_gen').X)
        strout += 'oracle %16.3f; \n' % (self.model.ObjVal - self.cx.getValue())
        nr = 10
        strout += 'R0'
        for r in range(nr):
            v = self.model.getVarByName('reservoir_level0[%i]' % r)
            strout = strout + '+%10.3f; ' % (v.X)
        strout += '\n'
        #         for r in range(nr):
        #             v=self.model.getVarByName('innovations[%i]' %r)
        #             strout = strout + '+%10.3f; ' %(v.X)
        #         strout += '\n'
        strout += 'I '
        for r in range(nr):
            # v = self.model.getVarByName('inflow[%i,1]' % r)
            v = self.model.getVarByName('inflow[%i]' % r)
            strout = strout + '+%10.3f; ' % (v.X)
        strout += '\n'
        strout += 'P '
        for r in range(nr):
            v = self.model.getVarByName('pour[%i]' % r)
            strout = strout + '+%10.3f; ' % (v.X)
        strout += '\n'
        strout += 'O '
        for r in range(nr):
            v = self.model.getVarByName('outflow[%i]' % r)
            strout = strout + '-%10.3f; ' % (v.X)
        strout += '\n'
        strout += 'S '
        for r in range(nr):
            v = self.model.getVarByName('spill[%i]' % r)
            strout = strout + '-%10.3f; ' % (v.X)
        strout += '\n'
        strout += 'R '
        for r in range(nr):
            v = self.model.getVarByName('reservoir_level[%i]' % r)
            strout = strout + '=%10.3f; ' % (v.X)
        print(strout)
    
    def print_theta(self):
        tvals = np.array([self.oracle[v].X for v in self.oracle])
        args_t_vals = np.argsort(tvals)
        print(args_t_vals)
    
    def printPostSolutionInformation(self):
        print('------------------------------------')
        print('Model in stage %i: obj>> %f' % (self.stage, self.model.ObjVal))
        #=======================================================================
        # for c in self.model.getConstrs():
        #     print('%20s %10.3f %10.3f'  %(c.ConstrName, c.RHS, c.PI))
        #=======================================================================
        print('%20s %10s %10s %10s %10s %10s' % ('name', 'lb', 'ub', 'obj', 'x', 'RC'))
        for v in self.model.getVars():
            if np.abs(v.X) > 1E-8 and 'travel_ind' not in v.varname:
                print('%30s %15.3f %15.3e %15.3f %15.3f %15.3f' % (v.varname, v.LB, v.UB, v.obj, v.X, v.RC))
        print('------------------------------------\n')
    
    #===========================================================================
    # def get_in_state_var(self, out_state):
    #     '''
    #     Returns the corresponding in state variable name
    #     given a out state variable name of the previous
    #     stage.
    #     '''
    #     sindex = out_state.index('[')
    #     newkey = out_state[:sindex]+'0'+out_state[sindex:]
    #     return newkey
    #===========================================================================
    def gen_states_map(self, in_states):
        if self.out_in_map == None:
            for in_state in in_states:
                if '[' in in_state:
                    sindex = in_state.index('[')
                    #self.model.getVarByName(
                    self.states_map[in_state] = self.model.getVarByName(in_state[:sindex - 1] + in_state[sindex:])
                    self.states_map[in_state[:sindex - 1] + in_state[sindex:]] = self.model.getVarByName(in_state)
                else:
                    sindex = in_state.index('0')
                    self.states_map[in_state] = self.model.getVarByName(in_state[:sindex] + in_state[sindex + 1:])
                    self.states_map[in_state[:sindex] + in_state[sindex + 1:]] = self.model.getVarByName(in_state)
        else:
            self.states_map = self.out_in_map
    
    def get_out_state_var(self, in_state):
        '''
        Returns the corresponding out state variable name
        given a in state variable name of the next
        stage.
        '''
        return self.states_map[in_state].VarName
    
    def update_cut_pool(self, random_container, current_outcome):
        if self.stage > 0 and len(self.cut_pool) > 0 and self.cut_pool.needs_update():
            srv = random_container[self.stage]
            omega_stage_abs_order = np.zeros((len(current_outcome), 1))
            for rhs_c in current_outcome:
                omega_stage_abs_order[srv.vector_order[rhs_c]] = current_outcome[rhs_c]
            
            for cut in self.cut_pool:
                cut.adjust_intercept(omega_stage_abs_order)
    
    def createStageCut(self, cut_id, sp_next, rnd_vector_next, outputs_next, sample_path_forward_states, sample_path):
        '''
        Creates a cut for this stage problem
        
        Args:
            cut_id (int): Numeric id of the cut (corresponds to the number of backward passes).
            sp_next (StageProblem): stage problem object of the next stage
                TODO: This might not be enough for lags>1
            rnd_vector_next (StageRandomVector): random vector containing the random variables
                of the next stage.
            outputs_next (list of dict): list of outputs for every outcome in the next stage. 
                Each element of the list is a dictionary with the same structure as the output
                of ::func::StageProblem.solve.
            sample_path_forward_states(dict): dictionary with the values of the states of this
                stage that were computed in the forward pass associated with the current sample
                path.
            sample_path (list of dict): current sample path. Each element of the
                list corresponds to the realizations of a different stage.
        
        '''
        pi_bar = {}  #Expected duals of the transition function Ax = b + Bx0
        srv = rnd_vector_next
        soo = outputs_next
        spfs = sample_path_forward_states
        cut_gradient_coeffs = self.risk_measure.compute_cut_gradient(self, sp_next, srv, soo, spfs, cut_id)
        cut_intercepts = self.risk_measure.compute_cut_intercept(self, sp_next, srv, soo, spfs, cut_id)
        
        stagewise_ind = srv.is_independent
        if stagewise_ind:
            new_cuts = [
                Cut(self, cut_gradient_coeffs[i], cut_intercepts[i], cut_id, outcome=i)
                for (i, grad) in enumerate(cut_gradient_coeffs)
            ]
            self.cut_pool.addCuts(self.model, new_cuts)
        
        else:
            if self.multicut == True:
                raise "Multicut is not yet implemented for the dependent case"
            #ctrRHSvName
            raise 'NEED to compute pi_bar'
            alpha_bar = {}  #Expected duals of the cuts
            ab_D = np.zeros((1, len(pi_bar)))  #Computation alpha_bar_{t+1}*D_{t+1}
            for cut in sp_next.cut_pool:
                alpha_bar[cut.name] = sum(srv.p[i] * soo[i]['cut_duals'][cut.name]
                                          for (i, o) in enumerate(srv.outcomes))
                ab_D = ab_D + alpha_bar[cut.name] * cut.dep_rhs_vector
            
            omega_stage_abs_order = np.zeros((len(sample_path[self.stage]), 1))
            pi_bar_abs_order = np.zeros((1, len(pi_bar)))
            for cpi in pi_bar:
                rhs_c = sp_next.ctrRHSvName[cpi]
                pi_bar_abs_order[0, srv.vector_order[rhs_c]] = pi_bar[cpi]
                omega_stage_abs_order[srv.vector_order[rhs_c]] = sample_path[self.stage][rhs_c]
            dep_rhs_vector = (sp.csr_matrix(pi_bar_abs_order + ab_D)).dot(
                srv.autoreg_matrices[-1]).toarray()  #TODO: generalize for mor lags!!
            dep_rhs = dep_rhs_vector.dot(omega_stage_abs_order).squeeze()
            ind_rhs = cut_intercepts - dep_rhs
            new_cut = Cut(self,
                          cut_gradient_coeffs,
                          cut_intercepts,
                          cut_id,
                          stagewise_ind=False,
                          ind_rhs=ind_rhs,
                          dep_rhs_vector=dep_rhs_vector)
            self.cut_pool.addCut(self.model, new_cut)
    
    def get_stage_objective_value(self):
        '''
        Returns the stage cost value
        '''
        try:
            return self.cx.getValue()
        except:
            #TODO: log for a warning
            print("ERROR IN get_stage_objective_value function")
            return 0
    
    def remove_oracle_bounds(self):
        for k in self.oracle:
            self.oracle[k].lb = -GRB.INFINITY
        self.model.update()
    
    def update_cut_pool_dro(self, cuts_left):
        self.cut_pool.update_cut_pool_dro(self, self.model, cuts_left)
    
    def __repr__(self):
        return "SP(%i): #cuts:%i" % (self.stage, len(self.cut_pool.pool))


class StageOracleProblem():
    '''
    Class to represent an oracle model. An oracle model
    starts at a particular stage t and includes decision 
    variables of all stages from t to T. Randomnes for stage
    t is a place holder and is fixed to the expected value 
    for stages t+1...T. 
    '''
    
    def __init__(self, stage, T, model_builder):
        '''
        Constructor
        
        Args:
            stage(int): Stage of the model
            T (int): max number of stages
            model_builder(func): A function that build the stage oracle model (GRBModel)
                The signature of the oracle builder is:
                oracle_model_builder params:
                    t (int): stage of the oracle (i.e., oracle represents cost from stage t to T).
                    T (int): number of stages
                oracle_model_build outputs:
                    model (GRBModel): gurobi model
                    in_state (list): list with the state values of the previous stage.
                    rsh_vars (list): list of RHS place holders for the stage given as parameter 
        '''
        self.stage = stage
        self.model_stats = MathProgStats()
        model, in_states, rhs_vars = model_builder(stage, T)
        
        self.states_map = {}
        self.in_state = [x for x in in_states]
        self.gen_states_map(self.in_state)
        self.rhs_vars = [x for x in rhs_vars]
        
        self.model = model
        
        # Optimizer parameters
        self.model.params.OutputFlag = 0
        self.model.params.Threads = SDDP.options['grb_threads']
        #self.model.params.Method = 2
        
        #Construct dictionaries of (constraints,variables) key where duals are needed
        self.ctrsForDuals = set()
        self.ctrInStateMatrix = {}
        for vname in self.in_state:
            var = self.model.getVarByName(vname)
            col = self.model.getCol(var)
            for j in range(0, col.size()):
                ctr = col.getConstr(j)
                ctr_coeff = col.getCoeff(j)
                self.ctrsForDuals.add(ctr.ConstrName)
                self.ctrInStateMatrix[ctr.ConstrName, vname] = -ctr_coeff
        
        self.ctrRHSvName = {}
        for vname in self.rhs_vars:
            var = self.model.getVarByName(vname)
            col = self.model.getCol(var)
            # assert col.size() == 1, "RHS noise is not well defined"
            for c_index in range(col.size()):
                ctr = col.getConstr(c_index)
                assert ctr.ConstrName not in self.ctrRHSvName, "Duplicated RHS noise variable in the a single constraint."
                self.ctrRHSvName[ctr.ConstrName] = vname
                self.ctrsForDuals.add(ctr.ConstrName)
    
    def gen_states_map(self, in_states):
        for in_state in in_states:
            sindex = in_state.index('[')
            self.states_map[in_state] = in_state[:sindex - 1] + in_state[sindex:]
            self.states_map[in_state[:sindex - 1] + in_state[sindex:]] = in_state
    
    def get_out_state_var(self, in_state):
        '''
        Returns the corresponding out state variable name
        given a in state variable name of the next
        stage.
        '''
        return self.states_map[in_state]
    
    def solve(self,
              in_state_vals=None,
              random_realization=None,
              forwardpass=False,
              random_container=None,
              sample_path=None,
              num_cuts=0):
        '''
        Solves a stage problem given the state variables of the previous stage
        Args:
            in_state_vals (dict of real): dictionary containing the values of the state variables
                in the previous stage. They are referred as in_state for this model.
            random_realization (dic of real): dictionary containing the realization values to be solved.
                the key is the name of the variable that models the random value as a placeholder.
        
        Output:
            output (dict of objects): a dictionary with output information
                'status': Optimization model status
                'duals': dictionary of the dual variables associated to previous 
                        stage cuts.
                'out_state': Value of the state variables at the end of the sate
                        (input for the next stage).
                'cputime': float time solving the problem
                'lptipe': float value of the time solving the lp only
                'cutupdatetime' float value of the time updating the cuts.
        '''
        tnow = time()
        cutupdatetime = 0
        setuptime = 0
        if self.stage > 0:
            #if forwardpass:
            setuptime = time()
            assert len(in_state_vals) == len(self.in_state), "In state vector has different cardinality than expected"
            for in_s_name in in_state_vals:
                #sindex = in_s_name.index('[')
                #newkey = in_s_name[:sindex]+'0'+in_s_name[sindex:]
                self.model.getVarByName(self.states_map[in_s_name]).lb = in_state_vals[in_s_name]
                self.model.getVarByName(self.states_map[in_s_name]).ub = in_state_vals[in_s_name]
            
            #===================================================================
            # assert len(random_realization)==len(self.rhs_vars), "In random vector has different cardinality than expected %i" %(len(random_realization)-len(self.rhs_vars))
            # for rr in random_realization:
            #     self.model.getVarByName(rr).lb = random_realization[rr]
            #     self.model.getVarByName(rr).ub = random_realization[rr]
            #===================================================================
            #self.model.update()
            setuptime = time() - setuptime
            cutupdatetime = 0
        
        #Solve LP
        lp_time = time()
        self.model.optimize()
        lp_time = time() - lp_time
        
        data_mgt_time = time()
        output = {}
        status = gurobiStatusCodeToStr(self.model.status)
        output['status'] = status
        if status == SP_OPTIMAL or status == SP_UNKNOWN:
            output['objval'] = self.model.objVal
            output['duals'] = {cname: self.model.getConstrByName(cname).Pi for cname in self.ctrsForDuals}
            self.model_stats.add_simplex_iter_entr(self.model.IterCount)
        data_mgt_time = time() - data_mgt_time
        output['lptime'] = lp_time
        output['cutupdatetime'] = cutupdatetime
        output['setuptime'] = setuptime
        output['datamanagement'] = data_mgt_time
        output['cputime'] = time() - tnow
        return output


class MathProgStats:
    '''
    An object to keep track of some stats of the coresponding math program.
    
    Stats tracked:
        Mean number of iterations in simplex.
        Std of iteration in simplex.
    
    '''
    
    def __init__(self):
        self._simplex_iter_sum = 0.0
        self._simplex_iter_sumSq = 0.0
        self._simplex_iter_entries = 0
    
    def add_simplex_iter_entr(self, x):
        self._simplex_iter_entries += 1
        self._simplex_iter_sum += x
        self._simplex_iter_sumSq += (x**2)
    
    def get_mean(self):
        return self._simplex_iter_sum / self._simplex_iter_entries
    
    def get_sd(self):
        n = self._simplex_iter_entries
        numerator = self._simplex_iter_sumSq - (self._simplex_iter_sum**2) / n
        return np.sqrt(numerator / (n - 1))
