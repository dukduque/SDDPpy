'''
Created on Nov 17, 2017

@author: dduque
'''
import numpy as np
from gurobipy import quicksum, Model
from SDDP import ZERO_TOL, options, LAST_CUTS_SELECTOR, SLACK_BASED_CUT_SELECTOR
from abc import ABC, abstractmethod
from collections import defaultdict


class CutPool():
    def __init__(self, sp):
        self.sp = sp
        self.stage = sp.stage
        self.pool = {}
        self.pool_by_oracle = defaultdict(list)
        self.cut_selector = None
        self._requires_adjustment = False
        
        if options['cut_selector'] == LAST_CUTS_SELECTOR:
            self.cut_selector = LastCutsSelector()
        elif options['cut_selector'] == SLACK_BASED_CUT_SELECTOR:
            self.cut_selector = SlackBasedCutSelector()
        else:
            self.cut_selector = None
    
    def addCuts(self, model: Model, new_cuts: list):
        '''
        Add cuts to the pool manager that are non-redundant 
        
        Params:
            model(GRBModel): Model that contain the cuts
            new_cuts (list of cut): list with the cuts to be added.
        '''
        for new_cut in new_cuts:
            if new_cut.recomputable_rhs:
                self._requires_adjustment = True
            assert new_cut.name not in self.pool
            if new_cut.ctrRef is not None:
                self.pool[new_cut.name] = new_cut
                self.pool_by_oracle[new_cut.outcome].append(new_cut.name)
        
        if self.cut_selector is not None:
            #self.cut_selector.add_recent_cuts([(c.name, c.is_active) for c in new_cuts])
            self.cut_selector.select_cuts(self)
    
    def needs_update(self):
        return self._requires_adjustment
    
    def get_non_zero_duals(self):
        tup_ind = []
        duals = []
        for ctr_name in self.pool:
            opt_cut = self.pool[ctr_name]
            if np.abs(opt_cut.ctrRef.Pi) > ZERO_TOL:
                tup_ind.append((opt_cut.cut_id, opt_cut.outcome))
                duals.append(opt_cut.ctrRef.Pi)
        return tup_ind, duals
    
    def update_cut_pool_dro(self, sp, model, cuts_left):
        '''
            Clean a model leaving only cuts_left. This is only used
            when solving DRO models sequentially and hence the cuts
            for a value of r_1 are valid for r_2 if r_1 < r_2.
        '''
        if sp._last_stage:
            return
        n_outcomes = sp.n_outmes
        n_oracles = len(sp.oracle)
        n_cuts_target = cuts_left if sp.multicut else cuts_left * n_outcomes
        for i in range(n_oracles):
            self.pool_by_oracle[i].sort(key=lambda x: self.pool[x].slack, reverse=True)
            while len(self.pool_by_oracle[i]) > n_cuts_target:
                cut_name = self.pool_by_oracle[i].pop(0)
                cut = self.pool[cut_name]
                if cut.ctrRef is not None:
                    model.remove(cut.ctrRef)
                cut.ctrRef = None
                del self.pool[cut_name]
            for cut_name in self.pool_by_oracle[i]:
                cut = self.pool[cut_name]
                if cut.ctrRef is None:
                    self.ctrRef = model.addConstr(cut.lhs >= cut.rhs, cut.name)
        assert sum(len(self.pool_by_oracle[i]) for i in range(n_oracles)) == len(self.pool)
    
    def __len__(self):
        return len(self.pool)
    
    def __iter__(self):
        return (x for x in self.pool.values())


class Cut():
    '''
    Structure for a cut
    '''
    
    def __init__(self,
                 sp,
                 var_coeffs,
                 intercept,
                 cut_id,
                 stagewise_ind=True,
                 ind_rhs=None,
                 dep_rhs_vector=None,
                 outcome=0):
        '''
        Args:
            var_coeffs (dict of reals): coefficient of each variable involved in the cut where variable name is the key.
            vars (dict of GRBVar): dictionary of decision variables where the variable name is the key.
            
        '''
        m = sp.model
        stage = sp.stage
        self.name = 'cut[%i,%i,%i]' % (stage, cut_id, outcome)
        self.lhs = quicksum(-var_coeffs[vn] * m.getVarByName(vn) for vn in var_coeffs)
        self.lhs.add(sp.oracle[outcome])
        self.recomputable_rhs = (stagewise_ind == False)
        self.rhs = intercept
        self.ind_rhs = ind_rhs
        self.dep_rhs_vector = dep_rhs_vector
        self.ctrRef = None
        try:
            # Try to evaluate LHS, is not possible if new vars were added to the cut
            # Add constraint and get a reference to it.
            if self.lhs.getValue() >= self.rhs - ZERO_TOL:
                self.is_active = False
            else:
                self.is_active = True
                self.ctrRef = m.addConstr(self.lhs >= self.rhs, self.name)
        except AttributeError:
            # Add the cut to the model if lhs cannot be evaluated
            self.is_active = True
            self.ctrRef = m.addConstr(self.lhs >= self.rhs, self.name)
        
        #==============================#
        # Extra information for dual retrieval
        self.cut_id = cut_id
        self.outcome = outcome
    
    def adjust_intercept(self, omega_last):
        '''
        omega_last (1D ndarray): current ancestor scenario.
        '''
        dep_rhs = self.dep_rhs_vector.dot(omega_last).squeeze()
        new_rhs = dep_rhs + self.ind_rhs
        self.ctrRef.RHS = new_rhs
    
    @property
    def slack(self):
        try:
            return self.lhs.getValue() - self.rhs
        except AttributeError:
            return 0


class CutSelector(ABC):
    @abstractmethod
    def __init__(self):
        self.active = []
        self.unactive = []
    
    @abstractmethod
    def select_cuts(self, sp, cut_pool: CutPool):
        '''
        Makes a selection of the cuts to consider in the model.
        Updates local active and unactive list as well as the 
        status of the cut (cut.is_active and cut.ctrRef)
        '''
        raise 'Un-implemented method in a cut selection class.'
    
    def enforce_all(self, model, pool):
        for u in self.unactive:
            pool[u].ctrRef = model.addConstr(pool[u].lhs >= pool[u].rhs, pool[u].name)
            self.active.append(u)
        self.unactive.clear()
    
    def add_recent_cuts(self, c_info):
        for (c_name, c_is_active) in c_info:
            if c_is_active:
                self.active.append(c_name)
            else:
                self.unactive.append(c_name)


class LastCutsSelector(CutSelector):
    '''
    Cut selector based on the last added cuts. The number of cuts
    is set in the algorithm options.
    '''
    
    def __init__(self):
        super().__init__()
        self.cut_capacity = options['max_cuts_last_cuts_selector']
    
    def select_cuts(self, cut_pool: CutPool):
        '''
            Drop older cuts if the cut capacity is met. The capacity
            is assumed to be specified for multicut algorithm. For
            single-cut algorithm, this capacity will be scaled with
            the number of outcomes of the stage. To avoid constantly 
            updating the cut pool, this is only shinked to the capacity
            when the number of cuts is at 110%. 
            Args:
                cut_pool (CutPool): an instance of the cut pool for a
                 paticular stage problem.
        '''
        sp = cut_pool.sp
        n_total_cuts = self.cut_capacity * sp.n_outmes
        if len(cut_pool) > n_total_cuts * 1.1:
            n_cuts_target = self.cut_capacity if sp.multicut else n_total_cuts
            pool_by_oracle = cut_pool.pool_by_oracle
            for i in cut_pool.pool_by_oracle:
                while len(pool_by_oracle[i]) > n_cuts_target:
                    cut_name = pool_by_oracle[i].pop(0)
                    cut = cut_pool.pool[cut_name]
                    sp.model.remove(cut.ctrRef)
                    cut.ctrRef = None
                    del cut_pool.pool[cut_name]


class SlackBasedCutSelector(CutSelector):
    '''
    Cut selector based on binding cuts. At each iteration
    statistics on the cuts are updated to keep track of the
    number of times a cut is non-binding and is removed from
    the problem after a threshold is exceeded. 

    Attributes:
        active_stats (dict of (str,int)): count the number of times a cut 
            has been non-binding.

    '''
    
    def __init__(self):
        super().__init__()
        self.active_stats = {}
        self.cut_capacity = options['max_cuts_slack_based']
        self.track_ini = 0
        self.track_end = 0
    
    def select_cuts(self, model, pool, pool_order):
        
        #Update stats
        tracked = int(len(self.active))
        ini_changed = False
        for i in range(self.track_ini, tracked):
            a = self.active[i]
            if (a in self.active_stats) == False:
                self.active_stats[a] = 0
            else:
                #Count unactive times
                if pool[a].lhs.getValue() >= pool[a].rhs + options['slack_cut_selector']:
                    self.active_stats[a] += 1
                    if self.active_stats[a] >= options['slack_num_iters_cut_selector'] and ini_changed == False:
                        self.track_ini = i
                    else:
                        ini_changed = True
                else:
                    self.active_stats[a] = 0
        
        if len(self.active) <= self.cut_capacity or len(self.active) == 0:
            pass  #Not cut management requiered
        else:
            #Check active cuts (removing cuts from subproblems)
            hay_cut = len(self.active)
            new_active = []
            for i in range(tracked):
                a = self.active[i]
                if self.active_stats[a] >= options['slack_num_iters_cut_selector']:
                    pool[a].is_active = False
                    model.remove(pool[a].ctrRef)
                    pool[a].ctrRef = None
                    self.unactive.append(a)
                else:
                    new_active.append(a)
            new_active.extend(self.active[tracked:])
            self.active = new_active
            
            #Check unactive cuts (adding cuts)
            new_unactive = []
            for u in self.unactive:
                if pool[u].lhs.getValue() < pool[u].rhs:
                    self.active.append(u)
                    pool[u].ctrRef = model.addConstr(pool[u].lhs >= pool[u].rhs, pool[u].name)
                    pool[u].is_active = True
                    self.active_stats[u] = 0
                else:
                    new_unactive.append(u)
            self.unactive = new_unactive
            self.track_ini = 0
            #print(self.active[0], hay_cut, len(self.active))
            if len(self.active) > self.cut_capacity:
                self.cut_capacity = int(1.5 * self.cut_capacity)
                #options['max_cuts_slack_based'] = int(1.5*options['max_cuts_slack_based'])
                print('max_cuts_slack_based -- > ', self.cut_capacity, a)