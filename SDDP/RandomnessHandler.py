'''
Created on Nov 19, 2017

@author: dduque
'''
import copy
import numpy as np
import scipy.sparse as sp
from gurobipy import *
from SDDP.SDDP_utils import print_model
import SDDP as cs


#NOT USED FOR THE MOMENT
class SamplePath():
    '''
    classdocs
    '''
    
    def __init__(self, sampler_func):
        '''
        Constructor
        '''
        self.sampledvalues = sampler_func
    
    def getStageRealizations(self, stage):
        assert stage in self.sampledvalues, 'Sample path has not being created'
        return self.sampledvalues[stage]


class RandomContainer:
    '''
    Class to store and handle all random elements
    Attributes:
        stage_vectors (list of StageRandomVector): A list 
        with a random vector per stage. Vector elements had
        specified dependencies with vectors of the previous
         stages if it applies.
    '''
    
    def __init__(self):
        self.stage_vectors = []
    
    def append(self, stageRndVector):
        '''
        Adds a random vector associated with a stage
        '''
        self.stage_vectors.append(stageRndVector)
    
    def getSamplePath(self, rnd_stream, ev=False):
        '''
        Generates a sample path. This is, a list of dictionaries
        where each dictionary corresponds to a different stage and 
        each dictionary has the realized value for each random element
        with the key being the name of the random element and the value
        if a float with the numeric value of the realization.
        
        Args:
            ev (bool): If the sample path is deterministic for expected value 
                analysis (Default value is false).
        
        Return:
            partial_sample_path (list of dic): a sample path including all stages.
        '''
        partial_sample_path = []
        partial_sample_path_ids = []
        if ev == True:
            partial_sample_path = [srv.get_ev_vector(partial_sample_path) for srv in self.stage_vectors]
            partial_sample_path_ids = [0 for _ in self.stage_vectors]
        else:
            for srv in self.stage_vectors:
                stage_sample, stage_outcome_id = srv.getSample(partial_sample_path, rnd_stream)
                partial_sample_path.append(stage_sample)
                partial_sample_path_ids.append(stage_outcome_id)
        return partial_sample_path, partial_sample_path_ids
    
    def getStageSample(self, t, partial_sample_path, rnd_stream, new_support=None, new_pmf=None):
        '''
        Generates a sample for a given stage and appends it to a partial path
        **** Modifies the partial_sample_path given as a parameter ****
        Args:
            t (int): stage number to be sample.
            partial_sample_path (list of dict): a sample path up to stage t-1.
            rnd_stream (np.random.RandomStat): dedicated random stream
            new_support (list of dict): New support to sample from (if different from None)
            new_pmf (list of float): New pmf for the sampling.
        Return: 
            satage_sample (dict of (str-float)): A dictionary containing the
                realized value of each random element.
            stage_outcome_id (int): id of the outcome drew in the stage 
        '''
        if new_support == None:  #Sample w.r.t the original tree and given probs
            srv = self.stage_vectors[t]
            stage_sample, stage_outcome_id = srv.getSample(partial_sample_path, rnd_stream)
            partial_sample_path.append(stage_sample)
            return stage_sample, stage_outcome_id
        else:
            #Ignores the tree and use the provided supprt points
            assert len(new_support) == len(new_pmf), 'Inconsistent distribution to sample from.'
            assert self.stage_vectors[t].is_independent, 'Stage-wise independence is required for variant trees.'
            lucky_outcome = rnd_stream.choice([i for i in range(len(new_support))], p=new_pmf)
            stage_sample = new_support[lucky_outcome]
            partial_sample_path.append(stage_sample)
            return stage_sample, np.nan
    
    def _preprocess_randomness(self):
        for sv in self.stage_vectors:
            sv._preproces_randomness()
    
    def _set_outcomes_for_run(self, ev):
        '''
        Sets the type of outcomes according to the problem type:
            expected value problem (single outcome per random variables)
            stochastic problem     (multiple outcomes per random variable)
        '''
        for sv in self.stage_vectors:
            if ev:
                sv.set_ev_problem()
            else:
                sv.set_stochastic_problem()
    
    def enumerate_scenarios(self, ):
        all_scenarios = []
        self.recursive_enum(all_scenarios, 0, [])
        return all_scenarios
    
    def recursive_enum(self, all_scenarios, stage, past):
        sv = self.stage_vectors[stage]
        for outcome in sv.outcomes:
            past.append(outcome)
            if stage < len(self.stage_vectors) - 1:
                self.recursive_enum(all_scenarios, stage + 1, past)
            else:
                all_scenarios.append(copy.deepcopy(past))
            past.pop()
    
    def reset_to_nominal_dist(self):
        for srv in self.stage_vectors:
            srv.reset_to_nominal_dist()
    
    #Aux methods to make the container iterable
    def __iter__(self):
        return (x for x in self.stage_vectors)
    
    def __getitem__(self, index):
        return self.stage_vectors[index]
    
    def __setitem__(self, key, value):
        self.stage_vectors[key] = value
    
    def __delitem__(self, key):
        self.stage_vectors.__delitem__(key)
    
    def __repr__(self):
        return [x for x in self.stage_vectors].__repr__()
    
    @property
    def support_dimension(self):
        assert len(self.stage_vectors[0].elements) == len(self.stage_vectors[0].elements), \
            "Support dimension is not supposed to change in dimension over time."
        return len(self.stage_vectors[0].elements)
    
    def get_noise_ub(self, noise_names, flip_sign=False):
        bounds = []
        for noise_term in noise_names:
            bounds.append(((-1)**flip_sign) * np.max([sv.get_ub(noise_term) for sv in self.stage_vectors]))
        return bounds
    
    def get_noise_lb(self, noise_names, flip_sign=False):
        bounds = []
        for noise_term in noise_names:
            bounds.append(((-1)**flip_sign) * np.min([sv.get_lb(noise_term) for sv in self.stage_vectors]))
        return bounds


class StageRandomVector:
    '''
    A class to represent the randomness of a stage
    Attributes:
        stage (int): Number of the corresponding stage.
        elements (dict (str,RandomElement)): Dictionary storing each random
            element of the vector by its name as a key.
        vector order (dict (str,int)): dictionary to map the names with a fixed
            order of the elements in the vector.
        outcomes (list of dict): a list of possible outcomes of the random vector.
            Each element of the list is a dictionary containing the numerical values
            for all the random elements of the vector. In the interstage dependent case,
            this values are the independent part only.
        p (ndarray): vector with the probabilities of each outcome. This vector is modifiable
            depending on the risk measure.
        is_indipendent (bool): flag to distinguish between independent and dependent cases. 
    '''
    
    def __init__(self, stage):
        self.stage = stage
        self.elements = {}
        self.vector_order = {}
        self.outcomes = []
        self.ev_outcomes = []
        self.outcomes_dim = 0
        self.p = None
        self.p_copy = None
        self.is_independent = True
        self._first_element_added = False
        self._is_expected_value_problem = False
    
    def addRandomElement(self, ele_name, ele_outcomes, ele_prob=None):
        if self._first_element_added == False:
            self.outcomes_dim = len(ele_outcomes)
            for e in ele_outcomes:
                self.outcomes.append({})
            self.ev_outcomes = {}
            self._first_element_added = True
            if type(ele_prob) == list or type(ele_prob) == np.ndarray:
                self.p = np.array(ele_prob)
            else:
                self.p = np.array([1 / len(ele_outcomes) for x in ele_outcomes])
            assert np.abs(1 - sum(self.p)) < cs.ZERO_TOL, "Invalid outcome probabilities"
            self.p_copy = self.p.copy()
        else:
            assert len(self.outcomes) == len(ele_outcomes), "Random element with a different number of outcomes."
        
        self.vector_order[ele_name] = len(self.elements)  #Defines order as it gets built.
        self.elements[ele_name] = RandomElement(self.stage, ele_name, ele_outcomes)
        self.ev_outcomes[ele_name] = np.mean(ele_outcomes)
        for (i, e) in enumerate(ele_outcomes):
            self.outcomes[i][ele_name] = e
        
        return self.elements[ele_name]
    
    def set_ev_problem(self):
        self.outcomes = [self.ev_outcomes]
        self._is_expected_value_problem = True
    
    def set_stochastic_problem(self):
        if self._is_expected_value_problem:
            self.outcomes = self.outcomes_copy
    
    def modifyOutcomesProbabilities(self, newp):
        assert len(newp) == len(self.outcomes)
        self.p = np.array(newp)
    
    def getSample(self, partial_sample_path, rnd_gen):
        '''
        Generates a sample for the associate stage random vector. 
        It receives the partial sample path in case the random vector
        is stagewise dependent.
        Attributes:
            partial_sample_path (list of dict): A list of the realizations
                of the previous stages. The dictionary has the same form as
                the method output.
        Return:
            satage_sample (dict of (str-float)): A dictionary containing the
                realized value of each random element.
            lucky_outcome (int): id of the outcome that was realized.
        '''
        try:
            lucky_outcome = rnd_gen.choice([i for i in range(0, self.outcomes_dim)], p=self.p)
            stage_sample = {
                e.name: e.getSample(lucky_outcome, e.get_depedencies_realization(partial_sample_path))
                for e in self.elements.values()
            }
            return stage_sample, lucky_outcome
        except:
            print(self.p)
    
    def get_ev_vector(self, partial_sample_path):
        return {
            e.name: e.comput_element_ev(e.get_depedencies_realization(partial_sample_path))
            for e in self.elements.values()
        }
    
    def getOutcomes(self, sample_path, ev):
        if self.is_independent:
            if ev == True:
                return [self.ev_outcomes]
            return self.outcomes
        else:
            if ev == True:
                raise 'Expected value mode not implemented for dependent case.'
            outcomes_copy = copy.deepcopy(self.outcomes)
            for e in self.elements.values():
                e_dependencies = e.get_depedencies_realization(sample_path)
                new_e_outcomes = e.compute_outcomes(e_dependencies)
                for i in range(0, len(new_e_outcomes)):
                    outcomes_copy[i][e.name] = new_e_outcomes[i]
            
            return outcomes_copy
    
    def get_sorted_outcome(self, outcome_index):
        '''
        Retrieve a specific outcome and returned as a
        vector according the the vector_order mapping.
        Args:
            outcome_index (int): index of the outcome of interest
        Return:
            xi (ndarray): outcome in vector form
        '''
        xi = np.zeros(len(self.elements))
        for ele in self.vector_order:
            xi[self.vector_order[ele]] = self.outcomes[outcome_index][ele]
        return xi
    
    def _preproces_randomness(self):
        '''
        Creates a copy of the input random vector and process autoregressive matrices
        '''
        self.outcomes_copy = copy.deepcopy(self.outcomes)
        vec_dim = len(self.elements)
        R_matrices = []  #Auto reg matrices
        created = False
        for e in self.elements.values():
            abs_order_e = self.vector_order[e.name]
            if e.has_dependencies():
                if created == False:
                    R_matrices = [sp.coo_matrix((vec_dim, vec_dim), dtype=np.float64) for de in e.dependencies]
                    created = True
                
                for (i, d_stage) in enumerate(e.dependencies):
                    #assert len(self.elements)==len(e.dependencies[d_stage])
                    for dep_e in e.dependencies[d_stage]:
                        abs_order_d = self.vector_order[dep_e]
                        #R_matrices[i][abs_order_e,abs_order_d] = e.dependencies[d_stage][dep_e]
                        R_matrices[i] = R_matrices[i] + sp.coo_matrix(
                            ([e.dependencies[d_stage][dep_e]], ([abs_order_e], [abs_order_d])),
                            shape=(vec_dim, vec_dim))
                
                self.is_independent = False
                self.lag = e.lag
        self.autoreg_matrices = R_matrices
    
    def reset_to_nominal_dist(self):
        self.p = self.p_copy.copy()
    
    def get_ub(self, ele_name):
        return np.max(self.elements[ele_name].outcomes)
    
    def get_lb(self, ele_name):
        return np.min(self.elements[ele_name].outcomes)
    
    def __repr__(self):
        return 't=%i %s' % (self.stage, self.elements.keys().__repr__())


class RandomElement:
    '''
    Class to represent a random element
    Attributes:
        stage (int): Stage in which the random element realizes. 
        name (str): Name of the random element.
        outcomes (1D - ndarray): possible outcomes of the random element.
        current_sample_path(list of StageRandon)
        
        dependencies (dict of (int, dict of (str-float)): dependencies stored by 
            stage number (key of the outer dictionary) and the independent random
            element names (str key of the by random element). The value is the
            multiplicative coefficient.
    '''
    
    def __init__(self, stage, name, rnd_outcomes):
        self.stage = stage
        self.name = name
        self.outcomes = np.array(rnd_outcomes)
        self.dependencyFunction = None
        self.dependencies = {}
        self.current_sample_path = None
    
    def __repr__(self):
        return 't=%i>%s' % (self.stage, self.name)
    
    def has_dependencies(self):
        return True if len(self.dependencies) > 0 else False
    
    def addDependecyFunction(self, dependency_coefs, depFunc):
        '''
        Specify a function for a dependent model.
        Args:
            dependency_coefs (dict of (int, dict of (str-float)): dependency coefficients
                stored by stage number (key of the outer dictionary) and the independent 
                random element names (str key of the by random element). The value 
                is the multiplicative coefficient.
            depFunc (::func::) a function that specifies how the dependency is computed. The
                    function receives to arguments:
                    rnd_element (RandomElement): the dependent random element.
                    realizations (dict): a dictionary of the realization with the same structure
                                         as the element dependencies dictionary.
                    The function returns a 1D-ndarry.
                    
        '''
        self.dependencyFunction = depFunc
        self.dependencies = dependency_coefs
        min_stage_dep = min(x for x in self.dependencies)
        self.lag = self.stage - min_stage_dep
    
    def compute_outcomes(self, depedencies_realizations):
        if self.dependencyFunction == None:
            return self.outcomes
        else:
            new_outcomes = self.dependencyFunction(self, depedencies_realizations)
            assert isinstance(new_outcomes, np.ndarray) and len(self.outcomes) == len(new_outcomes), \
                    "Dependency function returned outcomes that don't match the necessary dimensions (%i)" %(len(new_outcomes))
            return new_outcomes
    
    def comput_element_ev(self, depedencies_realizations):
        if self.dependencyFunction == None:
            return np.mean(self.outcomes)
        else:
            new_outcomes = self.dependencyFunction(self, depedencies_realizations)
            return np.mean(new_outcomes)
    
    def get_depedencies_realization(self, partial_sample_path):
        '''
        Gets the necessary values for the element dependencies
        given a partial path up to stage = self.stage - 1.
        '''
        my_depedencies = {}
        for (stage, realization) in enumerate(partial_sample_path):
            if stage in self.dependencies:
                my_depedencies[stage] = {}
                for dname in self.dependencies[stage]:
                    my_depedencies[stage][dname] = realization[dname]
        return my_depedencies
    
    def getSample(self, p, dependencies):
        '''
        Generates a sample of the random element give a probability vector
        and a list of dependencies.
        Attributes:
            p (1D ndarry): Array of probabilities for each outcome,
            
            dependencies (dict of dict): a dictionary of dependencies.
                The key of the outer dictionary is the stage corresponding
                to the dependency and the the value is another dictionary.
                The key of the inner dictionary is the name of the random
                element and the value is its numerical realization.
                e.g.: For a random element in stage 3
                dependencies={2:{'b_1':20, 'b_2':40},
                              1:{'b_1':5, 'b_2':3}}
                    So this random element depends on random elements
                    of stages 1 and 2.      
        '''
        generalized_outcomes = self.compute_outcomes(dependencies)
        return generalized_outcomes[p]


class ScenarioTree:
    '''
    Class to represent a subtree
    Attributes:
        root_node (ScenarioTreeNode): root node in a scenario subtree
    '''
    
    def __init__(self, rnd_cont):
        #Creates the root nod of the scenario tree
        self.root_node = ScenarioTreeNode(0, 0, None, True)
        self.rnd_cont = rnd_cont
        self.sample_path_costs = {}
        self.sample_path_outs = {}
    
    def add_sample_path(self, sample_id, sample_path_outcomes, sample_path_cost):
        '''
        Add a sample path in the tree. The descendants of ever node in the sample path
        are created right away regardless of whether that is a corresponding sample path or not.
        '''
        
        self.sample_path_costs[sample_id] = sample_path_cost
        self.sample_path_outs[sample_id] = sample_path_outcomes
        node_t = self.root_node
        for t in range(len(sample_path_outcomes)):
            node_t.add_sample_path_info(sample_id)
            if t < len(sample_path_outcomes) - 1:
                if len(node_t.children) == 0:
                    node_t.create_descendants(self.rnd_cont.stage_vectors[t + 1])
                
                #update next node in the tree
                next_outcome = sample_path_outcomes[t + 1]
                node_t = node_t.children[next_outcome]
    
    def compute_subtree_upper_bound(self, risk_measures):
        '''
        Args:
            risk_measures (list of RiskMeasure): list with the risk measure for each stage.
        '''
        
        m = Model('SubTree_UB')
        m.params.OutputFlag = 0
        n_paths = len(self.sample_path_costs)
        
        #Variables creation
        p = m.addVars(n_paths, lb=0, ub=1, obj=self.sample_path_costs, vtype=GRB.CONTINUOUS, name='p')
        self.root_node.create_phi_var(m, '')
        m.update()
        #Constraints
        m.addConstr(lhs=p.sum(), sense=GRB.EQUAL, rhs=1, name='global_prob')
        self.root_node.create_constraints(m, p, risk_measures, self.rnd_cont, '')
        m.ModelSense = GRB.MAXIMIZE
        m.update()
        #print_model(m)
        m.optimize()
        if m.status != GRB.OPTIMAL:
            m.computeIIS()
            m.write("model.ilp")
        else:
            print('%15.5e' % (m.Objval))
            for (i, v) in enumerate(m.getVars()):
                if v.VarName[0] == 'p' and v.X > 0 and v.VarName[0:3] != 'phi':
                    print(v)
                    if v.VarName[0] == 'p' and v.VarName[0:3] != 'phi':
                        print(self.sample_path_costs[i])
                        print(self.sample_path_outs[i])
            
            return m.Objval


class ScenarioTreeNode:
    '''
    A node of a scenario tree
    Args:
        parent (ScenarioTreeNode): parent node in the tree
        stage (int): Stage to which the node belongs
        children (list of ScenarioTreeNodes): list of ScenarioTreeNodes that 
            are descendants of this node.
        sampled (boolean): If there is a sample path that uses these node (default is false)
        
        phi_var (GRBVar): Decision variable for the upper bound model.
    '''
    
    def __init__(self, stage, outcome_id, parent, sampled):
        self.parent = parent
        self.outcome_id = outcome_id
        self.stage = stage
        self.children = []
        self.sampled = sampled
        self.sample_path_ids = []
        self.phi_var = None
    
    def add_sample_path_info(self, sample_id):
        '''
        Updates the node status to sampled and
        add the sample path id to the list of paths that
        visit the node.
        '''
        self.sample_path_ids.append(sample_id)
        self.sampled = True
    
    def create_descendants(self, stage_random_vector):
        '''
        Creates a descendant node for every outcome of the stage random 
        vector given as parameter.
        Args:
            stage_random_vector (StageRandomVector): object with the random elements of the next stage
        '''
        for o in range(stage_random_vector.outcomes_dim):
            new_child = ScenarioTreeNode(self.stage + 1, o, self, False)
            self.children.append(new_child)
    
    def create_phi_var(self, model, branch_name):
        '''
        Recursively creates all the variables in the model. 
        Creates a variable for the node and call the method for all 
        the descendant nodes.
        Args:
            model (GRBModel): Upper bound model
            branch_name (str): String identifier for a branch of the tree
        '''
        new_branch_name = branch_name + '_%i' % (self.outcome_id)
        self.phi_var = model.addVar(lb=0,
                                    ub=1,
                                    obj=0,
                                    vtype=GRB.CONTINUOUS,
                                    name='phi[%i,%s]' % (self.stage, new_branch_name))
        
        for node in self.children:
            node.create_phi_var(model, new_branch_name)
    
    def create_constraints(self, model, p, risk_measures, rc, branch_name):
        '''
        Recursively creates the constraint of the upper bound model.
        
        Args:
            model (GRBModel): Upper bound model
            p (tupledic of GRBVar): variables representing the probability of each sample path
            risk_measures (list of AbstracRiskMeasures): Risk measure for each stage
            rc (RandomContainer): Object with al the random information
            branch_name (str): String identifier for a branch of the tree
        '''
        t = self.stage
        new_branch_name = branch_name + '_%i' % (self.outcome_id)
        #Phi - p relation
        if self.sampled:
            model.addConstr(lhs=self.phi_var,
                            sense=GRB.EQUAL,
                            rhs=quicksum(p[i] for i in self.sample_path_ids),
                            name='PhiToP_%i%s' % (t, new_branch_name))
            
            if len(self.children) > 0:
                #prob consistency: Sum of descendants adds up to 1
                #model.addConstr(quicksum(n.phi_var for n in self.children), sense=GRB.EQUAL,  rhs=1, name='cox_%i%s' %(t,new_branch_name))
                #model.update()
                #Uncertainty set
                phis = [n.phi_var for n in self.children]
                risk_measures[t].define_scenario_tree_uncertainty_set(self.stage, self.outcome_id, model, rc[t + 1],
                                                                      phis, new_branch_name)
                
                for node in self.children:
                    node.create_constraints(model, p, risk_measures, rc, new_branch_name)


def AR1_depedency(rnd_ele, realizations):
    assert type(realizations) == dict, 'Realizations are not in the expected form.'
    assert len(realizations) == 1, "Dependency model has a lag > 1."
    AR1model = sum(realizations[k][j] * rnd_ele.dependencies[k][j] for k in realizations for j in realizations[k])
    return AR1model + rnd_ele.outcomes
