'''
Created on Nov 17, 2017

Module that contains the main logic of SDDP algorithm.

@author: Daniel Duque
'''
import logging
import time

import numpy as np
import SDDP as cs
from SDDP.MathProgs import StageProblem, not_optimal_sp,\
    StageOracleProblem

from SDDP.RandomManager import alg_rnd_gen, in_sample_gen, out_sample_gen, reset_all_rnd_gen, reset_out_sample_gen

from OutputAnalysis.SimulationAnalysis import SimResult
from SDDP.RandomnessHandler import ScenarioTree
from SDDP.RiskMeasures import Expectation
from SDDP.WassersteinWorstCase import solve_worst_case_expectation
from SDDP import check_options_concitency
from Utils.timer_utils import Chronometer

sddp_log = cs.logger

alg_options = cs.alg_options()
iteration_log = ''


class SDDP(object):
    '''
    Implementation of Stochastic Dual Dynamic Programming algorithm.
    '''
    
    def __init__(self,
                 T,
                 model_builder,
                 random_builder,
                 lower_bound=-1E10,
                 risk_measure=Expectation,
                 **risk_measure_params):
        '''
        Constructor
        '''
        self.stats = Stats()
        self.stage_problems = []
        
        # Setup stochastics of the model
        self.random_container = random_builder()
        self.random_container._preprocess_randomness()
        
        # Setup math models
        # In/Out states variables mapping
        self.global_states_mapping = {}
        self.createStageProblems(T, model_builder, lower_bound, risk_measure, **risk_measure_params)
        
        # Save instance parameters
        self.instance = {
            'risk_measure': risk_measure,
            'risk_measure_params': risk_measure_params,
            'alg_options': alg_options
        }
        
        self.lb = None
        self.ub = float('inf')
        self.ub_hw = 0
        self.upper_bounds = []
        self.pass_iteration = 0
        self.num_cuts = [0] * T
        
        #Attribute to keep track of the maximum violation when using cutting planes.
        self.cutting_plane_max_vio = None
        self.best_p = None  #Used in the opt-simulation method
        
        #Attributes to store oracle subproblems
        self.stage_oracle_subproblems = []
        
        #Save builders
        self._model_builder = model_builder
        self._random_builder = random_builder
    
    @classmethod
    def create_SDDP(cls, t_ini, T_max, sddp_alg):
        '''
        Create an instance of SDDP based on a previews sddp_alg
        '''
        print(cls)
    
    def createStageProblems(self, T, model_builder, lower_bound, risk_measure, **risk_measure_params):
        '''
        Creates all subproblems given a builder.
        Args:
            T (int): Number of stages.
            model_builder (::func::) a function the return a math model.
            in_states (list of str): list with the names of the variables that 
                represent the previous state.
            out_states (list of str): list with the names of the variables that 
                represent the next state.
        '''
        
        for i in range(T):
            sp_risk_measure = risk_measure(**risk_measure_params)
            next_stage_rnd_vector = self.random_container[i + 1] if i < T - 1 else None
            sp_t = StageProblem(i,
                                model_builder,
                                next_stage_rnd_vector,
                                lower_bound,
                                last_stage=(i == T - 1),
                                risk_measure=sp_risk_measure,
                                multicut=alg_options['multicut'])
            self.stage_problems.append(sp_t)
            
            if i > 0:
                sp_t_1 = self.stage_problems[i - 1]
                for out_name in sp_t_1.out_state:
                    self.global_states_mapping[(i - 1, i), out_name] = sp_t_1.states_map[out_name]
                for in_name in sp_t.in_state:
                    self.global_states_mapping[(i - 1, i), in_name] = sp_t.states_map[in_name]
        
        #Setup RHS noise of the stage problem 0
        rnd_vec_0 = self.random_container[0]
        stage_problem_0 = self.stage_problems[0]
        for vn in stage_problem_0.rhs_vars_var:
            v = stage_problem_0.rhs_vars_var[vn]
            v.lb = rnd_vec_0.outcomes[0][vn]
            v.ub = rnd_vec_0.outcomes[0][vn]
        
        #Pre-compute initial bounds on the oracles
        # TODO: Oracle bounds are valid but they seem to hinder the algorithm performance.
        #self.compute_oracle_bounds()
    
    def add_oracle_model(self, oracle_model_builder):
        '''
        Adds a handle on each subproblem to a model that serves as an oracle. 
        oracle_model_builder (func): Function that builds an auxiliary problem that serves as 
            an oracle of the value function. The signature of the oracle builder is:
                oracle_model_builder params:
                    t (int): stage of the oracle (i.e., oracle represents cost from stage t to T).
                    T (int): number of stages
        '''
        T = len(self.stage_problems)
        for i in range(T):
            if i <= alg_options['max_stage_with_oracle']:
                osp = StageOracleProblem(i, T, oracle_model_builder)
                self.stage_oracle_subproblems.append(osp)
    
    def compute_oracle_bounds(self):
        '''
            Computes lower bound for the oracle(s)  
        '''
        T = len(self.stage_problems)
        for t in range(T - 1, 0, -1):
            sp_t = self.stage_problems[t]
            sp_t_1 = self.stage_problems[t - 1]
            stage_rnd_vector = self.random_container[t]
            omega_t = stage_rnd_vector.getOutcomes(sample_path=None, ev=False)
            in_state_lb = {in_name: self.global_states_mapping[(t - 1, t), in_name].lb for in_name in sp_t.in_state}
            in_state_ub = {in_name: self.global_states_mapping[(t - 1, t), in_name].ub for in_name in sp_t.in_state}
            oracle_lbs = []
            for outcome in omega_t:
                sp_output = sp_t.lower_bounding_solve(in_state_lb, in_state_ub, random_realization=outcome)
                oracle_lbs.append(sp_output['objval'])
            
            if alg_options['multicut'] == True:
                assert len(oracle_lbs) == len(sp_t_1.oracle), 'Number of bounds computed does not match.'
                for (i, lb_i) in enumerate(oracle_lbs):
                    sp_t_1.lower_bounds[i] = lb_i
                    sp_t_1.oracle[i].lb = lb_i
            else:
                assert len(sp_t_1.oracle) == 1, 'More oracle variables than expected.'
                min_bound = np.min(oracle_lbs)
                sp_t_1.lower_bounds[0] = min_bound
                sp_t_1.oracle[0].lb = min_bound
            sp_t_1.model.update()
    
    def forwardpass(self, sample_path, simulation=False):
        '''
        Runs a forward pass given a sample path. If no sample pass is given,
        a dynamic version of the forward pass method is invoked.
        '''
        if sample_path is None or len(sample_path) == 0:  # No sample path was given
            return self.dynamic_forwardpass(sample_path, simulation)
        
        fp_out_states = []
        fp_ub_value = 0
        for (i, sp) in enumerate(self.stage_problems):
            in_state = fp_out_states[-1] if i > 0 else None
            sp_output = sp.solve(in_state_vals=in_state,
                                 random_realization=sample_path[i],
                                 forwardpass=True,
                                 random_container=self.random_container,
                                 sample_path=sample_path,
                                 num_cuts=self.num_cuts[i])
            
            if sp_output['status'] == cs.SP_INFEASIBLE:
                self.debrief_infeasible_sub(sample_path, i, sp_output, sp)
            
            fp_out_states.append(sp_output['out_state'])
            '''
            IO and stats updates
            '''
            if simulation and alg_options['outputlevel'] >= 3:
                sp.print_stage_res_summary()
            if i == 0:
                self.lb = sp_output['objval']
            if simulation:
                fp_ub_value += sp.get_stage_objective_value()
            self.stats.updateStats(cs.FORWARD_PASS,
                                   lp_time=sp_output['lptime'],
                                   cut_update_time=sp_output['cutupdatetime'],
                                   model_update_time=sp_output['setuptime'],
                                   data_out_time=sp_output['datamanagement'],
                                   num_lp_ctrs=sp.model.num_constrs,
                                   iteration=self.pass_iteration)
            if sp_output['risk_measure_info'] != None and i == 0:
                self.cutting_plane_max_vio = sp_output['risk_measure_info']
        if simulation:
            self.upper_bounds.append(fp_ub_value)
        if simulation and alg_options['outputlevel'] >= 3:
            print('---------------------------')
        return fp_out_states
    
    def dynamic_forwardpass(self, sample_path, simulation=False):
        '''
        Runs a forward pass in which the sample path is constructed
        as as the forward pass progresses in time.
        '''
        fp_out_states = []
        fp_ub_value = 0
        new_support = None
        new_pmf = None
        for (i, sp) in enumerate(self.stage_problems):
            in_state = fp_out_states[-1] if i > 0 else None
            #new_support = self.random_container[i].outcomes
            #new_pmf = [1] if i ==0 else [0.1,0.2,0.4,0.3]
            self.random_container.getStageSample(i, sample_path, alg_rnd_gen, new_support=new_support, new_pmf=new_pmf)
            sp_output = sp.solve(in_state_vals=in_state,
                                 random_realization=sample_path[i],
                                 forwardpass=True,
                                 random_container=self.random_container,
                                 sample_path=sample_path,
                                 num_cuts=self.num_cuts[i])
            
            if sp_output['status'] == cs.SP_INFEASIBLE:
                self.debrief_infeasible_sub(sample_path, i, sp_output, sp)
            
            fp_out_states.append(sp_output['out_state'])
            next_sp = self.stage_problems[i + 1] if i + 1 < len(self.stage_problems) else None
            sp.risk_measure.forward_prob_update(i, sp, next_sp, fp_out_states, sample_path, self.random_container)
            try:
                new_support, new_pmf = sp.risk_measure.forward_prob_update_WassCont(i, sp, self.random_container)
            except:
                pass
            '''
            IO and stats updates
            '''
            if not simulation and alg_options['outputlevel'] >= 3:
                pass  #  sp.print_stage_res_summary()
            if i == 0:
                self.lb = sp_output['objval']
            fp_ub_value += sp.get_stage_objective_value()
            self.stats.updateStats(cs.FORWARD_PASS,
                                   lp_time=sp_output['lptime'],
                                   cut_update_time=sp_output['cutupdatetime'],
                                   model_update_time=sp_output['setuptime'],
                                   data_out_time=sp_output['datamanagement'],
                                   num_lp_ctrs=sp.model.num_constrs,
                                   iteration=self.pass_iteration)
            if sp_output['risk_measure_info'] is not None and i == 0:
                self.cutting_plane_max_vio = sp_output['risk_measure_info']
        
        self.upper_bounds.append(fp_ub_value)
        if simulation and alg_options['outputlevel'] >= 3:
            sp.print_stage_res_summary()
        return fp_out_states
    
    def backwardpass(self, forward_out_states=None, sample_path=None, ev=False):
        '''
        Runs a backward pass given a sample path and the forward pass decision
        associated to the sample path.
        
        Args:
            forward_out_states (list of dict): List of the out states for each stage.
            The out states are the solutions of decision variables in the problem and
            are stored as a dictionary where the key is the variable name.
            
            sample_path (list of outcomes): List of the sample path being solved
            
            ev (bool): If expected value policy is being computed (Default is False).
        '''
        T = len(self.stage_problems)
        
        for t in range(T - 1, 0, -1):
            outputs_per_outcome = []
            sp = self.stage_problems[t]
            stage_rnd_vector = self.random_container[t]
            omega_t = stage_rnd_vector.getOutcomes(sample_path, ev)
            for outcome in omega_t:
                sp_output = sp.solve(in_state_vals=forward_out_states[t - 1],
                                     random_realization=outcome,
                                     forwardpass=False,
                                     random_container=self.random_container,
                                     sample_path=sample_path)
                
                if sp_output['status'] == cs.SP_INFEASIBLE:
                    self.debrief_infeasible_sub(sample_path, t, sp_output, sp)
                
                #sp.printPostSolutionInformation()
                outputs_per_outcome.append(sp_output)
                self.stats.updateStats(cs.BACKWARD_PASS,
                                       lp_time=sp_output['lptime'],
                                       cut_update_time=sp_output['cutupdatetime'],
                                       model_update_time=sp_output['setuptime'],
                                       data_out_time=sp_output['datamanagement'],
                                       num_lp_ctrs=sp.model.num_constrs,
                                       iteration=self.pass_iteration)
            sp_cut = self.stage_problems[t - 1]
            cut_creation_time = self.createStageCut(t - 1, sp_cut, sp, stage_rnd_vector, outputs_per_outcome,
                                                    forward_out_states[t - 1], sample_path)
            self.stats.updateStats(cs.BACKWARD_PASS, cut_gen_time=cut_creation_time)
            
            #DELLETE OR FIX LATER
            try:
                worst_case_dist = stage_rnd_vector.worst_case_dist
                for (i, outcome) in worst_case_dist['support']:
                    sp_output = sp.solve(in_state_vals=forward_out_states[t - 1],
                                         random_realization=outcome,
                                         forwardpass=False,
                                         random_container=self.random_container,
                                         sample_path=sample_path)
                    outputs_per_outcome2 = [sp_output.copy() for _ in range(len(omega_t))]
                    for (i, o) in enumerate(omega_t):
                        #Intercept adjustment. Remove the perturbed part (pi*b_wc) and adds the original dual obj intercept rhs
                        outputs_per_outcome2[i]['objval'] = outputs_per_outcome2[i]['objval'] - outputs_per_outcome2[i][
                            'dual_obj_rhs_noice'] + sum(o[sp.ctrRHSvName[ctr_name]] * sp_output['duals'][ctr_name]
                                                        for ctr_name in sp.ctrRHSvName)
                    cut_creation_time = self.createStageCut(t - 1, stage_rnd_vector, outputs_per_outcome2,
                                                            forward_out_states[t - 1], sample_path)
                    self.num_cuts[t] += 1
            except:
                pass
            # END EDITS
            #del(outputs_per_outcome)
        
        #pool0 = self.stage_problems[0].cut_pool
        #print('% active cuts in 0: ',  (len(pool0.cut_selector.active)/len(pool0.pool)), ' of ' , len(pool0.pool))
    
    def createStageCut(self, stage, sp_t, sp_t1, stage_rnd_vector, outputs_per_outcome, forward_out_states,
                       sample_path):
        '''
        Calls the cut routine in the stage given as a parameter.
        Args:
            stage (int): Stage where the cut will be added
            stage_rnd_vector (StageRandomVector): object with the stochastic 
                representation of the next stage.
            outputs_per_outcome (list of dict): a list with the outputs of the
                of the next stage (one per scenario of the current pass).
            omega (list of dict): a list of scenarios of the current sample 
                path for the next stage. Each scenario is represented as a 
                dictionary where the key is the random element name and the
                value is the the realization in each scenario.
            forward_out_states 
            sample_path (list of dict): current sample path. Each element of the
                list corresponds to the realizations of a different stage.
            
        '''
        cut_creation_time = time.time()
        if stage < 0:
            return
        sp_t.createStageCut(self.num_cuts[stage], sp_t1, stage_rnd_vector, outputs_per_outcome, forward_out_states,
                            sample_path)
        self.num_cuts[stage] += 1
        return time.time() - cut_creation_time
    
    def init_out(self, instance_name):
        sddp_log.info(instance_name)
        sddp_log.info('Multicut: %s' % (str(alg_options['multicut'])))
        sddp_log.info('Dynamic Sampling: %s' % (str(alg_options['dynamic_sampling'])))
        
        if alg_options['outputlevel'] >= 2:
            sddp_log.info(
                '==============================================================================================')
            sddp_log.info('%4s %15s %15s %15s %12s %15s' % ('Pass', 'LB', 'iUB', 'iHW', 'Wall time', 'Other'))
            sddp_log.info(
                '==============================================================================================')
    
    def get_wall_time(self):
        '''
        Return the wall time since the begining of the algorithm
        '''
        return self.alg_chrono.elapsed()
    
    def iteration_update(self, fp_time, bp_time, force_print=False, last_iter=False):
        if (alg_options['outputlevel'] >= 2
                and (self.pass_iteration % alg_options['lines_freq'] == 0 or last_iter == True)
                and force_print == False):
            elapsed_time = self.get_wall_time()
            additional_msg = ''
            if self.cutting_plane_max_vio != None:
                additional_msg = '%15.5e' % (self.cutting_plane_max_vio)
            sddp_log.info('%4i %15.8e %15.8e %15.5e %15.2f %15s' %
                          (self.pass_iteration, self.lb, self.ub, self.ub_hw, elapsed_time, additional_msg))
        if force_print:
            sddp_log.info(
                '==============================================================================================')
            sddp_log.info('%4s %15s %15s %15s %12s %12s %12s' %
                          ('Pass', 'LB', 'iUB', 'iHW', 'F time', 'B time', 'Wall time'))
            elapsed_time = self.get_wall_time()
            sddp_log.info('%4s %15.5e %15.5e %15.5e %12.2f %12.2f %12.2f' %
                          ("Sim%i" %
                           (alg_options['sim_iter']), self.lb, self.ub, self.ub_hw, fp_time, bp_time, elapsed_time))
            sddp_log.info(
                '==============================================================================================')
    
    def termination(self):
        elapsed_time = self.get_wall_time()
        if self.pass_iteration >= alg_options['max_iter'] or elapsed_time > alg_options['max_time']:
            return True
        if self.pass_iteration > 0:
            if self.lb >= self.ub - self.ub_hw - alg_options['opt_tol']:  #- self.ub_hw -
                return True
        return False
    
    def process_out_of_sample_simulation(self, out_of_sample_setup):
        '''
            If out_of_sample_setup is not None, performs an
            out-of-sample simulation during the training phase of SDDP.
            oos_setup (dict): see SDDP.run function for documentation.
        '''
        if out_of_sample_setup is not None and len(out_of_sample_setup['when']) > 0:
            wall_time = self.get_wall_time()
            self.alg_chrono.pause()  # pause to avoid counting the simulation time
            if wall_time > out_of_sample_setup['when'][0]:
                out_of_sample_setup['when'].pop(0)
                oos_rnd_container = out_of_sample_setup['random_container']
                sim = self.simulate_policy(oos_rnd_container)
                oos_output = (self.pass_iteration, wall_time, sim)
                out_of_sample_setup['output'].append(oos_output)
                reset_out_sample_gen()
            self.alg_chrono.resume()  # resume chronometer
    
    def run(self, pre_sample_paths=None, instance_name='Default', out_of_sample_setup=None):
        '''
        Starts the optimization routine in SDDP
        
        Args:
            pre_sample_paths(list of dict): list that contains sample paths
                to run in the forward passes
            instance_name (string): descriptive name of the problem
            out_of_sample_setup (dict): If passed (not None), runs an out-of-sample
                simulation. The setup contains a list of time stamps, a random
                container, and a output handler to save the simulations. the dictionary
                is as follows:
                    {'when' = [t1, t2, t3, t4, ...],
                     'random_container = RandomContainer(),
                     'output = [ ]}
                After running SDDP, the output list is populated with the results in the form:
                    (iteration, time, SimResult)
        '''
        '''
        ==================================================
            Algorithm setup
        ==================================================
        '''
        self.alg_chrono = Chronometer()
        self.alg_chrono.start()
        reset_all_rnd_gen()
        check_options_concitency()
        ev = alg_options['expected_value_problem']
        self.random_container._set_outcomes_for_run(ev)
        dynamic_sampling = alg_options['dynamic_sampling']
        lbs = []
        
        self.lb = None
        self.ub = float('inf')
        self.ub_hw = 0
        self.upper_bounds = []
        self.pass_iteration = 0
        
        self.pass_iteration = 0
        self.init_out(instance_name)
        T = len(self.stage_problems)
        bounded_problem = False
        '''
        ==================================================
            Initialization pass
        ==================================================
        '''
        self.run_oracle_initialization(lbs)
        fp_time = 0
        bp_time = 0
        termination = False
        while termination is False:
            self.process_out_of_sample_simulation(out_of_sample_setup)
            '''
            ==================================================
            Forward pass
            ==================================================
            '''
            f_timer = time.time()
            sample_paths = []
            fp_outputs = []
            for i in range(0, alg_options['n_sample_paths']):
                s_path = None
                if pre_sample_paths is None and dynamic_sampling is False:
                    s_path, _ = self.random_container.getSamplePath(alg_rnd_gen, ev=ev)
                elif pre_sample_paths is not None:
                    s_path = pre_sample_paths.pop()
                else:
                    s_path = list()
                
                output_fp = self.forwardpass(sample_path=s_path)
                sample_paths.append(s_path)
                fp_outputs.append(output_fp)
            fp_time = time.time() - f_timer
            self.stats.updateStats(cs.FORWARD_PASS, total_time=fp_time)
            
            lbs.append((self.lb, self.get_wall_time()))
            '''
            ==================================================
            Compute statistical upper bounds
            ==================================================
            '''
            if self.pass_iteration % 10 == 0 and self.pass_iteration > -1:
                pass
                #self.compute_statistical_bound(alg_options['in_sample_ub'])
                #===============================================================
                # if self.pass_iteration>3:
                #     self.compute_upper_bound_opt_sim_knitro(100)
                #===============================================================
                #    self.compute_upper_bound_math_prog(2*alg_options['in_sample_ub'])
            '''
            ==================================================
            Stopping criteria
            ==================================================
            '''
            termination = self.termination()
            self.iteration_update(fp_time, bp_time, last_iter=termination)
            if termination:
                break
            '''
            ==================================================
            Backward pass
            ==================================================
            '''
            b_timer = time.time()
            for i in range(0, alg_options['n_sample_paths']):
                s_path = sample_paths[i]
                output_fp = fp_outputs[i]
                self.backwardpass(forward_out_states=output_fp, sample_path=s_path, ev=ev)
            bp_time = time.time() - b_timer
            self.stats.updateStats(cs.BACKWARD_PASS, total_time=bp_time)
            
            self.pass_iteration += 1
        
        self.stats.print_report(instance_name, self.stage_problems)
        return lbs
    
    def run_oracle_initialization(self, lbs):
        '''
        If an oracle model is provided, runs SDDP up to a given stage
        and approximates the reminder stages with a single linear program
        in which all random variables take the expected value as realization. 
        '''
        if len(self.stage_oracle_subproblems) == 0:
            return
        reset_all_rnd_gen()
        for limit_stage in range(alg_options['max_stage_with_oracle']):
            oracle_stage = limit_stage + 1
            for lim_stage_iters in range(alg_options['max_iters_oracle_ini']):
                '''Forward path '''
                fp_out_states = []
                sample_path, _ = self.random_container.getSamplePath(alg_rnd_gen)
                for i in range(oracle_stage):
                    sp = self.stage_problems[i]
                    in_state = fp_out_states[-1] if i > 0 else None
                    sp_output = sp.solve(in_state_vals=in_state,
                                         random_realization=sample_path[i],
                                         forwardpass=True,
                                         random_container=self.random_container,
                                         sample_path=sample_path,
                                         num_cuts=self.num_cuts)
                    fp_out_states.append(sp_output['out_state'])
                    if i == 0:  #Update global lower bound
                        self.lb = sp_output['objval']
                lbs.append((self.lb, self.get_wall_time()))
                '''
                Backward pass
                First step in the backward pass calls the oracle problem.
                '''
                
                for t in range(oracle_stage, 0, -1):
                    outputs_per_outcome = []
                    stage_rnd_vector = self.random_container[t]
                    omega_t = stage_rnd_vector.getOutcomes(sample_path, ev=False)
                    sp = None
                    for outcome in omega_t:
                        sp_output = None
                        if t == oracle_stage:
                            sp = self.stage_oracle_subproblems[t]
                            sp_output = sp.solve(in_state_vals=fp_out_states[t - 1],
                                                 random_realization=outcome,
                                                 forwardpass=False,
                                                 random_container=self.random_container,
                                                 sample_path=sample_path)
                        else:
                            sp = self.stage_problems[t]
                            sp_output = sp.solve(in_state_vals=fp_out_states[t - 1],
                                                 random_realization=outcome,
                                                 forwardpass=False,
                                                 random_container=self.random_container,
                                                 sample_path=sample_path)
                        outputs_per_outcome.append(sp_output)
                    sp_cut = self.stage_problems[t - 1]
                    cut_creation_time = self.createStageCut(t - 1, sp_cut, sp, stage_rnd_vector, outputs_per_outcome,
                                                            fp_out_states[t - 1], sample_path)
                self.iteration_update(0, 0)
                self.pass_iteration += 1
    
    def simulate_policy(self, out_of_sample_random_container):
        self.upper_bounds = []
        for i in range(0, alg_options['sim_iter']):
            s_path, _ = out_of_sample_random_container.getSamplePath(out_sample_gen)
            #===================================================================
            # if i in [2, 60]:
            #     #     #pass
            #     print(s_path[3]['inflow[6]'],' ', s_path[9]['inflow[5]'])
            #===================================================================
            if alg_options['outputlevel'] >= 3:
                sddp_log.debug('Simulation %i:' % (i))
                sddp_log.debug(s_path)
            
            output_fp = self.forwardpass(sample_path=s_path, simulation=True)
        sr = SimResult(self.instance, self.upper_bounds.copy())
        if alg_options['outputlevel'] >= 1:
            self.iteration_update(0, 0, force_print=True)
        return sr
    
    def change_dro_radius(self, new_dro_r, cuts_left=20):
        assert self.instance['risk_measure_params']['radius'] <= new_dro_r, 'Invalid change of DRO radius.'
        self.instance['risk_measure_params']['radius'] = new_dro_r
        for sp in self.stage_problems:
            sp.update_cut_pool_dro(cuts_left=cuts_left)
            sp.risk_measure.modify_param(radius=new_dro_r)
            sp.model.update()
    
    def compute_statistical_bound1(self, n_samples):
        self.upper_bounds = []  # rest bound
        for i in range(0, n_samples):
            s_path, _ = self.random_container.getSamplePath(in_sample_gen)
            if alg_options['outputlevel'] >= 3:
                sddp_log.debug('Simulation %i:' % (i))
                sddp_log.debug(s_path)
            output_fp = self.forwardpass(sample_path=s_path, simulation=True)
        
        self.ub = np.mean(self.upper_bounds)
        self.ub_hw = 2 * np.std(self.upper_bounds) / np.sqrt(len(self.upper_bounds))
    
    def compute_statistical_bound(self, n_samples):
        '''
        Computes an statistical upper bound. For DRO risk measures,
        probabilities in the tree change for every sample path to account for
        the worse case expectation. Both the upper bound and halfwidth are stored
        as an attribute of the class.
        
        Args:
            n_samples(int): Number of sample paths to compute the upper bound
        
        NOTES: For DRO risk measure, the worst case probabilities are achieved
                asymptotically with the number of iterations of SDDP. 
        '''
        self.upper_bounds = []  # rest bound
        for k in range(0, n_samples):
            fp_out_states = []
            fp_ub_value = 0
            sample_path = []  #partial sample path
            for (t, sp) in enumerate(self.stage_problems):
                self.random_container.getStageSample(t, sample_path, in_sample_gen)
                in_state = fp_out_states[-1] if t > 0 else None
                sp_output = sp.solve(in_state_vals=in_state,
                                     random_realization=sample_path[t],
                                     forwardpass=True,
                                     random_container=self.random_container,
                                     sample_path=sample_path,
                                     num_cuts=self.num_cuts)
                if sp_output['status'] == cs.SP_INFEASIBLE:
                    self.debrief_infeasible_sub(sample_path, t, sp_output, sp)
                
                fp_out_states.append(sp_output['out_state'])
                fp_ub_value += sp.get_stage_objective_value()
                next_sp = self.stage_problems[t + 1] if t + 1 < len(self.stage_problems) else None
                sp.risk_measure.forward_prob_update(t, sp, next_sp, fp_out_states, sample_path, self.random_container)
            
            self.upper_bounds.append(fp_ub_value)
        self.ub = np.mean(self.upper_bounds)
        self.ub_hw = 2 * np.std(self.upper_bounds) / np.sqrt(len(self.upper_bounds))
        self.random_container.reset_to_nominal_dist()
    
    def compute_upper_bound_math_prog(self, n_samples):
        '''
        Computes an upper bound assigning probabilities to sample paths' solutions.
        First, all sample paths are realized and solved. These sample paths induce a 
        subtree of the full scenario tree which is used to assign probabilities to each 
        sample path. An auxiliary linear program is build to found the maximum (worst-case)
        expectation of the realized samples, introducing constraints to enforce that the 
        distribution at every stage is within the uncertainty set.  
        
        Args:
            n_samples(int): Number of sample paths to compute the upper bound
        '''
        self.random_container.reset_to_nominal_dist()
        sub_tree = ScenarioTree(self.random_container)
        for k in range(0, n_samples):
            fp_out_states = []
            fp_ub_value = 0
            sample_path = []  #partial sample path
            sample_path_outcomes = []  #partial sample path
            for (t, sp) in enumerate(self.stage_problems):
                _, stage_outcome = self.random_container.getStageSample(t, sample_path, in_sample_gen)
                sample_path_outcomes.append(stage_outcome)
                in_state = fp_out_states[-1] if t > 0 else None
                sp_output = sp.solve(in_state_vals=in_state,
                                     random_realization=sample_path[t],
                                     forwardpass=True,
                                     random_container=self.random_container,
                                     sample_path=sample_path,
                                     num_cuts=self.num_cuts)
                if sp_output['status'] != cs.SP_OPTIMAL:
                    self.debrief_infeasible_sub(sample_path, t, sp_output, sp)
                
                fp_out_states.append(sp_output['out_state'])
                fp_ub_value += sp.get_stage_objective_value()
                sp.risk_measure.forward_prob_update(t, self.random_container)
            
            sub_tree.add_sample_path(k, sample_path_outcomes, fp_ub_value)
        
        ub = sub_tree.compute_subtree_upper_bound([sp.risk_measure for sp in self.stage_problems])
        self.random_container.reset_to_nominal_dist()
        # print(ub)
    
    def compute_upper_bound_opt_sim(self, n_samples):
        '''
        Computes an upper bound based on simulation optimization. 
        '''
        ref_state = 1
        ref_sp = self.stage_problems[ref_state]
        ref_rm = ref_sp.risk_measure
        num_outcomes = self.random_container[-1].outcomes_dim  #dimension of the random vector in the last stage
        
        #Getting trial points for probabilities
        ub_max = -np.inf
        e = np.ones(num_outcomes)
        q = np.array([1 / num_outcomes for _ in range(num_outcomes)])
        p = q.copy()
        d = in_sample_gen.uniform(size=num_outcomes)
        n_trial = int(np.maximum(1, np.log(n_samples)))
        trial_points = []
        for k in range(n_trial):
            proj_d = (e.dot(d) / e.dot(e)) * e
            orth_d = d - proj_d
            assert np.abs(orth_d.dot(e)) < 1E-8, 'wrong projection'
            d_range = ref_rm.orthogonal_proj_uncertainty_set(p, orth_d, self.random_container[ref_state], ref_state - 1)
            p = p + orth_d * d_range  #in_sample_gen.uniform(0,d_range)
            trial_points.append(p.copy())
            
            #Apply solution to the random container
            for srv in self.random_container.stage_vectors[1:]:
                srv.p = p.copy()
            sample_path_outcomes = []
            self.upper_bounds = []  # rest bound
            for i in range(0, n_samples):
                s_path, s_path_outcomes = self.random_container.getSamplePath(in_sample_gen)
                self.forwardpass(sample_path=s_path, simulation=True)
                sample_path_outcomes.append(s_path_outcomes)
                
                #assert len(sample_path_outcomes)==len(self.upper_bounds), 'Scenario data mismatching'
            
            self.random_container.reset_to_nominal_dist()
            branches = np.zeros((num_outcomes, n_samples))
            for i in range(0, n_samples):
                for o_t in sample_path_outcomes[0][1:]:
                    branches[o_t, i] += 1
            print(p)
            # update direction
            for o in range(0, num_outcomes):
                d[o] = 0
                for w in range(n_samples):
                    if branches[o, w] > 0:
                        branch_weight = 1
                        for oo in range(0, num_outcomes):
                            if o == oo:
                                branch_weight = branch_weight * (p[oo]**(branches[o, w] - 1))
                            else:
                                branch_weight = branch_weight * (p[oo]**(branches[o, w]))
                            d[o] += branches[o, w] * branch_weight * self.upper_bounds[w]
            
            d = d / np.linalg.norm(d)  # unit vector
            ub_k = np.mean(self.upper_bounds)
            print(ub_k)
            if ub_k > ub_max:
                ub_max = ub_k
        
        print(ub_max)
    
    def compute_upper_bound_opt_sim_knitro(self, n_samples):
        '''
        Computes an upper bound based on simulation optimization. 
        '''
        ref_state = 1
        ref_sp = self.stage_problems[ref_state]
        ref_rm = ref_sp.risk_measure
        num_outcomes = self.random_container[-1].outcomes_dim  #dimension of the random vector in the last stage
        
        #Getting trial points for probabilities
        ub_max = -np.inf
        e = np.ones(num_outcomes)
        q = np.array([1 / num_outcomes for _ in range(num_outcomes)])
        p = q.copy()
        try:
            p = self.best_p.copy()
        except:
            pass
        d = in_sample_gen.uniform(size=num_outcomes)
        n_trial = int(np.maximum(1, np.log(n_samples)))
        trial_points = []
        for k in range(n_trial):
            trial_points.append(p.copy())
            #Apply solution to the random container
            print('Simulating: ', p)
            for srv in self.random_container.stage_vectors[1:]:
                srv.p = p.copy()
            sample_path_outcomes = []
            self.upper_bounds = []  # rest bound
            for i in range(0, n_samples):
                s_path, s_path_outcomes = self.random_container.getSamplePath(in_sample_gen)
                self.forwardpass(sample_path=s_path, simulation=True)
                sample_path_outcomes.append(s_path_outcomes)
            
            #Upper bound proxy
            ub_k = np.mean(self.upper_bounds)
            self.random_container.reset_to_nominal_dist()
            branches = np.zeros((num_outcomes, n_samples))
            for i in range(0, n_samples):
                for o_t in sample_path_outcomes[0][1:]:
                    branches[o_t, i] += 1
            
            #Solver worst case expectation and retrive new p vector
            outcomes_org, outcomes_des, distances, r_w = ref_rm.get_dus_params(self.random_container[ref_state],
                                                                               ref_state - 1)
            costs = np.array(self.upper_bounds)
            obj, new_p, status = solve_worst_case_expectation(outcomes_org, outcomes_des, branches, costs, distances,
                                                              r_w)
            
            print(ub_k, obj, status)
            if ub_k > ub_max:
                ub_max = ub_k
                self.best_p = new_p.copy()
                #update vector p
                p = new_p.copy()
        print(ub_max)
    
    def debrief_infeasible_sub(self, sample_path, t, sp_output, sp):
        for ss in sample_path:
            print(ss)
            print('GRB STATUS %i' % (sp.model.status))
            sp.model.write('%s%i_.lp' % (sp_output['status'], t))
            sp.model.computeIIS()
            sp.model.write("model.ilp")
            raise not_optimal_sp('A stage %i problem was not optimal' % (t))
    
    def simulate_single_scenario(self, out_of_sample_random_container, sample_path=None):
        self.upper_bounds = []
        s_path = None
        if sample_path == None:
            s_path, _ = out_of_sample_random_container.getSamplePath(out_sample_gen)
        else:
            s_path = sample_path
        output_fp = self.forwardpass(sample_path=s_path, simulation=True)
        models = [sp.model for sp in self.stage_problems]
        return s_path, output_fp, models


class Stats:
    '''
    Class to keep track of performance statistics.
    '''
    
    def __init__(self):
        self.lp_time = {cs.FORWARD_PASS: 0.0, cs.BACKWARD_PASS: 0.0}
        self.cut_update_time = {cs.FORWARD_PASS: 0.0, cs.BACKWARD_PASS: 0.0}
        self.cut_gen_time = {cs.FORWARD_PASS: 0.0, cs.BACKWARD_PASS: 0.0}
        self.pass_time = {cs.FORWARD_PASS: 0.0, cs.BACKWARD_PASS: 0.0}
        self.model_update_time = {cs.FORWARD_PASS: 0.0, cs.BACKWARD_PASS: 0.0}
        self.lp_counter = {cs.FORWARD_PASS: 0, cs.BACKWARD_PASS: 0}
        self.data_out = {cs.FORWARD_PASS: 0.0, cs.BACKWARD_PASS: 0.0}
        self.lp_times = []
    
    def updateStats(self,
                    passType=cs.FORWARD_PASS,
                    lp_time=0.0,
                    cut_update_time=0.0,
                    cut_gen_time=0.0,
                    model_update_time=0.0,
                    data_out_time=0.0,
                    total_time=0.0,
                    num_lp_ctrs=0,
                    iteration=0):
        self.lp_time[passType] += lp_time
        self.pass_time[passType] += total_time
        self.cut_update_time[passType] += cut_update_time
        self.cut_gen_time[passType] += cut_gen_time
        self.model_update_time[passType] += model_update_time
        self.data_out[passType] += data_out_time
        if lp_time > cs.ZERO_TOL:
            self.lp_counter[passType] += 1
            self.lp_times.append((lp_time, num_lp_ctrs, iteration))
    
    def print_lp_data(self, instance_name, stage_problems):
        sddp_log.info('Simplex Iterations Stats')
        sddp_log.info('%10s %12s %12s %12s' % ('Stage', 'Mean # iter', 'SD # iter', '# Entries'))
        for sp in stage_problems:
            stage = sp.stage
            mean_iter = sp.model_stats.get_mean()
            sd_iter = sp.model_stats.get_sd()
            n_entries = sp.model_stats._simplex_iter_entries
            sddp_log.info('%10i %12.2f %12.2f %12.2f' % (stage, mean_iter, sd_iter, n_entries))
        
        import csv
        file_name = '%s/%s.csv' % (cs.cwd, instance_name)
        with open(file_name, 'w') as myfile:
            fieldnames = ['pass', 'num_ctr', 'lp_time']
            writer = csv.DictWriter(myfile, fieldnames=fieldnames)
            writer.writeheader()
            for x in self.lp_times:
                writer.writerow({'lp_time': "%f" % (x[0]), 'num_ctr': "%i" % (x[1]), 'pass': "%i" % (x[2])})
    
    def print_report(self, instance_name, stage_problems):
        #self.print_lp_data(instance_name, stage_problems)
        sddp_log.info('Time profiling')
        sddp_log.info('%15s %12s %12s %12s %12s %12s %12s %12s %12s' %
                      ('Pass', '# LPs', 'setup', 'simplex', 'output', 'cut update', 'cut gen', 'other', 'total'))
        ''' FORWARD PASS '''
        f_mu = self.model_update_time[cs.FORWARD_PASS]
        f_lp = self.lp_time[cs.FORWARD_PASS]
        f_cu = self.cut_update_time[cs.FORWARD_PASS]
        f_cg = self.cut_gen_time[cs.FORWARD_PASS]
        f_do = self.data_out[cs.FORWARD_PASS]
        f_tot = self.pass_time[cs.FORWARD_PASS]
        f_other = f_tot - (f_mu + f_lp + f_cu + f_cg + f_do)
        f_lp_count = self.lp_counter[cs.FORWARD_PASS]
        sddp_log.info('%15s %12i %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f' \
              % ('Forward', f_lp_count, f_mu, f_lp, f_do, f_cu, f_cg, f_other, f_tot))
        ''' BACKWARD PASS '''
        b_mu = self.model_update_time[cs.BACKWARD_PASS]
        b_lp = self.lp_time[cs.BACKWARD_PASS]
        b_cu = self.cut_update_time[cs.BACKWARD_PASS]
        b_cg = self.cut_gen_time[cs.BACKWARD_PASS]
        b_do = self.data_out[cs.BACKWARD_PASS]
        b_tot = self.pass_time[cs.BACKWARD_PASS]
        b_other = b_tot - (b_mu + b_lp + b_cu + b_cg + b_do)
        b_lp_count = self.lp_counter[cs.BACKWARD_PASS]
        sddp_log.info('%15s %12i %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f' \
              % ('Backward',b_lp_count, b_mu, b_lp, b_do, b_cu, b_cg, b_other, b_tot))
