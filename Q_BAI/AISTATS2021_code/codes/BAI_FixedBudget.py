from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Version: Sept/2020
# This file implements Successive Accepts and Rejects Algorithm (SAR) for 
# summary statistics,including mean and quantiles.

# Benchmark algorithms: 
# Fixed budget: uniform sampling; batch elimination (functional bandit)

# Evaluation
# Fixed budget: probability of error; 
# ------------------------------------------------------------------------

class BAI_FixedBudget(ABC):
    """Base class. Best arm identification for fixed budget.

    Attributes
    -------------------------------------------------------------------------
    env: list
        sequence of instances of Environment (reward distribution of arms).
    true_quantile_list: list
        sequence of tau-quantile of arms, i.e. {Q_i^tau}_{i=1}^{K}
    epsilon: float (0,1)
        accuracy level
    tau: float (0,1)
        quantile level
    num_arms: int
        totoal number of arms K
    m: int
        number of arms for recommendation set 
    budget: int
        number of total round/budget.

    rec_set: set
        set of m recommended arms
    m_max_quantile: float
        max^m Q_i^{tau} 
    m_plus_one_max_quantile: float
        max^{m+1} Q_i^{tau}, for calculating gaps
    m_argmax_arm: int
        arm index: argmax^m Q_i^{tau}
    sample_rewards: dict 
        keys: arm i in {0,1,2,.., K}; 
        values: list of sampled rewards for corresponding arm
    selectedActions: list
        sequence of pulled arm ID 
    """
    def __init__(self, env, true_quantile_list, epsilon, tau, m, 
                fixed_samples = None, budget = 2000):
        """
        Parameters
        ----------------------------------------------------------------------
        env: list
            sequence of instances of Environment (reward distribution of arms).
        true_quantile_list: list
            sequence of tau-quantile of arms, i.e. {Q_i^\tau}_{i=1}^{K}
        epsilon: float (0,1)
            accuracy level
        tau: float (0,1)
            quantile level
        m: int
            number of arms for recommendation set 

        fixed_samples: dict, default is None
            key: arm_dix; values: list of fixed samples
        """

        self.env = env
        self.true_quantile_list = true_quantile_list
        self.epsilon = epsilon
        self.tau = tau
        self.m = m
        self.budget = budget
        
        self.num_arms = len(self.env)
        self.m_max_quantile = np.sort(self.true_quantile_list)[::-1][self.m-1]
        self.m_plus_one_max_quantile = np.sort(self.true_quantile_list)[::-1][self.m]
        self.m_argmax_arm = np.argsort(-1 * np.asarray(self.true_quantile_list))[self.m-1]
        self.sample_rewards = defaultdict(list)
        self.selectedActions = []

        # recommendations
        self.rec_set = set()
        
        # For debug
        self.print_flag = False
        self.print_every = 100
        self.fixed_samples = fixed_samples

    @abstractmethod
    def simulate(self):
        """Simulate experiments. 
        """

    def evaluate(self):
        """Evaluate the performance (probability of error).
        """
        #print('rec_Set: ', self.rec_set)
        rec_set_min = np.min(np.asarray(self.true_quantile_list)[np.asarray(list(self.rec_set))])
        #print('rec_set_min: ', rec_set_min)
        #print('m_max_quantile: ', self.m_max_quantile )
        simple_regret_rec_set =  self.m_max_quantile - rec_set_min
        # the probability is calculated in terms of a large number of experiments
        if simple_regret_rec_set > self.epsilon:
            return 1
        else:
            return 0 

    def sample(self, arm_idx, sample_idx = None):
        """sample for arm specified by idx

        Parameters
        -----------------------------
        arm_idx: int
            the idx of arm with maximum ucb in the current round

        sample_idx: int
            sample from fixed sample list (for debug)
            if None: sample from env
            if int: sample as fixed_sample_list[sample_idx]
        
        Return
        ------------------------------
        reward: float
            sampled reward from idx arm
        """
        if sample_idx == None:
            reward = self.env[arm_idx].sample()
        else:
            #print('sample idx: ', sample_idx)
            #print(self.fixed_samples[arm_idx])
            reward = self.fixed_samples[arm_idx][sample_idx]
        self.sample_rewards[arm_idx].append(reward)
        return reward

class Q_SAR(BAI_FixedBudget):
    """Quantile Successive accepts and rejects algorithm.
    """
    def __init__(self, env, true_quantile_list, epsilon, tau, m, 
                fixed_samples, budget):
        super().__init__(env, true_quantile_list, epsilon, tau, m, 
                fixed_samples,budget)
        
        self.barlogK = 0.5
        for i in range(2, self.num_arms + 1):
            self.barlogK += 1.0/i
        
        # number of arms left to recommend
        self.l = self.m
        
        # active arms with idx 0, 1, ... K-1
        self.active_set = set(list(range(self.num_arms)))

    def cal_n_p(self,p):
        """Calculate n_p, the number of samples of each arm for phase p

        Parameters
        ----------------------------------------------------------------
        p: int
            current phase

        Return
        -----------------------------------------------------------------
        n_p: int
            the number of samples of each arm for phase p
        """
        n_p_float = 1.0/self.barlogK * (self.budget - self.num_arms)/ (self.num_arms + 1 - p)
        if n_p_float - int(n_p_float) > 0:
            n_p = int(n_p_float) + 1
        else:
            n_p = int(n_p_float)
        return n_p

    def simulate(self):
        """Simulate experiments. 
        """
        n_last_phase = 0 # n_0
        for p in range(1, self.num_arms): # for p = 1, ..., K-1
            n_current_phase = self.cal_n_p(p)
            num_samples =  n_current_phase - n_last_phase
            # step 1
            for i in self.active_set:
                for j in range(num_samples):
                    if self.fixed_samples != None:
                        self.sample(i, len(self.sample_rewards[i]))
                    else:
                        self.sample(i)
            # step 2
            quantiles = {} # key: arm idx; value: empirical tau-quantile
            rank_dict = {} # key: arm idx; value: rank according to empirical tau-quantile
            #print('active set: ', self.active_set)
            for i in self.active_set:
                reward = self.sample_rewards[i]
                # not sure why returns an array of one element instead of a scalar
                quantiles[i] = np.quantile(list(reward), self.tau)
            argsort_quantiles = np.argsort(list(quantiles.values()))[::-1]

            for rank, idx in enumerate(argsort_quantiles):
                rank_dict[list(quantiles.keys())[idx]] = rank
                if rank == self.l: # l_p + 1
                    q_l_1 = list(quantiles.keys())[idx]
                if rank == self.l -1: # l_p
                    q_l = list(quantiles.keys())[idx]

            empirical_gap_dict = {} # key: arm idx; value: empirical gap
            for idx, rank in sorted(rank_dict.items(), key=lambda item: item[1]):
                if rank <= self.l - 1: # i <= l_p, rank starts from 0, so l-1
                    empirical_gap_dict[idx] = quantiles[idx] - quantiles[q_l_1]
                else:
                    empirical_gap_dict[idx] = quantiles[q_l] - quantiles[idx]
                
                if empirical_gap_dict[idx] < 0:
                    print('ERROR: empirical gap small than 0')
                    print('quantiles: ', quantiles)
                    print('rank_dict: ', rank_dict)
                    print('q_l_1: ', q_l_1)
                    print('q_l: ', q_l)
                    print('empirical_gap_dict: ', empirical_gap_dict)

            

            # step 3: select arm with maximum empirical gap
            # assume only one arm has the maximum empirical gap
            i_p = sorted(empirical_gap_dict.items(), key=lambda item: item[1])[-1][0]
            self.active_set.remove(i_p)

            # step 4
            if quantiles[i_p] > quantiles[q_l_1] - self.epsilon:
                self.rec_set.add(i_p)
                self.l -= 1

            n_last_phase = n_current_phase

            if self.print_flag: # for debug
                print('phase: ', p)
                print('quantiles: ', quantiles)
                print('rank_dict: ', rank_dict)
                print('empirical_gap_dict: ', empirical_gap_dict)
                print('active_Set: ', self.active_set)
                print('rec_set: ', self.rec_set)
                print('l: ', self.l)
                print('q_l_1: ', q_l_1)
                print('q_l: ', q_l)
                print('i_p: ', i_p)
                print()
            
        assert len(self.active_set) == 1
        self.rec_set = self.rec_set.union(self.active_set)
        # print('rec_set: ', self.rec_set)
        # TODO: the assert can be broken for epsilon > 0
        assert len(self.rec_set) == self.m
        
class Q_SAR_Simplified(Q_SAR):
    """Quantile Successive accepts and rejects algorithm, a simplified version.
    """
    def simulate(self):
        """Simulate experiments. 
        """
        n_last_phase = 0 # n_0
        #p_list = []
        for p in range(1, self.num_arms): # for p = 1, ..., K-1
            n_current_phase = self.cal_n_p(p)
            num_samples =  n_current_phase - n_last_phase
            #p_list.append(num_samples)
            # step 1
            for i in self.active_set:
                for j in range(num_samples):
                    if self.fixed_samples != None:
                        self.sample(i, len(self.sample_rewards[i]))
                    else:
                        self.sample(i)
            # step 2
            quantiles = {} # key: arm idx; value: empirical tau-quantile
            rank_dict = {} # key: arm idx; value: rank according to empirical tau-quantile
            #print('active set: ', self.active_set)
            for i in self.active_set:
                reward = self.sample_rewards[i]
                # not sure why returns an array of one element instead of a scalar
                quantiles[i] = np.quantile(list(reward), self.tau)
            argsort_quantiles = np.argsort(list(quantiles.values()))[::-1]
            if self.print_flag:
                print('qauntiles: ', quantiles)
                print('argsorted quantiles: ', argsort_quantiles)

            for rank, idx in enumerate(argsort_quantiles):
                arm_idx = list(quantiles.keys())[idx]
                rank_dict[arm_idx] = rank
                if rank == 0:
                    a_best = arm_idx
                if rank == self.l: # l_p + 1
                    q_l_1 = arm_idx
                if rank == self.l -1: # l_p
                    q_l = arm_idx
                if rank == len(argsort_quantiles) - 1:
                    a_worst = arm_idx

            gap_accept = quantiles[a_best] - quantiles[q_l_1]
            gap_reject = quantiles[q_l] - quantiles[a_worst]

            if self.print_flag:
                print('rank dict: ', rank_dict)
                print('a_best: ', a_best)
                print('q_l_1: ', q_l_1)
                print('q_l: ', q_l)
                print('a_worst: ', a_worst)
                print('gap accept: ', gap_accept)
                print('gap reject: ', gap_reject)

            if gap_accept > gap_reject:
                # print('accept ', a_best)
                self.rec_set.add(a_best)
                self.active_set.remove(a_best)
                self.l -= 1
            else:
                self.active_set.remove(a_worst)

            n_last_phase = n_current_phase

        assert len(self.active_set) == 1
        self.rec_set = self.rec_set.union(self.active_set)
        # print('rec_set: ', self.rec_set)
        # print()
        # TODO: the assert can be broken for epsilon > 0
        assert len(self.rec_set) == self.m
        #self.p_list = p_list
        #plt.plot(list(range(len(self.p_list))), p_list, marker = '.')
        #plt.show()

class Q_SAR_Simplified_Large_Margin(BAI_FixedBudget):
    """Quantile Successive accepts and rejects algorithm, a simplified version.
    """
    def __init__(self, env, epsilon, tauL, tauH, 
                 true_quantile_tauL, true_quantile_tauH, m, 
                 budget):
        """
        Parameters
        ----------------------------------------------------------
        budget: int
            number of total round/budget.
        """
        super().__init__(env, true_quantile_list, epsilon, tau, m, 
                fixed_samples, budget)
        self.budget = budget
        
        self.barlogK = 0.5
        for i in range(2, self.num_arms + 1):
            self.barlogK += 1.0/i
        
        # number of arms left to recommend
        self.l = self.m
        
        # active arms with idx 0, 1, ... K-1
        self.active_set = set(list(range(self.num_arms)))

    def cal_n_p(self,p):
        """Calculate n_p, the number of samples of each arm for phase p

        Parameters
        ----------------------------------------------------------------
        p: int
            current phase

        Return
        -----------------------------------------------------------------
        n_p: int
            the number of samples of each arm for phase p
        """
        n_p_float = 1.0/self.barlogK * (self.budget - self.num_arms)/ (self.num_arms + 1 - p)
        if n_p_float - int(n_p_float) > 0:
            n_p = int(n_p_float) + 1
        else:
            n_p = int(n_p_float)
        return n_p

    def simulate(self):
        """Simulate experiments. 
        """
        n_last_phase = 0 # n_0
        #p_list = []
        for p in range(1, self.num_arms): # for p = 1, ..., K-1
            n_current_phase = self.cal_n_p(p)
            num_samples =  n_current_phase - n_last_phase
            #p_list.append(num_samples)
            # step 1
            for i in self.active_set:
                for j in range(num_samples):
                    if self.fixed_samples != None:
                        self.sample(i, len(self.sample_rewards[i]))
                    else:
                        self.sample(i)
            # step 2
            quantiles = {} # key: arm idx; value: empirical tau-quantile
            rank_dict = {} # key: arm idx; value: rank according to empirical tau-quantile
            #print('active set: ', self.active_set)
            for i in self.active_set:
                reward = self.sample_rewards[i]
                # not sure why returns an array of one element instead of a scalar
                quantiles[i] = np.quantile(list(reward), self.tau)
            argsort_quantiles = np.argsort(list(quantiles.values()))[::-1]
            if self.print_flag:
                print('qauntiles: ', quantiles)
                print('argsorted quantiles: ', argsort_quantiles)

            for rank, idx in enumerate(argsort_quantiles):
                arm_idx = list(quantiles.keys())[idx]
                rank_dict[arm_idx] = rank
                if rank == 0:
                    a_best = arm_idx
                if rank == self.l: # l_p + 1
                    q_l_1 = arm_idx
                if rank == self.l -1: # l_p
                    q_l = arm_idx
                if rank == len(argsort_quantiles) - 1:
                    a_worst = arm_idx

            gap_accept = quantiles[a_best] - quantiles[q_l_1]
            gap_reject = quantiles[q_l] - quantiles[a_worst]

            if self.print_flag:
                print('rank dict: ', rank_dict)
                print('a_best: ', a_best)
                print('q_l_1: ', q_l_1)
                print('q_l: ', q_l)
                print('a_worst: ', a_worst)
                print('gap accept: ', gap_accept)
                print('gap reject: ', gap_reject)

            if gap_accept > gap_reject:
                # print('accept ', a_best)
                self.rec_set.add(a_best)
                self.active_set.remove(a_best)
                self.l -= 1
            else:
                self.active_set.remove(a_worst)

            n_last_phase = n_current_phase

        assert len(self.active_set) == 1
        self.rec_set = self.rec_set.union(self.active_set)
        # print('rec_set: ', self.rec_set)
        # print()
        # TODO: the assert can be broken for epsilon > 0
        assert len(self.rec_set) == self.m
        #self.p_list = p_list
        #plt.plot(list(range(len(self.p_list))), p_list, marker = '.')
        #plt.show()

    def evaluate(self):
        """Evaluate the performance (probability of error).
        """
        #print('rec_Set: ', self.rec_set)
        rec_set_min = np.min(np.asarray(self.true_quantile_list)[np.asarray(list(self.rec_set))])
        #print('rec_set_min: ', rec_set_min)
        #print('m_max_quantile: ', self.m_max_quantile )
        simple_regret_rec_set =  self.m_max_quantile - rec_set_min
        # the probability is calculated in terms of a large number of experiments
        if simple_regret_rec_set > self.epsilon:
            return 1
        else:
            return 0

#-------------------------------------------------------------------------
# Baseline Algorithms: Fixed budget
# Including: uniform sampling; batch_elimination
# ------------------------------------------------------------------------

class uniform_sampling(BAI_FixedBudget):

    def simulate(self):
        draw_each_arm = int(self.budget/self.num_arms)

        for idx in range(self.num_arms):
            for t in range(draw_each_arm):
                if self.fixed_samples != None:
                    self.sample(idx, len(self.sample_rewards[idx]))
                else:
                    self.sample(idx)
        
        if self.budget - draw_each_arm * self.num_arms > 0:
            for t in range(self.budget - draw_each_arm * self.num_arms):
                idx = int(np.random.uniform(0, self.num_arms - 1))
                if self.fixed_samples != None:
                    self.sample(idx, len(self.sample_rewards[idx]))
                else:
                    self.sample(idx)
        
        emp_quantile_list = []
        for idx in range(self.num_arms):
            reward = self.sample_rewards[idx]
            emp_quantile = np.quantile(reward, self.tau)
            emp_quantile_list.append(emp_quantile)

        self.rec_set = set(np.argsort(emp_quantile_list)[::-1][:self.m])
        

class batch_elimination(BAI_FixedBudget):
    """Tran-Thanh and Yu 2018,
       Functional Bandits, Algorithm 1.
       Select x1 = ... = xL = 1. i.e. L = K-1. Functional as quantiles.
    """
    def __init__(self, env, true_quantile_list, epsilon, tau, m, 
                fixed_samples, budget):
        super().__init__(env, true_quantile_list, epsilon, tau, m, 
                fixed_samples,budget)

        # number of arms left to recommend
        self.l = self.m
        
        # active arms with idx 0, 1, ... K-1
        self.active_set = set(list(range(self.num_arms)))

    def simulate(self):

        H = (self.num_arms - 1) * (1 + self.num_arms/2.0)                                                                                                                                                         
        num_samples = int(self.budget/H)
        for l in range(1, self.num_arms): # 1, ..., K - 1
            for i in self.active_set:
                for j in range(num_samples):
                    if self.fixed_samples != None:
                        self.sample(i, len(self.sample_rewards[i]))
                    else:
                        self.sample(i)

            quantiles = {} # key: arm idx; value: empirical tau-quantile
            rank_dict = {} # key: arm idx; value: rank according to empirical tau-quantile
            #print('active set: ', self.active_set)
            for i in self.active_set:
                reward = self.sample_rewards[i]
                # not sure why returns an array of one element instead of a scalar
                quantiles[i] = np.quantile(list(reward), self.tau)
            argsort_quantiles = np.argsort(list(quantiles.values()))[::-1]

            for rank, idx in enumerate(argsort_quantiles):
                arm_idx = list(quantiles.keys())[idx]
                rank_dict[arm_idx] = rank
                if rank == 0:
                    a_best = arm_idx
                if rank == self.l: # l_p + 1
                    q_l_1 = arm_idx
                if rank == self.l -1: # l_p
                    q_l = arm_idx
                if rank == len(argsort_quantiles) - 1:
                    a_worst = arm_idx

            self.active_set.remove(a_worst)

        self.rec_set = self.active_set
        # only works for self.m = 1
        assert len(self.rec_set) == self.m


