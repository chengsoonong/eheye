from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Version: May/2020
# This file implements Quantile-based Best Arm Identification Algorithms, including
# Fixed budget: Q-UGapEb; Q-SAR
# Fixed confidence: Q-UGapEc

# Benchmark algorithms: 
# Fixed budget: uniform sampling; batch elimination (functional bandit)
# Fixed confidence: uniform sampling; QPAC (Szorenyi et al. 2015); 
#                   Max-Q (David and Shimkin 2016); QLUCB (Howard and Ramdas 2019)

# Evaluation
# Fixed budget: probability of error; 
# Fixed confidence: sample complexity.
# ------------------------------------------------------------------------

class QBAI(ABC):
    """Base class. Best arm identification for quantiles. 

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
    est_flag: boolean 
        indicate whether estimation the lower bound of hazard rate L
        True: estimate L
        False: use the true L = f(0)/ (1 - F(0)), where f is PDF and F is CDF
    fixed_L: list, default is None
        if not None, set L to fixed_L
    hyperpara: list of  [TODO]
        L_est_thr: L estimate threshold 
    
    S_idx_list: list
        list of S_idx (set of m selected arms) for each round
    B_St_list: list
        list of B_St_set (upper bound for simple regret for set S_t) value 
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
    true_L_list: list
        sequence of true L (lower bound of hazard rate)
        used to compare with the results with estimated L
    estimated_L_dict: dict
        keys: arm idx
        values: list of estimated L (len is the current number of samples)
    """
    def __init__(self, env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples = None):
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

        hyperpara: list of hyperparameters
            alpha (fixed budget)/beta (fixed confidence), hyperparameter in gamma
            L_est_thr: L estimate threshold 
        est_flag: boolean 
            indicate whether estimation the lower bound of hazard rate L
            True: estimate L
            False: use the true L = f(0)/ (1 - F(0)), where f is PDF and F is CDF
        fixed_L: list, default is None
            if not None, set L to fixed_L
        fixed_samples: dict, default is None
            key: arm_dix; values: list of fixed samples
        """

        self.env = env
        self.true_quantile_list = true_quantile_list
        self.epsilon = epsilon
        self.tau = tau
        self.m = m
        
        self.num_arms = len(self.env)
        self.m_max_quantile = np.sort(self.true_quantile_list)[::-1][self.m-1]
        self.m_plus_one_max_quantile = np.sort(self.true_quantile_list)[::-1][self.m]
        self.m_argmax_arm = np.argsort(-1 * np.asarray(self.true_quantile_list))[self.m-1]
        self.sample_rewards = defaultdict(list)
        self.selectedActions = []

        self.S_idx_list = []
        self.B_St_list = []
        # recommendations
        self.rec_set = set()

        self.hyperpara = hyperpara
        self.est_flag = est_flag
        self.fixed_L = fixed_L

        # For lower bound of hazard rate
        self.true_L_list = []
        self.init_L()
        # for test of sensitivity
        self.estimated_L_dict = {}

        # For debug
        self.print_flag = False
        self.print_every = 100
        self.fixed_samples = fixed_samples

    @abstractmethod
    def simulate(self):
        """Simulate experiments. 
        """

    @abstractmethod
    def evaluate(self):
        """Evaluate the performance.
        """

    def init_reward(self):
        """pull each arm once and get the rewards as the initial reward 
        """
        self.init_times = int(np.ceil(1.0/(1 - self.tau)))
        assert self.init_times >= 1
        for j in range(self.init_times):
            for i, p in enumerate(self.env):
                if self.fixed_samples != None:
                    self.sample(i, len(self.sample_rewards[i]))
                else:
                    self.sample(i)

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

    def init_L(self):
        """Initialise the true_L_list for the use of true L,
        where L is the lower bound of hazard rate.
        L = f(0)/ (1 - F(0))
        """
        for i in range(len(self.env)):
            
            my_env = self.env[i]
            
            if hasattr(my_env, 'hazard_rate'):
                # if existing attribute hazard rate
                L = my_env.hazard_rate(0)
            elif hasattr(my_env, 'pdf'):
                # if pdf and cdf is defined
                L = my_env.pdf(0)/ (1- my_env.cdf(0))  
            else:
                L = my_env.L_estimate(self.hyperpara[-1])
            #assert L > 0
            
            self.true_L_list.append(L)
        # print(self.true_L_list)
    
    def calcu_L(self, arm_idx):
        """estimate the lower bound L of hazard rate for 
           a reward distribution specified by arm_idx
           (with the assumption of non-decreasing hazard rate)
        
        Parameter
        ---------------------------------------
        arm_idx: int
            the index of arm needed to be estimate
        
        Return
        ----------------------------------------
        L: positive float
            the lower bound of hazard rate for arm idx
        """

        sample_reward = self.sample_rewards[arm_idx]
        if self.est_flag:
            if self.fixed_L == None:
                # estimate L
                sorted_data = np.asarray(sorted(sample_reward))
                L = len(sorted_data[sorted_data <= self.hyperpara[-1]])/len(sorted_data)
                if len(self.sample_rewards) ==1 or L  == 0:
                    # TODO: init value
                    L = 0.01 
            else:
                # use fixed L, for test of sensitivity
                L = self.fixed_L[arm_idx]

            if arm_idx in self.estimated_L_dict.keys():
                self.estimated_L_dict[arm_idx].append(L)
            else: 
                self.estimated_L_dict[arm_idx] = []
                self.estimated_L_dict[arm_idx].append(L)
            return L
        else:
            # true L = f(0)/ (1 - F(0))
            return self.true_L_list[arm_idx]
     
class Q_UGapE(QBAI):

    @abstractmethod
    def cal_gamma(self,t):
        """Calculate exploration factor in confidence interval.
            Definition is different for the fixed budget and confidence setting.

        Parameter
        --------------------------------------
        t: int
            the current round

        Return 
        ----------------------------------------
        gamma: float
            exploration factor in confidence interval
        """

    def confidence_interval(self,t):
        """Compute the confidence interval D_i(t)

        Return 
        -----------------------------------
        D_list: list
            list of confidence intervals for arms of round t
        """
        D_list = []
        for arm in sorted(self.sample_rewards.keys()):
            
            reward = self.sample_rewards[arm]
            t_i = len(reward)
            # avoid k_i = 0
            # TODO: if k_i only takes integer, then D_i will increase after observe one sample 
            # k_i = np.max([int(t_i * (1- self.tau)), 1])
            k_i = t_i * (1- self.tau)
            # TODO: try Q-UGapE without L
            L_i = self.calcu_L(arm)
            # L_i = 1
    
            v_i = 2.0/(k_i * L_i ** 2)
            c_i = 2.0/(k_i * L_i)
            
            gamma = self.cal_gamma(t)

            # TODO: hyperparameter
            D_i = (np.sqrt(2 * v_i * gamma) + c_i * gamma)
            D_list.append(D_i)

            if self.print_flag and t % self.print_every == 0:
                print('arm: ', arm)
                print('k_i: ', k_i)
                print('L_i: ', L_i)
                print('v_i: ', v_i)
                print('c_i: ', c_i)
                print('gamma: ', gamma)

        if self.print_flag and t % self.print_every == 0:
            print('D_list: ', D_list)
        return D_list

    def select_arm(self, t, D_list):
        """SELECT ARM Algorithm.

        Parameters
        --------------------------------
        t: int
            the number of current round
        D_list: list
            list of confidence intervals for arms of round t

        Return:
        int: selected arm idx
        --------------------------------
        """
        ucb = []
        lcb = []
        B = []

        for arm in sorted(self.sample_rewards.keys()):
            reward = self.sample_rewards[arm]
            emp_quantile = np.quantile(reward, self.tau)
            D = D_list[arm]
            ucb.append(emp_quantile + D)
            lcb.append(emp_quantile - D)

        m_max_ucb = np.sort(ucb)[::-1][self.m - 1]
        for arm in sorted(self.sample_rewards.keys()):
            if ucb[arm] >= m_max_ucb: # if arm is in the first m, select m+1 
                # TODO: this may lead to negative B 
                B.append(np.sort(ucb)[::-1][self.m] - lcb[arm])
            else:
                B.append(m_max_ucb - lcb[arm])
            
        self.S_idx = np.argsort(B)[:self.m]
        self.S_idx_list.append(self.S_idx)
        non_S_idx = np.argsort(B)[self.m:]

        u_t = np.asarray(non_S_idx)[np.argmax(np.asarray(ucb)[np.asarray(non_S_idx)])]
        l_t = np.asarray(self.S_idx)[np.argmin(np.asarray(lcb)[np.asarray(self.S_idx)])]

        self.B_St = np.max(np.asarray(B)[np.asarray(self.S_idx)])
        self.B_St_list.append(self.B_St)

        if self.print_flag and t % self.print_every == 0:
            print('UCB: ', ucb)
            print('LCB: ', lcb)
            print('m_max_ucb: ', m_max_ucb)
            print('B: ', B)
            print('S idx: ', self.S_idx)
            print('u_t: ', u_t)
            print('l_t: ', l_t)
            print('B_St: ', self.B_St)
            

        if D_list[u_t] > D_list[l_t]:
            if self.print_flag and t % self.print_every == 0:
                print('choose: ', u_t)
                print()
            return u_t
        else:
            if self.print_flag and t % self.print_every == 0:
                print('choose: ', l_t)
                print()
            return l_t
    
    def cal_gap(self, idx):
        """Calculate the (true) gap of arm idx.
        """
        if self.true_quantile_list[idx] > self.m_plus_one_max_quantile: # idx in S_\star
            return self.true_quantile_list[idx] - self.m_plus_one_max_quantile
        else:
            return self.m_max_quantile - self.true_quantile_list[idx] 


    def cal_prob_complexity(self):
        """Calculate the (true) probability complexity H for Q-UGapE algorithms
        """
        H = 0
        for idx in range(self.num_arms):
            omega_1 = np.sqrt(((self.cal_gap(idx) + self.epsilon) * self.true_L_list[idx] + 2)/8.0) - 0.5
            omega_2 = (np.sqrt(self.epsilon * self.true_L_list[idx] + 1) - 1)/2.0 
            
            if self.print_flag:
                print('arm ', idx)
                print('omega_1: ', omega_1)
                print('omega_2: ', omega_2)

            H += 1.0/((1 - self.tau) * (np.max([omega_1, omega_2]) ** 2))
        return H

class Q_UGapEb(Q_UGapE):
    """Fixed budget.

    Arguments
    ---------------------------------------------------------------
    prob_error: float
        probability of error (evaluation metric)
    """
    def __init__(self, env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples, budget):
        """
        Parameters
        ----------------------------------------------------------
        budget: int
            number of total round/budget.
        """
        super().__init__(env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples)
        self.budget = budget
        self.prob_complexity = self.cal_prob_complexity()
        if self.print_flag:
            print('prob complexity: ', self.prob_complexity)
        self.last_time_pulled = {} # record the round that each arm is pulled last time
                                   # key: arm idx; value: round of arm idx last time pulled 

    def cal_gamma(self,t):
        """Calculate exploration factor in confidence interval.
            Definition is different for the fixed budget and confidence setting.

        Parameter
        --------------------------------------
        t: int
            the current round

        Return 
        ----------------------------------------
        gamma: float
            exploration factor in confidence interval
        """
        
        # self.hyperpara[0]: alpha
        gamma = self.hyperpara[0] * (t - self.num_arms)/self.prob_complexity
        # gamma = t - self.num_arms
        # print('gamma: ', gamma)
        return gamma

    def simulate(self):
        """Simulate experiments. 
        """
        self.init_reward()
        for i,t in enumerate(range(self.num_arms * self.init_times + 1, self.budget+1)): # t = K + 1, ... N
            if self.print_flag and t % self.print_every == 0:
                print('Round ', t)
            idx = self.select_arm(t, self.confidence_interval(t))
            if self.fixed_samples != None:
                self.sample(idx, len(self.sample_rewards[idx]))
            else:
                self.sample(idx)
            self.last_time_pulled[idx] = i


        # TODO: B_St increases, so the choice seems not make sense
        # self.rec_set = set(self.S_idx_list[np.argmin(self.B_St_list)])

        # Try to return the last run directly
        self.rec_set = self.S_idx

        # Try to return 
        # $\mathcal{M}_N = \min_{i \in \mathcal{K}} B_{\mathcal{S}_{t_{i}}}\left(t_{i}\right)$, 
        # where t_i is the last time arm i is pulled

        # last_time_pulled_list = np.asarray(list(self.last_time_pulled.values()))
        # B_St_last_time_pulled_list = np.asarray(self.B_St_list)[last_time_pulled_list]
        # S_idx_last_time_pulled_list = np.asarray(self.S_idx_list)[last_time_pulled_list]
        
        # print(self.last_time_pulled)
        # print(last_time_pulled_list)
        # print(B_St_last_time_pulled_list)
        # self.rec_set = set(S_idx_last_time_pulled_list[np.argmin(B_St_last_time_pulled_list)])
        # print(self.rec_set)
        assert len(self.rec_set) == self.m
        if self.print_flag:
            print('Return:')
            print(self.rec_set)
            print(np.min(self.B_St_list))
            print(self.B_St_list)

    def evaluate(self):
        """Evaluate the performance (probability of error).
        """
        rec_set_min = np.min(np.asarray(self.true_quantile_list)[np.asarray(list(self.rec_set))])
        simple_regret_rec_set =  self.m_max_quantile - rec_set_min
        # the probability is calculated in terms of a large number of experiments
        if simple_regret_rec_set > self.epsilon:
            return 1
        else:
            return 0  

class Q_UGapEc(Q_UGapE):
    """Fixed confidence.
    """

    def __init__(self, env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples, delta):
        """
        Parameters
        ----------------------------------------------------------
        delta: float
            confidence level
        sample_complexity: int
            number of rounds needed, init as inf
        """
        super().__init__(env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples)
        self.delta = delta
        self.sample_complexity = np.inf

    def cal_gamma(self,t):
        """Calculate exploration factor in confidence interval.
            Definition is different for the fixed budget and confidence setting.

        Parameter
        --------------------------------------
        t: int
            the current round

        Return 
        ----------------------------------------
        gamma: float
            exploration factor in confidence interval
        """
        # self.hyperpara[0]: beta
        return self.hyperpara[0] * np.log(4 * self.num_arms * t **3/self.delta)

    def simulate(self):
        """Simulate experiments. 

        Return
        ---------------------------------------------
        t: int
            number of round before stopping
            i.e. sample complexity
        S_idx: list
            list of idx of recommended m arms
        """
        self.init_reward()
        t = self.num_arms * self.init_times + 1
        self.B_St = 1 # init B_St
        while self.B_St >= self.epsilon:
            #print('B_St: ', self.B_St)
                
            idx = self.select_arm(t, self.confidence_interval(t))
            if self.fixed_samples != None:
                self.sample(idx, len(self.sample_rewards[idx]))
            else:
                self.sample(idx)
            t += 1

        self.sample_complexity = t
        self.rec_set = set(self.S_idx)
        
        assert len(self.rec_set) == self.m

    def evaluate(self):
        """Evaluate the performance.

        Return
        ---------------------------------------
        t: int
            number of round before stopping
            i.e. sample complexity
        """
        #print('rec_Set: ', self.rec_set)
        rec_set_min = np.min(np.asarray(self.true_quantile_list)[np.asarray(list(self.rec_set))])
        #print('rec_set_min: ', rec_set_min)
        #print('m_max_quantile: ', self.m_max_quantile )
        simple_regret_rec_set =  self.m_max_quantile - rec_set_min
        # the probability is calculated in terms of a large number of experiments
        if simple_regret_rec_set > self.epsilon:
            return [1, self.sample_complexity]
        else:
            return [0, self.sample_complexity]

class Q_SAR(QBAI):
    """Quantile Successive accepts and rejects algorithm.
    """
    def __init__(self, env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples, budget):
        """
        Parameters
        ----------------------------------------------------------
        budget: int
            number of total round/budget.
        """
        super().__init__(env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples)
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
                # TODO: the assert does not satisfied, check.
                # assert empirical_gap_dict[idx] >= 0
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

class Q_SAR_Simplified(QBAI):
    """Quantile Successive accepts and rejects algorithm, a simplified version.
    """
    def __init__(self, env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples, budget):
        """
        Parameters
        ----------------------------------------------------------
        budget: int
            number of total round/budget.
        """
        super().__init__(env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples)
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

class uniform_sampling(QBAI):
    def __init__(self, env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples, budget):
        """
        Parameters
        ----------------------------------------------------------
        budget: int
            number of total round/budget.
        """
        super().__init__(env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples)
        self.budget = budget

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
        

    def evaluate(self):
        """Evaluate the performance (probability of error).
        """
        rec_set_min = np.min(np.asarray(self.true_quantile_list)[np.asarray(list(self.rec_set))])
        simple_regret_rec_set =  self.m_max_quantile - rec_set_min
        # the probability is calculated in terms of a large number of experiments
        if simple_regret_rec_set > self.epsilon:
            return 1
        else:
            return 0  

class batch_elimination(QBAI):
    """Tran-Thanh and Yu 2018,
       Functional Bandits, Algorithm 1.
       Select x1 = ... = xL = 1. i.e. L = K-1. Functional as quantiles.
    """
    def __init__(self, env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples, budget):
        """
        Parameters
        ----------------------------------------------------------
        budget: int
            number of total round/budget.
        """
        super().__init__(env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples)
        self.budget = budget

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
# Baseline Algorithms: Fixed confidence
# Including: QPAC, MaxQ
# ------------------------------------------------------------------------

class QPAC(QBAI):
    """Szorenyi et al 2015, 
       Qualitative Multi-Armed Bandits: A Quantile-Based Approach.
       Algorithm 1 QPCA(delta, epsilon, tau)
    """
    def __init__(self, env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples, delta):
        """
        Parameters
        ----------------------------------------------------------
        delta: float
            confidence level
        sample_complexity: int
            number of rounds needed, init as inf
        """
        super().__init__(env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples)
        self.delta = delta
        self.sample_complexity = np.inf

        # active arms with idx 0, 1, ... K-1
        self.active_set = set(list(range(self.num_arms)))

    def simulate(self):
        t = 1
        while len(self.active_set) > 0:
            for i in self.active_set:
                if self.fixed_samples != None:
                    self.sample(i, len(self.sample_rewards[i]))
                else:
                    self.sample(i)
            
            # TODO: c_t bigger than 1
            c_t = np.sqrt(1.0/(2 * t) * np.log((np.pi ** 2 * t ** 2 * self.num_arms)/(3 * self.delta)))
            if t % 10000 == 0:
                print('t: ', t)
            
            quantile_tau_plus_c = []
            quantile_tau_mins_c = []

            quantile_tau_plus_c_plus_epsilon = []
            quantile_tau_mins_c_plus_epsilon = []

            for i in self.active_set:
                rewards = self.sample_rewards[i]

                if self.tau + c_t > 1 or self.tau - c_t < 0\
                    or self.tau + c_t + self.epsilon > 1 or self.tau - c_t + self.epsilon < 0: 
                    continue
                #     tau_plus_c = 1
                # if self.tau - c_t < 0:
                #     tau_mins_c = 0

                quantile_tau_plus_c.append(np.quantile(rewards, self.tau + c_t))
                quantile_tau_mins_c.append(np.quantile(rewards, self.tau - c_t))
                quantile_tau_plus_c_plus_epsilon.append(np.quantile(rewards, self.tau + c_t + self.epsilon))
                quantile_tau_mins_c_plus_epsilon.append(np.quantile(rewards, self.tau - c_t + self.epsilon))

            if len(quantile_tau_plus_c) < len(self.active_set) or len(quantile_tau_mins_c) < len(self.active_set)\
                or len(quantile_tau_plus_c_plus_epsilon) < len(self.active_set) or len(quantile_tau_mins_c_plus_epsilon) < len(self.active_set):
                if self.print_flag and t % self.print_every == 0:
                    print('t: ' + str(t) + ' not enough samples.')
                t += 1
                continue
            x_plus = np.max(quantile_tau_plus_c)
            x_mins = np.max(quantile_tau_mins_c)

            current_active_set = self.active_set.copy()
            for i in range(len(current_active_set)):
                if quantile_tau_plus_c_plus_epsilon[i] < x_mins:
                    self.active_set.remove(list(current_active_set)[i])
                    if self.print_flag and t % self.print_every == 0:
                        print('t: ', t)
                        print('active set: ', self.active_set)
                if x_plus <= quantile_tau_mins_c_plus_epsilon[i]:
                    hat_i = list(current_active_set)[i]
                    self.rec_set = {hat_i}
                    if self.print_flag and t % self.print_every == 0:
                        print('hat_i: ', hat_i)
                    break
            t += 1

            if len(self.rec_set) > 0:
                break

        self.sample_complexity = t

    def evaluate(self):
        """Evaluate the performance.

        Return
        ---------------------------------------
        t: int
            number of round before stopping
            i.e. sample complexity
        """
        #print('rec_Set: ', self.rec_set)
        rec_set_min = np.min(np.asarray(self.true_quantile_list)[np.asarray(list(self.rec_set))])
        #print('rec_set_min: ', rec_set_min)
        #print('m_max_quantile: ', self.m_max_quantile )
        simple_regret_rec_set =  self.m_max_quantile - rec_set_min
        # the probability is calculated in terms of a large number of experiments
        if simple_regret_rec_set > self.epsilon:
            return [1, self.sample_complexity]
        else:
            return [0, self.sample_complexity]
        
        
class MaxQ(QBAI):
    """David and Shimkin 2016,
    Pure Exploration for Max-Quantile Bandits.
    Algorithm 1 Maximal Quantile (Max-Q) Algorithm.
    """
    def __init__(self, env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples, delta):
        """
        Parameters
        ----------------------------------------------------------
        delta: float
            confidence level
        sample_complexity: int
            number of rounds needed, init as inf
        """
        super().__init__(env, true_quantile_list, epsilon, tau, m, 
                hyperpara, est_flag, fixed_L, fixed_samples)
        self.delta = delta
        self.sample_complexity = np.inf

        # active arms with idx 0, 1, ... K-1
        self.active_set = set(list(range(self.num_arms)))

    def init_reward(self, L):
        """pull each arm once and get the rewards as the initial reward 
        """
        self.init_times = int(3 * L/self.tau) + 1
        assert self.init_times >= 1
        for j in range(self.init_times):
            for i, p in enumerate(self.env):
                if self.fixed_samples != None:
                    self.sample(i, len(self.sample_rewards[i]))
                else:
                    self.sample(i)
        
    def simulate(self):
        L = 6 * np.log(self.num_arms * (1 + (-10) * self.tau * np.log(self.delta)/(self.epsilon ** 2))) - np.log(self.delta)
        self.init_reward(L)
        t = self.num_arms * self.init_times
       
        while True:
            vk_list = []
            t += 1
            if t % 1e4 == 0:
                print(t)
            for k in range(self.num_arms):
                reward = self.sample_rewards[k]
                t_k = len(reward) # C(k) in the paper
                mk = int(self.tau * t_k - np.sqrt(3 * self.tau * t_k * L)) + 1
                vk = np.sort(reward)[::-1][mk - 1] # mk=th largest reward
                vk_list.append(vk)
            
            # TODO: ties need to be broken arbitrary
            k_ast = np.argsort(vk_list)[::-1][0]

            if len(self.sample_rewards[k_ast]) > 10 * self.tau * L / (self.epsilon ** 2):
                self.rec_set = {k_ast}
                self.sample_complexity = t
                break
            else:
                if self.fixed_samples != None:
                    self.sample(k_ast, len(self.sample_rewards[k_ast]))
                else:
                    self.sample(k_ast)
                
        
    def evaluate(self):
        """Evaluate the performance.

        Return
        ---------------------------------------
        t: int
            number of round before stopping
            i.e. sample complexity
        """
        #print('rec_Set: ', self.rec_set)
        rec_set_min = np.min(np.asarray(self.true_quantile_list)[np.asarray(list(self.rec_set))])
        #print('rec_set_min: ', rec_set_min)
        #print('m_max_quantile: ', self.m_max_quantile )
        simple_regret_rec_set =  self.m_max_quantile - rec_set_min
        # the probability is calculated in terms of a large number of experiments
        if simple_regret_rec_set > self.epsilon:
            return [1, self.sample_complexity]
        else:
            return [0, self.sample_complexity]




