from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Version: June/2020
# This file implements Mean-based Best Arm Identification baseline algorithms, including
# Fixed budget: UGapEb; SAR
# Fixed confidence: UGapEc

# Evaluation
# Fixed budget: probability of error; 
# Fixed confidence: sample complexity.
# ------------------------------------------------------------------------

class MBAI(ABC):
    """Base class. Best arm identification for means. 
    
    Attributes
    -------------------------------------------------------------------------
    env: list
        sequence of instances of Environment (reward distribution of arms).
    true_mean_list: list
        sequence of mean of arms
    epsilon: float (0,1)
        accuracy level
    num_arms: int
        totoal number of arms K
    m: int
        number of arms for recommendation set 
    
    S_idx_list: list
        list of S_idx (set of m selected arms) for each round
    B_St_list: list
        list of B_St_set (upper bound for simple regret for set S_t) value 
    rec_set: set
        set of m recommended arms
    m_max_mean: float
        max^m mu_i 
    m_plus_one_max_mean: float
        max^{m+1} mu_i, for calculating gaps
    m_argmax_arm: int
        arm index: argmax^m mu_i
    sample_rewards: dict 
        keys: arm i in {0,1,2,.., K}; 
        values: list of sampled rewards for corresponding arm
    selectedActions: list
        sequence of pulled arm ID 
    """

    def __init__(self, env, true_mean_list, epsilon,  m, hyperpara, fixed_samples = None, est_H_flag = False):
        """
        Parameters
        ----------------------------------------------------------------------
        env: list
            sequence of instances of Environment (reward distribution of arms).
        true_mean_list: list
            sequence of tau-mean of arms, i.e. {Q_i^\tau}_{i=1}^{K}
        epsilon: float (0,1)
            accuracy level
        tau: float (0,1)
            mean level
        m: int
            number of arms for recommendation set 
        hyperpara: list of hyperparameters
            alpha (fixed budget)/beta (fixed confidence), hyperparameter in gamma
            b: hyperparameter in confidence interval, default is 1.

        fixed_samples: dict, default is None
            key: arm_dix; values: list of fixed samples
        """

        self.env = env
        self.true_mean_list = true_mean_list
        self.epsilon = epsilon
        self.m = m
        self.hyperpara = hyperpara
        
        self.num_arms = len(self.env)
        self.m_max_mean = np.sort(self.true_mean_list)[::-1][self.m-1]
        self.m_plus_one_max_mean = np.sort(self.true_mean_list)[::-1][self.m]
        self.m_argmax_arm = np.argsort(-1 * np.asarray(self.true_mean_list))[self.m-1]
        self.sample_rewards = defaultdict(list)
        self.selectedActions = []

        self.S_idx_list = []
        self.B_St_list = []
        # recommendations
        self.rec_set = set()

        # For debug
        self.print_flag = False
        self.print_every = 100
        self.fixed_samples = fixed_samples
        self.est_H_flag = est_H_flag

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

class UGapE(MBAI):

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
            
            gamma = self.cal_gamma(t)
            b = self.hyperpara[-1]

            D_i = b * np.sqrt(gamma/ t_i)
            D_list.append(D_i)

            if self.print_flag and t % self.print_every == 0:
                print('arm: ', arm)
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
            emp_mean = np.mean(reward)
            D = D_list[arm]
            ucb.append(emp_mean + D)
            lcb.append(emp_mean - D)

        m_max_ucb = np.sort(ucb)[::-1][self.m - 1]
        for arm in sorted(self.sample_rewards.keys()):
            if ucb[arm] >= m_max_ucb: # if arm is in the first m, select m+1 
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

    def cal_empirical_gap(self):
        means = {} # key: arm idx; value: empirical tau-mean
        rank_dict = {} # key: arm idx; value: rank according to empirical tau-mean
        #print('active set: ', self.active_set)
        for i in range(self.num_arms):
            reward = self.sample_rewards[i]
            # not sure why returns an array of one element instead of a scalar
            means[i] = np.mean(list(reward))
        argsort_means = np.argsort(list(means.values()))[::-1]

        for rank, idx in enumerate(argsort_means):
            rank_dict[list(means.keys())[idx]] = rank
            if rank == self.m: # m + 1
                q_l_1 = list(means.keys())[idx]
            if rank == self.m -1: # m
                q_l = list(means.keys())[idx]

        empirical_gap_dict = {} # key: arm idx; value: empirical gap
        for idx, rank in sorted(rank_dict.items(), key=lambda item: item[1]):
            # estimate gap by its upper confidence bound, which gives the lower confidence bound for H 
            gap_interval = 1.0/np.sqrt(2 * len(self.sample_rewards[idx]))
            if rank <= self.m - 1: # i <= m, rank starts from 0, so m-1
                empirical_gap_dict[idx] = means[idx] - means[q_l_1] + gap_interval
            else:
                empirical_gap_dict[idx] = means[q_l] - means[idx] + gap_interval

        return empirical_gap_dict

    def cal_gap(self, idx):
        """Calculate the (true) gap of arm idx.
        """
        if self.true_mean_list[idx] > self.m_plus_one_max_mean: # idx in S_\star
            return self.true_mean_list[idx] - self.m_plus_one_max_mean
        else:
            return self.m_max_mean - self.true_mean_list[idx] 


    def cal_prob_complexity(self):
        """Calculate the (true) probability complexity H for Q-UGapE algorithms
        """
        H = 0
        b = self.hyperpara[-1]
        for idx in range(self.num_arms):
            H += b ** 2/(np.max([(self.cal_gap(idx) + self.epsilon)/2.0, self.epsilon]) ** 2)
        return H

class UGapEb(UGapE):
    """Fixed budget.

    Arguments
    ---------------------------------------------------------------
    prob_error: float
        probability of error (evaluation metric)
    """
    def __init__(self, env, true_mean_list, epsilon, m, 
                hyperpara,  fixed_samples, est_H_flag, budget):
        """
        Parameters
        ----------------------------------------------------------
        budget: int
            number of total round/budget.
        """
        super().__init__(env, true_mean_list, epsilon, m, hyperpara, fixed_samples, est_H_flag = False)
        self.budget = budget
        if self.est_H_flag == False: # use true prob complexity
            self.prob_complexity = self.cal_prob_complexity()
        if self.print_flag:
            print('prob complexity: ', self.prob_complexity)
        self.last_time_pulled = {} # record the round that each arm is pulled last time
                                   # key: arm idx; value: round of arm idx last time pulled 
        self.est_H_list = []

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
        if self.est_H_flag:
            self.prob_complexity = self.cal_prob_complexity()
            self.est_H_list.append(self.prob_complexity)
        # self.hyperpara[0]: alpha
        gamma = self.hyperpara[0] * (t - self.num_arms)/self.prob_complexity
        # gamma = t - self.num_arms
        # print('gamma: ', gamma)
        return gamma

    def simulate(self):
        """Simulate experiments. 
        """
        self.init_reward()
        for i,t in enumerate(range(self.num_arms + 1, self.budget+1)): # t = K + 1, ... N
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
        rec_set_min = np.min(np.asarray(self.true_mean_list)[np.asarray(list(self.rec_set))])
        simple_regret_rec_set =  self.m_max_mean - rec_set_min
        # the probability is calculated in terms of a large number of experiments
        if simple_regret_rec_set > self.epsilon:
            return 1
        else:
            return 0  

class UGapEc(UGapE):
    """Fixed confidence.
    """

    def __init__(self, env, true_mean_list, epsilon, m, 
                hyperpara, fixed_samples, est_H_flag, delta):
        """
        Parameters
        ----------------------------------------------------------
        delta: float
            confidence level
        sample_complexity: int
            number of rounds needed, init as inf
        """
        super().__init__(env, true_mean_list, epsilon, m, hyperpara, fixed_samples, est_H_flag = False)
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
        t = self.num_arms + 1
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
        #print(self.rec_set)
        return self.sample_complexity

class SAR_Simplified(MBAI):
    """Successive accepts and rejects algorithm, a simplified version.
    """
    def __init__(self, env, true_mean_list, epsilon, m, 
                hyperpara, fixed_samples, est_H_flag, budget):
        """
        Parameters
        ----------------------------------------------------------
        budget: int
            number of total round/budget.
        """
        super().__init__(env, true_mean_list, epsilon, m, 
                hyperpara, fixed_samples, est_H_flag = False)
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
            means = {} # key: arm idx; value: empirical tau-mean
            rank_dict = {} # key: arm idx; value: rank according to empirical tau-mean
            #print('active set: ', self.active_set)
            for i in self.active_set:
                reward = self.sample_rewards[i]
                # not sure why returns an array of one element instead of a scalar
                means[i] = np.mean(list(reward))
            argsort_means = np.argsort(list(means.values()))[::-1]
            if self.print_flag:
                print('mean: ', means)
                print('argsorted means: ', argsort_means)

            for rank, idx in enumerate(argsort_means):
                arm_idx = list(means.keys())[idx]
                rank_dict[arm_idx] = rank
                if rank == 0:
                    a_best = arm_idx
                if rank == self.l: # l_p + 1
                    q_l_1 = arm_idx
                if rank == self.l -1: # l_p
                    q_l = arm_idx
                if rank == len(argsort_means) - 1:
                    a_worst = arm_idx

            gap_accept = means[a_best] - means[q_l_1]
            gap_reject = means[q_l] - means[a_worst]

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
        rec_set_min = np.min(np.asarray(self.true_mean_list)[np.asarray(list(self.rec_set))])
        #print('rec_set_min: ', rec_set_min)
        #print('m_max_mean: ', self.m_max_mean )
        simple_regret_rec_set =  self.m_max_mean - rec_set_min
        # the probability is calculated in terms of a large number of experiments
        if simple_regret_rec_set > self.epsilon:
            return 1
        else:
            return 0