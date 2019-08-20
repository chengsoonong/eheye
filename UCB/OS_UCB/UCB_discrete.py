from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict
from qreg import QRegressor

# Version: 11/08/2019
# True/Estimate the lower bound of hazard rate L
# True L is calculated by f(0)/ (1 - F(0))
# AbsGau, Exp env can be unified as using the base calss UCB_discrete
# others havn't writen 

class UCB_discrete(ABC):
    """Base class for UCB algorithms of finite number of arms.
    """
    def __init__(self, env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation):
        """
        Arguments
        ------------------------------------
        env: list
            sequence of instances of Environment (distribution reward of arms)
        medians: list
            list of medians of arms 
        num_rounds: int
            total number of rounds
        est_flag: boolean
            indicate whether estimation the lower bound of hazard rate L
            True: estimate L
            False: use the true L = f(0)/ (1 - F(0)), where f is PDF and F is CDF
        hyperpara: list
            [alpha, beta]
        evaluation: list
            ['sd', 'r', 'bd'] 
            i.e. (sub-optimal draws (sd), regret (r), % best arm draws (bd))

        sample_rewards: dict 
            keys: arm ID (0,1,2,..); 
            values: list of sampled rewards for corresponding arm
        selectedActions: list
            sequence of pulled arm ID 

        true_L_list: list
            sequence of true L (lower bound of hazard rate)

        ---------------For Evaluation-----------------------------------------    
        cumulativeRegrets: list
            difference between cumulativeReward and bestActionCumulativeReward
            for each round 
        suboptimalDraws: list
            sum of draws of each sub-optimal arm
        bestDraws: list
            % best arm drawn
        """
        self.env = env
        self.num_rounds = num_rounds
        self.medians = medians
        self.est_flag = est_flag
        self.hyperpara = hyperpara
        self.evaluation = evaluation

        self.bestarm = np.argmax(medians)
        self.selectedActions = []
        self.sample_rewards = defaultdict(list)

        self.cumulativeRegrets = []
        self.suboptimalDraws = []
        self.bestDraws = []
        self.estimated_para = defaultdict(list)

        self.true_L_list = []

    def init_L(self):
        """Initialise the true_L_list for the use of true L,
        where L is the lower bound of hazard rate.
        L = f(0)/ (1 - F(0))
        """
        for i in range(len(self.env)):
            my_env = self.env[i]
            x = 0
            L = my_env.pdf(x)/ (1- my_env.cdf(0))
            assert L > 0
            self.true_L_list.append(L)
        #print(self.true_L_list)
    
    def Calcu_L(self, arm_idx):
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

        if self.est_flag:
            # estimate L
            # To be finished
            sorted_data = np.asarray(sorted(self.sample_rewards[arm_idx]))
            L = len(sorted_data[sorted_data <= 1])/len(sorted_data)
            if L  == 0:
                L = 0.1
            return L
        else:
            # true L = f(0)/ (1 - F(0))
            return self.true_L_list[arm_idx]
    
    #@abstractmethod
    def argmax_ucb(self, t):
        """Compute upper confidence bound 

        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """

        if t % 10000 ==0:
            print('In round ', t)
            print_flag = True
        else:
            print_flag = False

        policy = []
        alpha, beta = self.hyperpara
        for arm in sorted(self.sample_rewards.keys()):  
            reward = self.sample_rewards[arm]
            emp_median = np.median(reward)
            t_i = len(reward)
            
            L = self.Calcu_L(arm)
            #print(L)
            v_t = 4.0 /( t_i * L**2)
            eps = alpha * np.log(t)
            d = np.sqrt(2 * v_t * eps) + 2 * eps * np.sqrt(v_t/t_i)
            policy.append(emp_median + beta * d)
            if print_flag:
                print('For arm ', arm, ' Meidan: ', emp_median, ' d: ', d, 'policy: ', emp_median + self.beta * d)
            #print(policy)
        #print('Choose policy: ', np.argmax(policy))
        return np.argmax(policy)
    
    def init_reward(self):
        """pull each arm once and get the rewards as the inital reward 
        """
        for i, p in enumerate(self.env):
            self.sample_rewards[i].append(p.sample())
       
    def sample(self, idx):
        """sample for arm specified by idx

        Parameters
        -----------------------------
        idx: int
            the idx of arm with maximum ucb in the current round
        
        Return
        ------------------------------
        reward: float
            sampled reward from idx arm
        """
        reward = self.env[idx].sample()
        self.sample_rewards[idx].append(reward)

    def evaluate(self, t):
        """Evaluate the policy
            sd: sub-optimal draws
            r: regret
            bd: percent of best draws
 
        Parameters
        -----------------------------
        t: current time 

        Return
        -----------------------------
        None
        """
        for i in self.evaluation:
            if i == 'sd': 
                sd = 0
                for key, value in self.sample_rewards.items():
                    if key != self.bestarm:
                        sd += len(value)
                self.suboptimalDraws.append(sd)
            if i == 'r':
                regret = 0
                for key, value in self.sample_rewards.items():
                    median_diff = self.medians[self.bestarm] - self.medians[key]
                    regret += median_diff * len(value)
                self.cumulativeRegrets.append(regret)
            if i == 'bd':
                bd = float(len(self.sample_rewards[self.bestarm]))/t
                self.bestDraws.append(bd)

    def play(self):
        """Simulate UCB algorithms for specified rounds.
        """ 
        self.init_reward()
        self.init_L()
        #print('init L:', self.true_L_list)
        for i in range(len(self.env),self.num_rounds):
            #print('Round: ', i)
            idx = self.argmax_ucb(i)
            self.selectedActions.append(idx)
            #print('select arm: ', idx)
            self.sample(idx)
            #self.num_played[idx] += 1
            #self.evaluate(idx, reward)
            self.evaluate(i)

    def sd_bound(self):
        bounds = []
        for m in range(self.num_rounds):
            j = m + 1
            bound = 0
            for i in range(len(self.medians)):
                delta = self.medians[self.bestarm] - self.medians[i]
                gamma = 32 * np.log(j) * (1 + delta * self.true_L_list[i])
                if i != self.bestarm:
                    bound += (np.sqrt(gamma) + 4 * np.sqrt(2 * np.log(j)))**2/ (delta**2 * self.true_L_list[i] ** 2)
                bound+= (1+ np.pi**2/3)
            bounds.append(bound)
        return bounds

    def r_bound(self):
        bounds = []
        for m in range(self.num_rounds):
            j = m + 1
            bound = 0
            for i in range(len(self.medians)):
                delta = self.medians[self.bestarm] - self.medians[i]
                gamma = 32 * np.log(j) * (1 + delta * self.true_L_list[i])
                if i != self.bestarm:
                    bound += (np.sqrt(gamma) + 4 * np.sqrt(2 * np.log(j)))**2/ (delta * self.true_L_list[i] ** 2)
                bound+= (1+ np.pi**2/3) * delta
            bounds.append(bound)
        return bounds

class UCB_clinical(UCB_discrete):
    def __init__(self, env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation):
        """
        est_flag: whether estimate lower bound of hazard rate
        """
        super().__init__(env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation)

    def init_L(self):
        """Initialise the true_L_list for the use of true L,
        where L is the lower bound of hazard rate.
        L = f(0)/ (1 - F(0))
        """
        for i in range(len(self.env)):
            my_env = self.env[i]
            L = my_env.L_estimate()
            assert L > 0
            self.true_L_list.append(L)

    def Calcu_L(self, arm_idx):
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

        if self.est_flag:
            # estimate L
            # To be finished
            sorted_data = np.asarray(sorted(self.sample_rewards[arm_idx]))
            L = len(sorted_data[sorted_data <= 100])/len(sorted_data)
            if L  == 0:
                L = 0.1
            return L
        else:
            # true L = f(0)/ (1 - F(0))
            return self.true_L_list[arm_idx]

class UCB_os_comb(UCB_discrete):
    """ucb os policy for comb of AbsGau and Exp
    """
    def __init__(self, env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation):
        """
        est_flag: whether estimate lower bound of hazard rate
        """
        super().__init__(env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation)

    def argmax_ucb(self, t):
        """Compute upper confidence bound 

        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """
        if t % 10000 ==0:
            print('In round ', t)
            print_flag = True
        else:
            print_flag = False

        policy = []
        for arm in sorted(self.sample_rewards.keys()):  
            reward = self.sample_rewards[arm]
            emp_median = np.median(reward)
            t_i = len(reward)
            if self.est_flag:
                #TODO
                L = 0.6
            else:
                L = 1.2
            v_t = 4.0 /( t_i * L**2)
            eps = self.alpha * np.log(t)
            b = np.sqrt(2 * v_t * eps) + 2 * eps * np.sqrt(v_t/t_i)
            policy.append(emp_median + self.beta * b)
            if print_flag:
                print('For arm ', arm, ' Meidan: ', emp_median, ' b: ', b, 'policy: ', emp_median + self.beta * b)
            #print(policy)
        #print('Choose policy: ', np.argmax(policy))
        return np.argmax(policy)

    def sd_bound(self):
        bounds = []
        return bounds
        
    def r_bound(self):
        bounds = []
        return bounds

class UCB_os_exp(UCB_discrete):
    """class for UCB of order statistics 
    (in terms of exponential distribution)
    regret is evaluted in terms of median rather than mean. 

    Based on the policy:
    argmax_{i \in \mathcal{K}} \hat{m}_{i, T_i(t)} + \beta(\sqrt{2v_t \varepsilon} + 2 \varepsilon \sqrt{v_t/T_i(t)}),
    where $\hat{m}_{i, T_i(t)}$ is the empirical median for arm i at the round t, 
    $\varepsilon = \alpha \log t$, $v_t = \frac{8 \sigma^2}{T_i(t) log2}$. 
    $T_i(t)$ is the number of times arm i has been played until round t. 
    
    """
    def __init__(self, env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation):
        super().__init__(env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation)

    def argmax_ucb(self, t):
        """Compute upper confidence bound 

        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """
        if t % 10000 ==0:
            print('In round ', t)
            print_flag = True
        else:
            print_flag = False

        policy = []
        for arm in sorted(self.sample_rewards.keys()):  
            reward = self.sample_rewards[arm]
            emp_median = np.median(reward)
            t_i = len(reward)
            
            L = self.Calcu_L(arm)
            v_t = 4.0 /( t_i * L**2)
            eps = self.alpha * np.log(t)
            b = np.sqrt(2 * v_t * eps) + 2 * eps * np.sqrt(v_t/t_i)
            policy.append(emp_median + self.beta * b)
            if print_flag:
                print('For arm ', arm, ' Meidan: ', emp_median, ' b: ', b, 'policy: ', emp_median + self.beta * b)
            #print(policy)
        #print('Choose policy: ', np.argmax(policy))
        return np.argmax(policy)

    def sd_bound(self):
        bounds = []
        for m in range(self.num_rounds):
            j = m + 1
            bound = 0
            for i in range(len(self.medians)):
                theta = 1.0/self.env[i].para
                delta = self.medians[self.bestarm] - self.medians[i]
                beta = 32 * np.log(j) * (1 + delta * theta)
                if i != self.bestarm:
                    bound += (np.sqrt(beta) + 4 * np.sqrt(2 * np.log(j)))**2/ (delta**2 * theta ** 2)
                bound+= (1+ np.pi**2/3)
            bounds.append(bound)
        return bounds

    def r_bound(self):
        bounds = []
        for m in range(self.num_rounds):
            j = m + 1
            bound = 0
            for i in range(len(self.medians)):
                theta = 1.0/self.env[i].para
                delta = self.medians[self.bestarm] - self.medians[i]
                beta = 32 * np.log(j) * (1 + delta * theta)
                if i != self.bestarm:
                    bound += (np.sqrt(beta) + 4 * np.sqrt(2 * np.log(j)))**2/ (delta * theta ** 2)
                bound+= (1+ np.pi**2/3) * delta
            bounds.append(bound)
        return bounds

class UCB_os_gau(UCB_discrete):
    """class for UCB of order statistics 
    (in terms of absolute value of standard gaussian distribution)
    regret is evaluted in terms of median rather than mean. 

    Arguments
    -------------------------------------------------------------
    medians: list
        sequence of medians of arms 
    """
    def __init__(self, env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation):
        super().__init__(env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation)

    def var_estimate(self, arm_idx):
        """estimate variance for exponential distribution of arm_idx)
        var + mean ** 2
        
        Parameter
        ---------------------------------------
        arm_idx: int
            the index of arm needed to be estimate
        
        Return
        ----------------------------------------
        var: positive float
            the estimated variance for exponential distribution of arm_idx
        """
        rewards = self.sample_rewards[arm_idx]
        var = np.var(rewards)
        mean = np.mean(rewards)

        if var == 0:
            return 1.0
        else:
            return var + mean ** 2

    def argmax_ucb(self, t):
        """Compute upper confidence bound 

        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """
        if t % 10000 ==0:
            print('In round ', t)
            print_flag = True
        else:
            print_flag = False

        policy = []
        for arm in sorted(self.sample_rewards.keys()):  
            reward = self.sample_rewards[arm]
            emp_median = np.median(reward)
            t_i = len(reward)
            L = self.Calcu_L(arm)
            v_t = 4
            if self.est_flag:
                v_t = 8.0 * self.var_estimate(arm)/( t_i * np.log(2))
            else:
                v_t = 8.0 * self.env[arm].para**2 /(t_i * np.log(2))
            eps = self.alpha * np.log(t)
            b = np.sqrt(2 * v_t * eps) + 2 * eps * np.sqrt(v_t/t_i)
            policy.append(emp_median + self.beta * b)
            if print_flag:
                print('For arm ', arm, ' Meidan: ', emp_median, ' b: ', b)
            #print(policy)
        #print('Choose policy: ', np.argmax(policy))
        return np.argmax(policy)

    def sd_bound(self):
        bounds = []
        for m in range(self.num_rounds):
            j = m + 1
            bound = 0
            for i in range(len(self.medians)):
                delta = self.medians[self.bestarm] - self.medians[i]
                sigma = self.env[i].para
                beta = 32 * np.log(j) * sigma * (2 * sigma + delta * np.sqrt(2 * np.log(2)))
                if i != self.bestarm:
                    bound += (np.sqrt(beta) + 8 * sigma * np.sqrt(np.log(j))) ** 2 /(np.log(2) * delta)**2 
                    bound+= (1+ np.pi**2/3) 
            bounds.append(bound)
        return bounds

    def r_bound(self):
        bounds = []
        for m in range(self.num_rounds):
            j = m + 1
            bound = 0
            for i in range(len(self.medians)):
                delta = self.medians[self.bestarm] - self.medians[i]
                sigma = self.env[i].para
                beta = 32 * np.log(j) * sigma * (2 * sigma + delta * np.sqrt(2 * np.log(2)))
                if i != self.bestarm:
                    bound += (np.sqrt(beta) + 8 * sigma * np.sqrt(np.log(j))) ** 2 /(np.log(2) ** 2 * delta) 
                    bound+= (1+ np.pi**2/3) * delta
            bounds.append(bound)
        return bounds

class UCB1_os(UCB_discrete):
    """Implement for UCB1 algorithm
    """
    def __init__(self, env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation):
        super().__init__(env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation)

    def argmax_ucb(self, t):
        """
        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """
        if t % 10000 ==0:
            print('In round ', t)
            print_flag = True
        else:
            print_flag = False

        ucbs = []
        for arm in sorted(self.sample_rewards.keys()):
            reward = self.sample_rewards[arm]
            emp_mean = np.mean(reward)
            b = np.sqrt(2* self.hyperpara[0] ** 2 * np.log(t)/len(reward))
            ucbs.append( emp_mean + b)
            if print_flag:
                print('For arm ', arm, ' Mean: ', emp_mean, ' b: ', b)
        assert len(ucbs) == len(self.env)
        #print('Choose policy: ', np.argmax(ucbs))
        return np.argmax(ucbs)
    
    def sd_bound(self):
        bounds = []
        return bounds

    def r_bound(self):
        bounds = []
        return bounds

class UCB1_os_clinical(UCB1_os):
    def __init__(self, env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation):
        super().__init__(env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation)
    # FOR clinical data, need to be deleted for simulated data
    def init_L(self):
        """Initialise the true_L_list for the use of true L,
        where L is the lower bound of hazard rate.
        L = f(0)/ (1 - F(0))
        """
        for i in range(len(self.env)):
            my_env = self.env[i]
            L = my_env.L_estimate()
            assert L > 0
            self.true_L_list.append(L)


class test_policy(UCB_discrete):
    """class for test policy: 
    choose one arm A with probility p, choose other arms uniformly with probability (1-p)
    """
    def __init__(self, env, num_rounds, medians, p, A):
        self.p = p 
        self.A = A
        self.medians = medians
        bestarm = np.argmax(self.medians)
        super().__init__(env, num_rounds, bestarm)

    def argmax_ucb(self, t):
        if np.random.uniform() <= self.p:
            return self.A
        else:
            other_arms = set(np.arange(len(self.env))) - set([self.A])
            return np.random.choice(list(other_arms))

    def evaluate(self):
        """Calculate cumulative regret and record it

        Parameters
        -----------------------------
        idx: int
            the idx of arm with maximum ucb in the current round
        reward: float
            sample reward for the current round

        Return
        -----------------------------
        None
        """
        my_regret = 0
        for key, value in self.sample_rewards.items():
            mu_diff = self.medians[self.bestarm] - self.medians[key]
            my_regret += mu_diff * len(value)
        self.cumulativeRegrets.append(my_regret)

'''
# To be finished
class KLUCB(UCB_discrete):
    """Code from SMPyBandits Library"""
    def __init__(self, env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation):
        super().__init__(env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation)
    

    def klucb(x, d, kl, upperbound,
        precision=1e-6, lowerbound=float('-inf'), max_iterations=50,):
        """ The generic KL-UCB index computation.

        - ``x``: value of the cum reward,
        - ``d``: upper bound on the divergence,
        - ``kl``: the KL divergence to be used (:func:`klBern`, :func:`klGauss`, etc),
        - ``upperbound``, ``lowerbound=float('-inf')``: the known bound of the values ``x``,
        - ``precision=1e-6``: the threshold from where to stop the research,
        - ``max_iterations=50``: max number of iterations of the loop 
            (safer to bound it to reduce time complexity).
        """
        value = max(x, lowerbound)
        u = upperbound
        _count_iteration = 0
        while _count_iteration < max_iterations and u - value > precision:
            _count_iteration += 1
            m = (value + u) * 0.5
            if kl(x, m) > d:
                u = m
            else:
                value = m
        return (value + u) * 0.5

    def klucbExp(x, d, precision=1e-6):
    """ KL-UCB index computation for exponential distributions, using :func:`klucb`."""
        if d < 0.77:  # XXX where does this value come from?
            upperbound = x / (1 + 2. / 3 * d - sqrt(4. / 9 * d * d + 2 * d))
            # safe, klexp(x,y) >= e^2/(2*(1-2e/3)) if x=y(1-e)
        else:
            upperbound = x * exp(d + 1)
        if d > 1.61:  # XXX where does this value come from?
            lowerbound = x * exp(d)
        else:
            lowerbound = x / (1 + d - sqrt(d * d + 2 * d))
        return klucb(x, d, klGamma, upperbound, precision, lowerbound)
        

    def argmax_ucb(self, t):
        """
        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """

        if t % 10000 ==0:
            print('In round ', t)
            print_flag = True
        else:
            print_flag = False

        policy = []
        c, tolerance = self.hyperpara
        for arm in sorted(self.sample_rewards.keys()):  
            reward = self.sample_rewards[arm]
            t_i = len(reward)
            policy.append(self.klucb(np.sum(reward)/t_i, c * np.log(t)/ t_i, tolerance))
        return np.argmax(policy)
'''