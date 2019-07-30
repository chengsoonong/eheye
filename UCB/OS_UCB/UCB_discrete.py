from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict
from qreg import QRegressor

class UCB_discrete(ABC):
    """Base class for UCB algorithms of finite number of arms.
    """
    def __init__(self, env, medians, num_rounds, 
                 est_var, hyperpara, evaluation):
        """
        Arguments
        ------------------------------------
        env: list
            sequence of instances of Environment (distribution reward of arms)
        medians: list
            list of medians of arms 
        num_rounds: int
            total number of rounds
        est_var: bool
            true: use estimate variance; false: use ground true variance
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
        self.est_var = est_var
        self.alpha, self.beta = hyperpara
        self.evaluation = evaluation

        self.bestarm = np.argmax(medians)
        self.selectedActions = []
        self.sample_rewards = defaultdict(list)

        self.cumulativeRegrets = []
        self.suboptimalDraws = []
        self.bestDraws = []
        

    @abstractmethod
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

    def var_estimate(self, arm_idx):
        """estimate variance for exponential distribution of arm_idx)
        pdf = \theta e^(-\theta x)
        var = \theta^(-2)
        
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

        if var == 0:
            return 1.0
        else:
            return var
        
    def regret(self, t):
        """Calculate cumulative regret and record it

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
    
    def init_reward(self):
        """pull each arm once and get the rewards as the inital reward 
        """
        for i, p in enumerate(self.env):
            self.sample_rewards[i].append(p.sample(i))
       
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
        reward = self.env[idx].sample(idx)
        self.sample_rewards[idx].append(reward)

    def play(self):
        """Simulate UCB algorithms for specified rounds.
        """
        
        self.init_reward()
        for i in range(len(self.env),self.num_rounds):
            #print('Round: ', i)
            idx = self.argmax_ucb(i)
            self.selectedActions.append(idx)
            #print('select arm: ', idx)
            self.sample(idx)
            #self.num_played[idx] += 1
            #self.regret(idx, reward)
            self.regret(i)

class UCB_os_comb(UCB_discrete):
    """ucb os policy for comb of AbsGau and Exp
    """
    def __init__(self, env, medians, num_rounds, 
                 est_var, hyperpara, evaluation):
        """
        est_var: whether estimate lower bound of hazard rate
        """
        super().__init__(env, medians, num_rounds, 
                 est_var, hyperpara, evaluation)

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
            if self.est_var:
                #TODO
                L = 0.8
            else:
                L = 1.4
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
                 est_var, hyperpara, evaluation):
        super().__init__(env, medians, num_rounds, 
                 est_var, hyperpara, evaluation)

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
            if self.est_var:
                theta = np.sqrt(1.0/self.var_estimate(arm))
            else:
                theta = 1.0/self.env[arm].para
            v_t = 4.0 /( t_i * theta**2)
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
                 est_var, hyperpara, evaluation):
        super().__init__(env, medians, num_rounds, 
                 est_var, hyperpara, evaluation)

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
            if self.est_var:
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
                 est_var, hyperpara, evaluation):
        super().__init__(env, medians, num_rounds, 
                 est_var, hyperpara, evaluation)

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
            b = np.sqrt(2* self.alpha * np.log(t)/len(reward))
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

    def regret(self):
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

#------------------------------------------------------------------------------------

