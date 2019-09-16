from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict
#from qreg import QRegressor

# Version: 11/08/2019
# True/Estimate the lower bound of hazard rate L
# True L is calculated by f(0)/ (1 - F(0))


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
    
    @abstractmethod
    def sd_bound(self):
        """return sub-optimal draws bound list"""
        
    @abstractmethod
    def r_bound(self):
        """return regret bound list"""

class H_UCB(UCB_discrete):
    """Hazard UCB class for UCB algorithms of finite number of arms.
    """
    def __init__(self, env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation):
        """
        Arguments
        ------------------------------------
        true_L_list: list
            sequence of true L (lower bound of hazard rate)

        hyperpara: list of  
            alpha: eps = alpha * log t,
            beta: balance empirical median and cw, 
            L_est_thr: L estimate threshold
        """
        super().__init__(env, medians, num_rounds, 
                est_flag, hyperpara, evaluation)
        self.true_L_list = []

    def init_L(self):
        """Initialise the true_L_list for the use of true L,
        where L is the lower bound of hazard rate.
        L = f(0)/ (1 - F(0))
        """
        for i in range(len(self.env)):
            
            my_env = self.env[i]
            #if 'pdf' in dir(my_env):
            #print(hasattr(my_env, 'pdf'))
            if hasattr(my_env, 'pdf'):
                # if pdf and cdf is defined
                x = 0
                L = my_env.pdf(x)/ (1- my_env.cdf(0))  
            else:
                L = my_env.L_estimate(self.hyperpara[-1])

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
            L = len(sorted_data[sorted_data <= self.hyperpara[-1]])/len(sorted_data)
            if L  == 0:
                L = 0.1
            return L
        else:
            # true L = f(0)/ (1 - F(0))
            return self.true_L_list[arm_idx]
    
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
        alpha, beta = self.hyperpara[:2]
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

class UCB1(UCB_discrete):
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
            cw = np.sqrt(2* self.hyperpara[0] ** 2 * np.log(t)/len(reward))
            ucbs.append( emp_mean + cw)
            if print_flag:
                print('For arm ', arm, ' Mean: ', emp_mean, ' cw: ', cw)
        assert len(ucbs) == len(self.env)
        #print('Choose policy: ', np.argmax(ucbs))
        return np.argmax(ucbs)
    
    def sd_bound(self):
        bounds = []
        return bounds

    def r_bound(self):
        bounds = []
        return bounds

class UCB_V(UCB_discrete):
    """Implement for UCB-V algorithm
    Policy:
    argmax emp_mean + confidence width (cw)
        cw = np.sqrt(2 * emp_var * eps/len(reward)) 
                + beta * 3 * b * eps/len(reward)
        eps = alpha * np.log(t)
    """
    def __init__(self, env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation):
        super().__init__(env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation)
        self.alpha, self.beta, self.b = self.hyperpara

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
            emp_var = np.var(reward)
            eps = self.alpha * np.log(t) 
            cw = np.sqrt(2 * emp_var * eps/len(reward)) \
                + self.beta * 3 * self.b * eps/len(reward)
            ucbs.append( emp_mean + cw)
            if print_flag:
                print('For arm ', arm, ' Mean: ', emp_mean, ' cw: ', cw)
        assert len(ucbs) == len(self.env)
        #print('Choose policy: ', np.argmax(ucbs))
        return np.argmax(ucbs)
    
    def sd_bound(self):
        bounds = []
        return bounds

    def r_bound(self):
        bounds = []
        return bounds

class MV_LCB(UCB_discrete):
    """Implement for MV_LCB algorithm
    Policy:
    argmax MV - cw
        MV = emp_var - rho * emp_mean
        cw = (5 + rho) * np.sqrt(np.log(1/theta)/ len(reward))
    """
    def __init__(self, env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation):
        super().__init__(env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation)
        self.rho, self.beta = self.hyperpara
        self.theta = 1/(num_rounds ** 2)
        self.bestarm = np.argmin(self.medians)

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
            emp_var = np.var(reward)
            MV = emp_var - self.rho * emp_mean
            cw = (5 + self.rho) * np.sqrt(np.log(1/self.theta)/ len(reward))
            ucbs.append( MV- self.beta * cw)
            if print_flag:
                print('For arm ', arm, ' MV: ', MV, ' cw: ', cw)
        assert len(ucbs) == len(self.env)
        #print('Choose policy: ', np.argmax(ucbs))
        return np.argmin(ucbs)
    
    def sd_bound(self):
        bounds = []
        return bounds

    def r_bound(self):
        bounds = []
        return bounds

class MARAB(UCB_discrete):
    """Implement for MARAB algorithm
    Policy:
    argmax CVaR - cw
        cw = beta * np.sqrt(np.log(int(t * alpha) + 1)/ (int(len(reward) * alpha)+1) )
    """
    def __init__(self, env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation):
        super().__init__(env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation)
        self.alpha, self.beta = self.hyperpara

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
            n_alpha = int(self.alpha * len(reward)) + 1
            cvar = np.mean(sorted(reward, reverse= True)[: n_alpha])
            cw = self.beta * np.sqrt(np.log(int(t * self.alpha) + 1)/ n_alpha )
            ucbs.append( cvar- cw)
            if print_flag:
                print('For arm ', arm, ' cvar: ', cvar, ' cw: ', cw)
        assert len(ucbs) == len(self.env)
        #print('Choose policy: ', np.argmax(ucbs))
        return np.argmax(ucbs)

    
    def sd_bound(self):
        bounds = []
        return bounds

    def r_bound(self):
        bounds = []
        return bounds

class Exp3():
    # implementation based on https://github.com/j2kun/exp3
    def __init__(self, env, medians, num_rounds, 
                 est_flag, hyperpara, evaluation, 
                 rewardMin = 0, rewardMax = 20):
        """
        Arguments
        ------------------------------------
        hyperpara: gamma, an egalitarianism factor

        weights: list
            sequence of each actions' weight
        n_arm: number of arms
        """
        self.env = env
        self.num_rounds = num_rounds
        self.medians = medians
        self.gamma, self.rewardMin, self.rewardMax = hyperpara
        self.evaluation = evaluation

        self.bestarm = np.argmax(medians)
        self.sample_rewards = defaultdict(list)
        self.n_arms = len(self.env)
        self.weights = [1.0] * self.n_arms

        self.cumulativeRegrets = []
        self.suboptimalDraws = []
        self.bestDraws = []

    def draw(self, t):
        """Compute upper confidence bound 

        Parameters
        --------------------------------
        t: int
            the number of current round

        Return
        --------------------------------
        the index of arm with the maximum ucb
        """
        choice = np.random.uniform(0, sum(self.weights))
        #print(sum(self.weights))

        choiceIndex = 0

        for weight in self.weights:
            choice -= weight
            if choice <= 0:
                return choiceIndex

            choiceIndex += 1

    def distr(self):
        theSum = float(sum(self.weights))
        return tuple((1.0 - self.gamma) * (w / theSum) + (self.gamma / self.n_arms) for w in self.weights)

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
        return reward

    def play(self):
        """Simulate Exp3 algorithms for specified rounds.
        """ 
        
        for i in range(self.num_rounds):
            probabilityDistribution = self.distr()
            choice = self.draw(probabilityDistribution)
            theReward = self.sample(choice)
            scaledReward = (theReward - self.rewardMin) / (self.rewardMax - self.rewardMin) # rewards scaled to 0,1

            estimatedReward = 1.0 * scaledReward / probabilityDistribution[choice]
            self.weights[choice] *= np.exp(estimatedReward * self.gamma / self.n_arms) # important that we use estimated reward here!
            self.evaluate(i)
        #return choice, theReward, estimatedReward

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

    def sd_bound(self):
        """return sub-optimal draws bound list"""
        return self.suboptimalDraws
        
    def r_bound(self):
        """return regret bound list"""
        return self.cumulativeRegrets

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