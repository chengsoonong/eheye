# Median-based Bandits for Unbounded Rewards

This folder includes three subfolders:

* codes: 
    - Environment: implements the bandit environments (arm reward distributions)
    - UCB_discrete: implements bandit algorithms (Our policy M-UCB; baseline algorithms: U-UCB, Epsilon-greedy, Median-of-means-UCB, Exp3)
    - SimulatedGames: Simulate experiments
    - plots: plot methods for expected sub-optimal draws

* notebooks:
    - M_UCB_Simulations-ICML: simuated experiments in the main paper
    - M_UCB_Clinical_ICML: clinical experiments in the main paper
    - M_UCB_analysis: supplementary experiments in the appendix A

* Data: clinical datasets

To run the experiments, open notebook, restart and run all. 