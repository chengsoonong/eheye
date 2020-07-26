## Upper Confidence Bounds (UCB)

The upper confidence bounds (UCB) algorithms are proposed to balance the exploration and exploitation trade-off dilemma in multi-armed bandits problems. This folder shows ideas about considering quantiles or medians as summary statistics for reward distributions. Two ideas are contained:
- Discrete case (M_UCB):
This folder contains code for the work: median-based bandits for unbounded rewards. We assume the arms are in discrete space and independent, using medians to design the UCB type policy. 


- Q-BAI: quantile best arm identification.

- Continuous case (Q_UCB):
This folder contains code for the arms that are in continuous space and dependent. The idea is to explore the quantiles based on the GPUCB algorithm. 
