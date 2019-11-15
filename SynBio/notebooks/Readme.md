This folder shows experiments for SynBio with machine learning algorithms.

# Regression

- Evaluate based on the RMSE score (the smaller the better). 
- Labels are min-max normalized (between 0 and 1).
- Regression models: kernel ridge regression; Gaussian process regression.
- Kernels: 
  - Kernels from sklearn library: DotProduct; RBF
  - [String kernels](https://dx.plos.org/10.1371/journal.pcbi.1000173): spectrum, mixed spectrum, weighted degree, weighted degree with shifting

### Experiments

[Regression for RBS - Predict FC](https://github.com/chengsoonong/eheye/blob/master/SynBio/notebooks/Regression_RBS_FC.ipynb):
Kernel ridge regression with one-hot embedding with DotProduct is the best (in terms of test RMSE ~0.15).

[Regression for RBS - Predict TIR](https://github.com/chengsoonong/eheye/blob/master/SynBio/notebooks/Regression_RBS_TIR.ipynb):
Kernel ridge regression with label embedding with WD(shift) Kenel is the best (in terms of test RMSE ~ 0.23).

With cross-validation:
[Regression for RBS - Predict F - CV](https://github.com/chengsoonong/eheye/blob/master/SynBio/notebooks/Regression_RBS_FC%20_CV.ipynb)


# Recommendation for sequentail experiemntal desgin: Multi-armed Bandits

- Model: GP-UCB
- Comparison: Random selection
- Evaluation metric: expected cumulative regrets

### Experiments

[Recommend RBS sequences FC](https://github.com/chengsoonong/eheye/blob/master/SynBio/notebooks/Recommend%20RBS%20sequences%20FC.ipynb)  
[Recommend RBS sequences TIR](https://github.com/chengsoonong/eheye/blob/master/SynBio/notebooks/Recommend%20RBS%20sequences%20TIR.ipynb)

For both cases, GP-UCB has smaller cumulative regrets compared with random selection.

Interesting facts:
  - For TIR, regression is worse, but recommendations with ucb are better. (potential bugs?)
  - regret is in linear rate, rather than log rate (which is not ideal?)
  
TODO:
  - Recommend more than one arm once (subset selection)
  - PWM calculation
  - Unsupervised regresentation 
 
