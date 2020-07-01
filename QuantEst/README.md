# Stochasic Gradient Descent (SGD) for Quantile Estimation

## Project description

In this project, we explore the usage of SGD on quantile estimation on data streams. Generally, the outcome of the research is:

- Our results: This research shows that the SGD algorithm is an empirically effective method for quantiles estimation.
- Experiments of SGD:
  -  Different settings on input data streams (data size, data distribution, data sequence)
  -  Different settings on the SGD algorithms (step size)
- Extension of SGD:
  - Step size adaptation:
    - SAG for a faster convergence
    - DH-SGD for a smaller fluctuation after convergence
  - Multi-quantile estimation:
    - We looked into other people's work: shiftQ and extended P2
    - We have yet done any improvement on SGD

## Structure of the folder


1. `code/`: All quantile estimation experiments. It contains code for the experiments of SGD and other algorithms

     - `.ipynb` files: main code to run the experiment
     - `.py`  files: the code that are used as libraries for the main running files.


2. `thesis/`: The Latex files for my honours thesis.

3. `Q_init/`: Not included in the final thesis results.
   - A bit of research that shows how a initial value of quantile estimate might help with the estimation accuracy.

4. `anomaly_detection/`: Not included in the final thesis results.
   - Contains data from XPlane and an ipynb file that tries to build anomaly data according to the paper.