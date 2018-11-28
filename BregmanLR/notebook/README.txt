The demo requires Python 3, numpy, scipy, matplotlib, scikit-learn and CVXPY. 

To set up a conda environment to run the BregmanLR Demo:

1) Enter this into a terminal opened in the folder containing demo_requirements.txt:
 
$ conda create --name demo_env --file demo_requirements.txt
$ source activate demo_env
$ jupyter notebook

2) If for some reason 1) failed, try this instead:

$ conda create --name demo_env python=3.6 scipy scikit-learn=0.20.1 numpy matplotlib

$ conda install -n demo_env -c conda-forge lapack
$ conda install -n demo_env -c cvxgrp cvxpy

$ jupyter notebook