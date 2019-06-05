This folder contains a Jupyter notebook which contains implementations for the models discussed
in the report and produces evaluations of those models on 4 compositional datasets. 

The notebook relies on 3 other files:
error_metrics.py contains implementations of the metrics with which we evaluate our models
compositional_datasets.py and gdp_wiki.csv allow us to retrieve our 4 datasets from within
the notebook in a convenient manner.

To be able to run the code:
1) Install the Anaconda Python Distribution

2) Create a new environment with the following terminal commands:
>>> conda create -n coda_env python=3.6 scipy numpy pandas scikit-bio

If you run into issues running the notebook, specify the following version numbers:
>>> conda create -n coda_env python=3.6 scipy=1.2.1 numpy=1.16.2 pandas=0.22.0 scikit-bio=0.5.4

3) Activate that environment with the command:
>>> activate coda_env

4) Root a terminal session inside the folder and launch Jupyter with the command:
>>> jupyter notebook

5) Open COMP8755_Notebook.ipynb and run all cells. 


All code submitted is completely my own work. 
