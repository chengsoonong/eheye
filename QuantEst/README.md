## Structure of the folder

1. `anomaly_detection/`: Contains data from XPlane and an ipynb file that tries to build anomaly data according to the paper.

2. `sgd_est_experiment/`: SGD quantile estimation experiment. It's currently very messy, I am trying to re-arrange everything so that we have:

     - `Experiment_final.ipynb` : Generates all the data for experiment, will be used for plotting.
     - `Experiment_plots.ipynb` : Generates the plots from data.
     - `Experiment_test.ipynb` : The test file to check `Experiemnt_final.ipynb` is correct (hopefully)
     - `Experiment_results/` : Generated when by `Experiment_final.ipynb`. Stores all the plots of comparison results, those will be used in the experiment document
         

3. `Q_init/`: A bit of research that shows how a initial value of quantile estimate might help with the estimation acuracy.

4. `thesis/`: The Latex draft files in the `drafts/`.

    - `Figures/`: All images inserted in latex
    - `SGD.tex`: Explain how SGD works for quantile estimation
    - `algorithm_equivalence.tex`: Explain why SGD and Frugal are 'equivalent' (they actually aren't)
    - `sgd_experiment`: The experiment on SGD estimation. Filled with plots
    - `nams.tex`: The file that includes most of the latex structual settings


5. `weekly_notes/`: The notes about weekly meeting notes or the records of my work. Basically nothing inside QAQ