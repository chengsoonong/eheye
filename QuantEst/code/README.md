## Code for the experiments

---

### File Descriptions

This folder contains experiment code and presentation code. The description of each files are:

#### Jupyter notebook files

- `Experiment_final.ipynb`: the main file that runs all the experiments.
- `Experiment_test.ipynb`: the testing files to check the data generation by `Dataset_generation.py`
- `Presentation plot.ipynb`: the plotting file for some figures in my thesis and the presentation

#### Python files

- `Dataset_generation.py`: Generates the input data streams
- Quantile estimation algorithms: 
  - `Method_sgd_frugal_adaptive.py`
  - `Method_sag.py`
  - `Method_p2.py`
  - `Method_shiftQ.py`
- `Quantile_procs.py`: the interface between the quantile estimation function and the different algorithms
- `Output_data_generation.py`: save the quantile estimation results into `.txt` files
- `Plot.py`: Draw figures from quantile estimation records.

---

### Run the code

Run all the cells in `Experiment_final.ipynb`, which includes:

- creates empty folders
- generates input data streams
- creates quantiles estimation record files
- plots from the quantile estimation records

You can change the settings of `main_folder` to generate results from different algorithms.