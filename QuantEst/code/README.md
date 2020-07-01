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
- `Output_data_generation.py`: save the quantile estimation results into `.txt` files.
- `Plot.py`: Draw figures from quantile estimation records.

#### Output structure

The quantile estimation files and the plots files will be saved in the following way:
```
/Experiment_results
│
├── SGD                             // Method SGD
│   ├── distro                      // different distributions
│   │   │
│   │   ├── exp_overview.txt        // distro is exp
│   │   ├── exp_q_batches.txt
│   │   ├── exp_q_true.txt
│   │   ├── exp_q_est_proc.txt
│   │   ├── exp_q_est_res.txt
│   │   ├── exp_q_est_proc.txt
│   │   ├── exp_q_E.txt
│   │   │
│   │   ├── exp_proc.png
│   │   ├── exp_res.png
│   │   ├── exp_error.png
│   │   │
│   │   ├── mix_overview.txt        // distro is mix
│   │   ├── mix_q_batches.txt
│   │   ├── mix_q_true.txt
│   │   ├── mix_q_est_proc.txt
│   │   ├── mix_q_est_res.txt
│   │   ├── mix_q_est_proc.txt
│   │   ├── mix_q_E.txt
│   │   │
│   │   ├── mix_proc.png
│   │   ├── mix_res.png
│   │   ├── mix_error.png
│   │   │
│   │   └─ ...                      // other distributions
│   │   
│   ├── data_size                   // different data sizes
│   │   ├── 100_overview.txt        // data size is 100
│   │   ├── ...
│   │   ├── 1000_overview.txt
│   │   └── ...
│   │
│   └── ...                         // different other settings
│
├── SAG                             // Method SAG
│   ├── distro
│   ├── data size
│   └-- ...
│
├── ...
```
---

### Run the code

Run all the cells in `Experiment_final.ipynb`, which includes:

- creates empty folders
- generates input data streams
- creates quantiles estimation record files
- plots from the quantile estimation records

You can change the settings of `main_folder` to generate results from different algorithms.
**This part is still a bit messy, I'm going to change it soon**