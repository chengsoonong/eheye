# Applying deep learning to gene promoters (Nathan Hu)

Code submitted as part of my S1 2021 project.

All code can be found in 'code' directory.

Organised in two jupyter notebooks: 

- `train_pipelines.ipynb`:
	Can be used to train each pipeline (one-hot+CNN, DNABERT+Dense or DNABERT+CNN) on yeast or human promoters from scratch.

- `results.ipynb`:
	Can be used to load trained models from the project and reproduce the results (models found in example/trained_models).


## Step 1: Setting up conda environment

This environment set-up was done on win64 so if there are some issues, different channels may need to be added in the `environment.yml` file.

A new anaconda environment will be created using `environment.yml` which will install pytorch=1.8.1, transformers=4.6.1 and python=3.6. First open a terminal and change to the directory 'code'

```
	conda env create -f environment.yml
	conda activate nathan-project
```

## Step 2: Open notebooks

With the environment succesfully installed open jupyter by running

```
	jupyter notebook
```

Then open either `train_pipelines.ipynb` or `results.ipynb`.


## Step 3: Downloading trained models and data

Download the three folders `data`, `dnabert_model_base` and `trained_models` from [here](https://drive.google.com/drive/folders/1O4B3GWgbR6ooU0y-EI9zo7H7y6fkmzOe?usp=sharing) and place in 'example/', such that the directory 'example' will now include 3 folders: data, dnabert_model_base and trained_models. This will allow you to train DNABERT models from scratch using `train_pipelines.ipynb`, as well as reproducing the results using `results.ipynb`.

The models are named in the following way:
- Pretrained DNABERT: This is the pretrained DNABERT model downloaded from [DNABERT](https://github.com/jerryji1993/DNABERT) and can be used to train the DNABERT pipelines from scratch. The relevant files are located within the directory `dnabert_model_base`.
- One-hot+CNN: This is a pipeline that was trained during the project. There is a yeast promoter (`one-hot_yeast.pth`) and human promoter version (`one-hot_human.pth`).
- DNABERT+Dense: This is a pipeline that was trained during the project. There is a yeast promoter (`dnabert-dense_yeast.pth`) and human promoter version (`dnabert-dense_human.pth`).
- DNABERT+CNN: This is a pipeline that was trained during the project. There is a yeast promoter (`dnabert-cnn_yeast.pth`) and human promoter version (`dnabert-cnn_human.pth`).

The data includes yeast (ZEV) promoters (split into train, val, test) and human promoters (split into train, val).

### Explanation of other files

- `dnabertcnn.py`:
	Used when loading in DNABERT+CNN.
- `dnabertdense.py`:
	Used when loading in DNABERT+Dense.
- `tokenization_dna.py`:
	As Ji et al. create a custom tokenizer that maps kmer tokens to a token id, this had to be included.
- `utils.py`:
	Used to declutter the two notebooks by containing all the main classes and functions used/written in the project