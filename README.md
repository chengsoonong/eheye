# eheye

Research on online experiment design (ABS):

One of the key chanlleges for machine learning is to obtain high quality labels for data, which can be expensive and/or noisy. We consider two popular methods for online experiment design with limit labels: Active learning uses computational algorithms to suggest the most useful measurements, and one possible theoretical framework is by balancing exploration and exploitation in a function maximization setting.

## A. Active Learning

[acton experiments](https://github.com/chengsoonong/eheye/tree/master/acton_experiment): experiments for testing on [acton](https://github.com/chengsoonong/acton) package, which is an active learning Python library.

## B. Bandits problems

Multi-Armed Bandit (MAB) problems are well-studied sequential experiment design problems where an agent adaptively choose one option among several actions with the goal of maximizing the profit (i.e. minimizing the regret). 

[Upper confidence bounds](https://github.com/chengsoonong/eheye/tree/master/QuantUCB): Among all polices have been proposed for stochastic bandits with finitely many arms, a particular family called "upper confidence bound" algorithm has raised a strong interest. The upper confidence bound algorithm is based on the principle of optimism in face of uncertainty.

[Thompson sampling experiment](https://github.com/chengsoonong/eheye/tree/master/Thompson_sampling): Gaussian reward with 5-arm bandit simulator.

## S. Synthetic Biology

One application for active learning and bandits algorithms is the experiment design for [synthetic biology](https://github.com/chengsoonong/eheye/tree/master/SynBio). The goal is to use machine learning algorithms (e.g. GPUCB) to identify the most probable combination of ribosome-binding site (RBS) which gives the best protein we need.

## Other unsorted stuff

### [matrix factorization](https://github.com/chengsoonong/eheye/tree/master/matrix_factorazation): 
matrix factorization using MovieLens; tutorial for maxtrix factorization.

### Writing

### Ideas



