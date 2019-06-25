# Synthetic Biology with GPUCB

The goal is to use machine learning algorithms (e.g. GPUCB) to identify the most probable combination of ribosome-binding site (RBS) which gives the best protein we need. 

It is important to have similarities among different combinations of RBS (i.e. similar RBS results in similar FC score) to do some ML on that, so [independence test](https://github.com/chengsoonong/eheye/blob/master/SynBio/RBS_Independence_Test.png) is done, which shows similar RBS does tend to have similar FC score.

## Content
1. Toy data comes from https://github.com/synbiochem/opt-mva
2. [Regression](https://github.com/chengsoonong/eheye/blob/master/SynBio/Regression%20on%20SynBio.ipynb)
3. [GPUCB](https://github.com/chengsoonong/eheye/blob/master/SynBio/gpucb_bio.ipynb)
