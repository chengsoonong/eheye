
# Meeting logs

## Week 4
Over the week I tried the DNABERT paper promoter prediction task with a non-fine-tuned model (so only pretrained) and the accuracy was \~0.5, so in this case fine-tuning does improve model performance. I then pre-processed the yeast paper data into the required format for DNABERT and without changing the hyperparameters tried fine-tuning the model and seeing the results. The model was only run once just to ensure it ran with no errors, luckily no errors but the pearson and spearman correlation coefficient was 0 so there is a bit of digging and debugging to do to figure out why. All the DNABERT paper downstream tasks were classification tasks so I had to edit some of the files in DNABERT/src to add a _Transformer.DataProcessor_ class that could deal with a regression 
task.

Mengyan has told me to start looking into the fine-tuning of the DNABERT/BERT model to see what changes need to be made when doing regression vs classification as a starting place to fixing the model. I will also try fine-tuning on the whole dataset as in my test run I attempted to match the dataset size used by DNABERT (around 60,000 whereas the yeast data is about 300,000+), however if it is too slow I will need to figure out a way to get a subset of the data (I think the data is organised in a certain way and not random).

We decided for me to write up a week-to-week plan of what results I have/where I should be by week 8, so I will push that to git when that is ready. 

Next week, I will present the DNABERT paper to Mengyan in a 15-20 minute presentation to ensure I understand it well, then I will present the yeast paper in the following week. At the same time, I will be trying to get the model to fine-tune with the yeast data and produce reasonable results (will also need to implement a different evaluation metric, ie R^2 or RMSE as huggingface just provides pearson and spearman correlation coefficients as a metric for regression). Then I will be able to produce some meaningful plots hopefully.

As a back-up if the regression task is not working, we may just convert it into a classification task so I have some results to present at the conclusion of the project (High/Medium/Low expression level).

## Week 5
This week I fixed some formatting errors with the way I preprocessed the yeast data for DNABERT. Successfully ran the fine-tuning with no errors after a little bit of debugging (DNABERT run_finetune.py had some flexibility to allow for regression tasks but some code was missing). I added the _sklearn.metrics.r2_score_ metric to the evaluation as the yeast paper uses R2 score.

After the code ran without errors I noticed that the prediction always saved an empty numpy array, so after some digging around in the script I found that they were only saving the prediction probabilities so I had to do a little more editing to make it work for regression and instead output the model prediction and not the model prediction passed through a softmax function. 

I was able to run the fine-tuning of the DNABERT model on the yeast data (PGD promoters) two times with different values for _adam_epsilon_ and see that the evaluation scores were improving with each iteration, which is promising. However, the scores so far are quite poor so some hyperparameter tuning will be required (R2 of 0.444 and 0.420).

This weeks meeting I presented the DNABERT paper to Mengyan. Some things the three of us discussed:
- There are 4 tasks, classification on yeast DNA and human DNA and regression on yeast DNA and human DNA.
- We are missing the negative class for classification on the yeast data and the negative class for regression on the human data.
- Hopefully I have some meaningful results with the yeast data soon so we can make a decision on what to do next (by the mid-semester break)

Next week I will present the yeast promoter paper.

### Answers to the questions Mengyan had:

__Why separate TATA and non-TATA promoters?__
I must have confused myself reading the paper but the fine-tuning of DNABERT-Prom-300 and DNABERT-Prom-core used a dataset that included both TATA and non-TATA promoters. They just specify in their Supplementary Materials that they had to construct the negative set of this dataset separately for the TATA and non-TATA promoters. 

> "We constructed the negative set separately for TATA and non-TATA promoters ... We trained our model using TATA and non-TATA core promoters altogether while predict separately on TATA and non-TATA datasets."

__More on the differences between BERT and DNABERT__
Besides a different tokenization and masking method (contiguous) I am unable to discern more obvious differences between the two models. I will have to spend more time with DNABERT and read up more on BERT in order to see differences on my own.

__DNABERT pretraining data__
I mentioned how they created their pretraining data but forgot exactly what they meant when they said non-overlapping splitting and random sampling. Quoted from the Supplementary Materials:
> "Since human genomes are much longer than 512, we used two methods to generate training data. First, we directly split a complete human genome into non-overlapping sub-sequences. Second, we randomly sampled sub-sequences from a complete human genome. The length of each sub-sequence lies in the range of 5 and 510. Specifically, with a 50% probability, we set the length of a sub-sequence as 510.  With another 50% probability, we set its length as a random integer between 5 and 510.  We regarded each sub-sequence as an independent sequence to train the model."