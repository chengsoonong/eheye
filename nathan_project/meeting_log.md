# Meeting logs

## Week 4
Over the week I tried the DNABERT paper promoter prediction task with a non-fine-tuned model (so only pretrained) and the accuracy was ~0.5, so in this case fine-tuning does improve model performance. I then pre-processed the yeast paper data into the required format for DNABERT and without changing the hyperparameters tried fine-tuning the model and seeing the results. The model was only run once just to ensure it ran with no errors, luckily no errors but the pearson and spearman correlation coefficient was 0 so there is a bit of digging and debugging to do to figure out why. All the DNABERT paper downstream tasks were classification tasks so I had to edit some of the files in DNABERT/src to add a _Transformer.DataProcessor_ class that could deal with a regression task.

Mengyan has told me to start looking into the fine-tuning of the DNABERT/BERT model to see what changes need to be made when doing regression vs classification as a starting place to fixing the model. I will also try fine-tuning on the whole dataset as in my test run I attempted to match the dataset size used by DNABERT (around 60,000 whereas the yeast data is about 300,000+), however if it is too slow I will need to figure out a way to get a subset of the data (I think the data is organised in a certain way and not random).

We decided for me to write up a week-to-week plan of what results I have/where I should be by week 8, so I will push that to git when that is ready. 

Next week, I will present the DNABERT paper to Mengyan in a 15-20 minute presentation to ensure I understand it well, then I will present the yeast paper in the following week. At the same time, I will be trying to get the model to fine-tune with the yeast data and produce reasonable results (will also need to implement a different evaluation metric, ie R^2 or RMSE as huggingface just provides pearson and spearman correlation coefficients as a metric for regression). Then I will be able to produce some meaningful plots hopefully.

As a back-up if the regression task is not working, we may just convert it into a classification task so I have some results to present at the conclusion of the project (High/Medium/Low expression level).

## Week 5