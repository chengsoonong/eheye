# All the helper functions required to run the jupyter notebooks:
# - results.ipynb
# - train_pipelines.ipynb


import numpy as np 
import pandas as pd
import sys
import copy
from random import shuffle
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertConfig
from tokenization_dna import DNATokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

plt.rc('font', size=15)
plt.rc('axes', labelsize=20)

DNA = ['A','C','G','T']

def one_hot_encode(sequences):
    """
    Function provided by Kotopka and Smolke. (https://github.com/smolkelab/promoter_design)
    """
    dna_dict = {}
    for i,q in enumerate(DNA):
        dna_dict[q] = i
    ans = [_encode_one(q, dna_dict) for q in tqdm(sequences, desc='one-hot encoding sequences', file=sys.stdout, leave=True, position=0)]
    return np.squeeze(ans)

def _encode_one(seq, dna_dict):
    """
    Function provided by Kotopka and Smolke. (https://github.com/smolkelab/promoter_design)
    """
    ans = np.zeros(shape=(len(seq), len(DNA)), dtype = 'int')
    for i,q in enumerate(seq):
        ans[i, dna_dict[q]] = 1
    return ans

class shifting_batch_generator():
    """
    Function provided by Kotopka and Smolke. (https://github.com/smolkelab/promoter_design)
    
    Iterator which will provide batched training examples during model training
    
    Examples will be from an augmented dataset where each sequence has been subsequenced by a sliding window.
    
    The sliding window will slide SHIFT amount(s) of steps.
    """
    def __init__(self, dataset_X, dataset_y, batch_size, shift):
        self.dataset_X = dataset_X
        self.dataset_y = dataset_y
        self.batch_size = batch_size
        self.shift = shift
        self.window_size = dataset_X.shape[1] - shift + 1
        tuple_gen = ((p,q) for p in range(self.dataset_X.shape[0]) for q in range(self.shift))
        self.tuples = [i for i in tuple_gen]
        self._regen_tuples()

    def _regen_tuples(self):
        self.curr_tuples = copy.copy(self.tuples)
        shuffle(self.curr_tuples)

    def __len__(self):
        return self.dataset_X.shape[0]

    def _get_batch(self):
        X_out = []
        y_out = []
        for i in range(self.batch_size):
            if len(self.curr_tuples) == 0:
                self._regen_tuples()
            sample_id, offset = self.curr_tuples.pop()
            X_out.append(self.dataset_X[sample_id,offset:offset+self.window_size,:])
            y_out.append(self.dataset_y[sample_id])

        return (np.stack(X_out,0), np.stack(y_out,0))

    def iter(self):
        while True:
            yield self._get_batch()
            
            
def seq2kmer(seq, k):
    """
    Function provided by Ji et al. (https://github.com/jerryji1993/DNABERT)
    Will convert given sequence into kmer.
    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers
            
            
def to_tensor(array, dtype=torch.float):
    """
    Simple function to convert numpy arrays to the appropriate tensor
    """
    return torch.as_tensor(array, dtype=dtype, device=torch.device(device))


def load_yeast_promoters(data_dir: str):
    train = pd.read_csv(data_dir+'zev_train.csv')
    val = pd.read_csv(data_dir+'zev_val.csv')
    test = pd.read_csv(data_dir+'zev_test.csv')

    print(f'Loaded yeast promoters dataset \n {len(train)} Training examples \n {len(val)} Validation and test examples \n')
    return train, val, test


def load_human_promoters(data_dir: str):
    train = pd.read_csv(data_dir+'human_train_nonkmer.csv')
    dev = pd.read_csv(data_dir+'human_dev_nonkmer.csv')

    print(f'Loaded human promoters dataset \n {len(train)} Training examples \n {len(dev)} Validation examples \n')
    return train, dev


class PromoterSequences(Dataset):
    """
    Subclass of PyTorch Dataset. Used for one-hot encoding pipelines.
    """
    def __init__(self, data):
        self.data = data
        self.shape = data[0].shape

    def __len__(self):
        return len(self.data[0])

    def shape(self):
        return self.shape.item()

    def __getitem__(self,idx):
        sequence = self.data[0][idx]
        label = self.data[1][idx]
        return (sequence, label)
    

class PromoterTokenizer(Dataset):
    """
    Subclass of PyTorch Dataset. Used for DNABERT pipelines (need to pass a tokenizer as input).
    """
    def __init__(self, data, tokenizer):
        self.data = data
        self.shape = data[0].shape
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data[0])

    def shape(self):
        return self.shape.item()

    def __getitem__(self,idx):
        sequence = self.tokenizer(self.data[0][idx]).convert_to_tensors('pt')
        label = self.data[1][idx]
        return (sequence, label)

class ConvNet(nn.Module):
    """
    One-hot+CNN
    """
    def __init__(self, n_outputs, in_c=1,size=256):
        super(ConvNet, self).__init__()
        FILTERS = 128

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_c, FILTERS, kernel_size=8,stride=1,padding=(4,)),
            nn.BatchNorm1d(FILTERS),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(FILTERS, FILTERS, kernel_size=8,stride=1,padding=(4,)),
            nn.BatchNorm1d(FILTERS),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FILTERS*size, FILTERS),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(FILTERS, FILTERS)
        )
        self.output = nn.Linear(FILTERS, n_outputs)
    
    def forward(self, x):
        x = self.conv_block1(x) # 6 convolutional layers
        x = self.conv_block2(x)
        x = self.conv_block2(x)
        x = self.conv_block2(x)
        x = self.conv_block2(x)
        x = self.conv_block2(x)

        x = self.dense1(x) # 2 dense layers
        x = self.dense2(x)
        x = self.output(x)

        return x

class DNABERT(BertForSequenceClassification):
    """
    DNABERT+Dense or DNABERT+CNN (specified via 'add_cnn')
    
    If dense, it will pass the output of DNABERT directly to the linear layer. Otherwise it will pass the output
    of DNABERT to the same CNN as ConvNet
    """
    def __init__(self, dnabert, config, n_outputs, add_cnn):
        super().__init__(config)
        self.dnabert = dnabert
        self.add_cnn = add_cnn
        FILTERS = 128 if add_cnn else 768

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, FILTERS, kernel_size=3,stride=1),
            nn.BatchNorm1d(FILTERS),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(FILTERS, FILTERS, kernel_size=3,stride=1),
            nn.BatchNorm1d(FILTERS),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FILTERS*10, FILTERS)
        )
        self.dense2 = nn.Sequential(
            nn.Linear(FILTERS, FILTERS)
        )
        self.output = nn.Linear(FILTERS, n_outputs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        next_sentence_label=None,
  ):
        x = self.dnabert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]

        if self.add_cnn:
            x = self.conv_block1(x.unsqueeze(1))
            x = self.conv_block2(x)
            x = self.conv_block2(x)
            x = self.conv_block2(x)
            x = self.conv_block2(x)
            x = self.conv_block2(x)

            x = self.dense1(x)
            x = self.dense2(x)
        
        x = self.output(x)

        return x

    
def train_loop(model, dataloader, loss_fn, optimizer, pipeline: str, steps_per_epoch: int, task: str):
    """

    Args:
        dataloader - the training dataloader
        
        steps_per_epoch - important if dataloader is infinite generator (shifting_batch_generator)
        
        pipeline: which embedding is being used onehot/dnabert
        
        task: which promoter is being used yeast/human
       
    """
    size = len(dataloader)
    batch_loss = 0
    
    generator = dataloader if pipeline == 'dnabert' else dataloader.iter()

    model.train()
    for batch, (X, y) in tqdm(enumerate(generator), file=sys.stdout, leave=True, position=0, total=int(steps_per_epoch)):   
        
        if pipeline == 'dnabert':
            input_ids, attention_mask, token_type_ids = to_tensor(X['input_ids'],torch.long), to_tensor(X['attention_mask'],torch.long), to_tensor(X['token_type_ids'],torch.long)

        if pipeline == 'onehot':
            X = to_tensor(np.reshape(X, (X.shape[0], X.shape[2], X.shape[1])))
        
        y = to_tensor(y).unsqueeze(1) if task == 'yeast' else to_tensor(y, dtype=torch.long)

        # Compute prediction and loss
        pred = model(input_ids, 
                     attention_mask=attention_mask, 
                     token_type_ids=token_type_ids) if pipeline == 'dnabert' else model(X)
        
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()
    
        if batch == steps_per_epoch and pipeline == 'onehot':
            break

    batch_loss /= batch
    print(f" average loss: {batch_loss:>5f}")

def test_loop(model, dataloader, loss_fn, pipeline: str, task: str, SHIFT=4):
    """
    Perform evaluation on a test or validation set.
    """
    size = len(dataloader.dataset)

    score, test_loss, true, preds = 0, 0, [], []

    model.eval()
    with torch.no_grad():
        for batch, (X, y) in tqdm(enumerate(dataloader), file=sys.stdout, leave=True, position=0, total=size//dataloader.batch_size):
            
            if pipeline == 'dnabert':
                input_ids, attention_mask, token_type_ids = to_tensor(X['input_ids'],torch.long), to_tensor(X['attention_mask'],torch.long), to_tensor(X['token_type_ids'],torch.long)

            # Validation and test data are not shifted, so need to shorten each sequence to the correct length
            if pipeline == 'onehot':
                X = to_tensor(np.reshape(X[:][:,0:X.shape[1]-SHIFT+1], (X.shape[0], X.shape[2], X.shape[1]-SHIFT+1)))
            
            y = to_tensor(y).unsqueeze(1) if task == 'yeast' else to_tensor(y, dtype=torch.long)

            # Compute prediction and record loss
            pred = model(input_ids, 
                     attention_mask=attention_mask, 
                     token_type_ids=token_type_ids) if pipeline == 'dnabert' else model(X) 
            test_loss += loss_fn(pred, y).item()

            # To print a final score for the predictions. R2 for yeast and accuracy for human.
            if task == 'yeast':
                preds = np.append(preds, pred.cpu().detach().numpy())
                true = np.append(true, y.cpu().detach().numpy())
            if task == 'human':
                score += (pred.argmax(1) == y).type(torch.float).sum().item()
                preds = np.append(preds, pred.argmax(1).cpu().detach().numpy())
                true = np.append(true, y.cpu().detach().numpy())

    test_loss /= batch
    score = r2_score(true, preds) if task == 'yeast' else score / size
    metric = 'R2 score' if task == 'yeast' else 'Accuracy'
  

    print(f"\n{task.capitalize()} Promoters Validation set: \n Avg loss: {test_loss:>8f}, {metric}: {score:.3f} \n {size} Promoters evaluated. \n")

    return (true, preds)


def plot_predictions(true_labels, pred_labels, task):
    """
    Plots a prediction plot with the density of points
    """
    xy = np.vstack([true_labels,pred_labels])
    z = gaussian_kde(xy)(xy)

    idx = z.argsort()
    x, y, z = true_labels[idx], pred_labels[idx], z[idx]

    fig = plt.figure(figsize=(6,6))

    ax = fig.add_subplot(111)

    range = np.ptp(true_labels)*0.1
    left_lim = min(np.min(pred_labels), np.min(true_labels)) - range*2
    right_lim = max(np.max(pred_labels), np.max(true_labels)) + range*2

    plt.scatter(x,y, s=50, alpha=0.2, c=z, marker='.')
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black',ls='--',alpha=0.5)
    ax.annotate(f'$R^2 = {r2_score(true_labels, pred_labels).round(2)}$',(0.65,0.1),xycoords='axes fraction')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(left_lim, right_lim)
    ax.set_ylim(left_lim, right_lim)

    plt.colorbar(fraction=0.05,aspect=5)
    plt.xlabel('Activity (pZEV-uninduced)')
    plt.ylabel('Predicted activity')
    plt.title(f'One-hot+CNN for {task} promoters')

    plt.show()

def load_dnabert(path, n_outputs):
    """
    Loads the pretrained DNABERT model (configuration, tokenizer and base model)
    """
    config = BertConfig.from_pretrained(path,num_labels=n_outputs)
    tokenizer = DNATokenizer.from_pretrained(path)
    dnabert = BertForSequenceClassification.from_pretrained(path, config=config).base_model

    return config, tokenizer, dnabert

def load_data(task='human', pipeline='onehot'):
    """
    Used in results.ipynb to load validation data
    """
    
    n_outputs = 1 if task == 'yeast' else 2

    if task == 'yeast':
        train, val, test = load_yeast_promoters('example/data/')
    if task == 'human':
        train, val = load_human_promoters('example/data/')


    if pipeline == 'onehot':
        valX, valy = one_hot_encode(val['Seq']), val['label'].values

        val_dataset = PromoterSequences((valX, valy))
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

        return val_loader

    if pipeline == 'dnabert':
        valX, valy = val['Seq'].apply(lambda x: seq2kmer(x,6)).values, val['label'].values

        config, tokenizer, dnabert_base = load_dnabert('example/dnabert_model_base', n_outputs)

        val_dataset = PromoterTokenizer((valX, valy), tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

        return val_loader, config, dnabert_base