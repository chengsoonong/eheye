# DNABERT+Dense model

from transformers import BertForSequenceClassification, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class DNABERTConvNet(BertForSequenceClassification):
    """
        Because DNABERT+CNN and DNABERT+Dense were trained with the same module name (DNABERConvNet) the trained models which
        were saved expect to be loaded using the same class. Therefore, both had to be seperated when loading the models that
        were trained for the project. 

        ie. training from scratch using train_pipelines.ipynb will not have this problem as the module subclass used for training
        takes as input an 'add_cnn' argument.
    """
    def __init__(self, dnabert, config, n_outputs):
        super().__init__(config)
        self.dnabert = dnabert

        self.output = nn.Linear(768, n_outputs)

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
        
        x = self.output(x)

        return x