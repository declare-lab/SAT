import torch
import torch.nn as nn
from torch.nn.functional import normalize
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertLayer, BertEmbeddings, BertPooler

class TextClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.main_linear = nn.Sequential(
            # nn.Linear(self.bert_model.config.hidden_size, config.hidden_size),
            nn.Linear(self.bert_model.config.hidden_size, config.num_classes),
            # nn.Tanh(),
            # nn.Linear(config.hidden_size, config.num_classes)
        )
        self.aux_linear = nn.Sequential(
            # nn.Linear(self.bert_model.config.hidden_size, config.hidden_size),
            nn.Linear(self.bert_model.config.hidden_size, config.num_augs),
            # nn.Tanh(),
            # nn.Linear(config.hidden_size, config.num_augs)
        )

    def forward(self, inputs, mode='main'):
        outputs = self.bert_model(**inputs)
        outputs = outputs.pooler_output
        if mode == 'main':
            return self.main_linear(outputs) 
        elif mode == 'aux':
            return self.aux_linear(outputs)
        elif mode == 'both':
            return self.main_linear(outputs), self.aux_linear(outputs)

class TextClassifierNoAux(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.main_linear = nn.Sequential(
            nn.Linear(self.bert_model.config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.num_classes)
        )

    def forward(self, inputs, return_hidden=False):
        outputs = self.bert_model(**inputs)
        outputs = outputs.pooler_output
        return self.main_linear(outputs), outputs if return_hidden else None

class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(768, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.num_augs)
        )
    
    def forward(self, inputs):
        return self.model(inputs)
    

class EmbeddingScorer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj_raw = nn.Sequential(
            nn.Linear(768, config.scorer_hidden_size),
            nn.Tanh(),
            nn.Linear(config.scorer_hidden_size, config.scorer_hidden_size)
        )
        self.proj_aug = nn.Sequential(
            nn.Linear(768, config.scorer_hidden_size),
            nn.Tanh(),          
            nn.Linear(config.scorer_hidden_size, config.scorer_hidden_size)
        )
    
    def forward(self, x, x_aug):
        # return (normalize(self.proj_raw(x)) * normalize(self.proj_aug(x_aug))).sum(dim=-1, keepdim=True)    # (bs, 1)
        return (normalize(self.proj_raw(x)) * normalize(self.proj_aug(x_aug))).sum(dim=-1, keepdim=True)