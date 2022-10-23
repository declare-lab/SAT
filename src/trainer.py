import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import TextClassifier, EmbeddingScorer, TextClassifierNoAux, Classifier
from utils import compare_and_gen_augchoicemask, move_to_cuda, AverageMeter
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import logging

class BaseTrainer():
    def __init__(self, config, loaders):
        self.config = config
        
        self.loaders = loaders
        self.num_epoch = config.num_epoch
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = TextClassifier(config)
        self._model.to(self._device)
        
        # hyperparams
        self.thr = config.thr
        self.lbd = config.lbd
        self.patience = config.patience
        
        self.losses = AverageMeter()
        
        aux_params = []
        main_params = [] 
        bert_params = []
        for n, p in self._model.named_parameters():
            if 'aux' in n:
                aux_params.append(p)
            elif 'bert' in n:
                bert_params.append(p)
            elif 'main' in n:
                main_params.append(p)
                
        self._aux_optimizer = getattr(optim, config.optimizer)(
            lr=config.lr_aux,
            params=aux_params
        )
        self._main_optimizer = getattr(optim, config.optimizer)(
            [
                    {'params': main_params, 'lr': config.lr_main},
                    {'params': bert_params, 'lr': config.lr_bert}
            ]
        )
        self._criterion = nn.CrossEntropyLoss()
        self._criterion_no_reduce = nn.CrossEntropyLoss(reduction='none')
    
    def train(self, loader_label, loader_unlabel, epoch):
        with tqdm(loader_label) as pbar:
            for batch_label in pbar:
                sents_l, sents_l_aug1, sents_l_aug2, labels = batch_label
                sents_l, sents_l_aug1, sents_l_aug2 = map(move_to_cuda, [sents_l, sents_l_aug1, sents_l_aug2])
                labels = labels.cuda()

                if self.config.dataset == 'imdb':
                    labels = labels + 1
                # step1: run and compare the outputs from two augmentations of the same labeled input
                preds1 = self._model(sents_l_aug1)  # (n, c)
                preds2 = self._model(sents_l_aug2)  # (n, c)
                aug_choice = compare_and_gen_augchoicemask(preds1, preds2, labels, metric=self.config.aux_metric).cuda()
                
                # step2: tune the auxiliary network using labeled inputs
                raw_preds, aug_preds = self._model(sents_l, 'both')
                loss_aug_choice = self._criterion(aug_preds, aug_choice)
                self._model.zero_grad()
                loss_aug_choice.backward(retain_graph=True)
                self._aux_optimizer.step()
                
                loss_raw = self._criterion(raw_preds, labels)
                self._model.zero_grad()
                loss_raw.backward()

                # step3: predict which augmentation is relatively weak on unlabeled inputs
                total_loss_aug = 0
                
                if epoch + 1 > self.config.warmup_epoch:
                    # only do choice inference after warmup ends
                    for _ in range(self.config.mu):
                        try:
                            batch_unlabel = next(generator)
                        except:
                            generator = iter(loader_unlabel)
                            batch_unlabel = next(generator)
                        
                        sents_u, sents_u_aug1, sents_u_aug2, _ = batch_unlabel
                        sents_u, sents_u_aug1, sents_u_aug2 = map(move_to_cuda, [sents_u, sents_u_aug1, sents_u_aug2])
                        
                        if self.config.method == 'fix_match':
                            aug_choices = torch.LongTensor([0]*32).cuda()
                        else:    
                            aug_choices = self._model(sents_u, 'aux').argmax(dim=-1).detach()    # (n, 1)
                        preds1, preds2 = self._model(sents_u_aug1), self._model(sents_u_aug2)   # (n,c), (n,c)
                        # these are logits
                        all_preds = torch.stack([preds1, preds2], dim=1) # (n, 2, c)
                        
                        # step4: calculate loss for both labeled and unlabeled sententces
                        bs = aug_choices.size(0)
                        aug_target_logits = all_preds[torch.arange(bs), aug_choices]
                        aug_target = F.softmax(aug_target_logits, dim=-1).detach()
                        
                        # hard label
                        aug_target_tensor, aug_target_indx = aug_target.topk(1, dim=-1)    # (n,), (n,)

                        # generate mask according to threshold and maximum probabilities
                        loss_mask = aug_target_tensor >  self.thr
                        # the augmentation logits not used for targets serve as preds
                        aug_pred_logits = all_preds[torch.arange(bs), 1-aug_choices]
                        
                        loss_aug_raw = self._criterion_no_reduce(aug_pred_logits, aug_target_indx.squeeze())
                        loss_aug = (loss_aug_raw * loss_mask).sum()/(self.config.mu * bs)
                        loss_aug *= self.config.lbd
                        total_loss_aug += loss_aug
                        loss_aug.backward()

                self._main_optimizer.step()
                
                pbar.set_description('Training epoch {:2d}: loss_labeled: {:.3f}, loss_aug: {:.3f}, loss_aug_choice: {:3f}'.format(epoch+1, loss_raw, total_loss_aug, loss_aug_choice))

    def train_base(self, loader_label, loader_unlabel, epoch):
        self.losses.reset()
        
        with tqdm(loader_label) as pbar:
            for (sents_l, _, _, labels) in pbar:
                sents_l = move_to_cuda(sents_l)
                labels = labels.cuda()
                if self.config.dataset == 'imdb':
                    labels = labels + 1
                
                raw_preds = self._model(sents_l)
                loss = self._criterion(raw_preds, labels)

                self._model.zero_grad()
                loss.backward()
                self._main_optimizer.step()
                self.losses.update(loss, raw_preds.size(0))
                pbar.set_description('Training epoch {:2d}: Loss = {:.4f}'.format(epoch+1, self.losses.val))

    def evaluate(self, loader, epoch=1, stage='val'):
        all_preds, all_labels = [], []
        self.losses.reset()
        
        with tqdm(loader) as pbar:
            for (sents, _, _, labels) in pbar:
                sents = move_to_cuda(sents)
                labels = labels.cuda()

                if self.config.dataset == 'imdb':
                    labels = labels + 1
                with torch.no_grad():
                    preds = self._model(sents)
                    all_preds.append(preds.argmax(-1))
                    all_labels.append(labels)
                pbar.set_description('{}ing'.format('validat' if stage=='val' else 'test'))
        
        all_preds = torch.cat(all_preds).detach().cpu().numpy()
        all_labels = torch.cat(all_labels).detach().cpu().numpy()
        f1_eval = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        return f1_eval, acc
    
    def run(self):
        best_test_f1 = 0.0
        best_test_acc = 0.0
        best_dev_f1 = 0.0
        patience = self.patience
        for epoch in range(self.num_epoch):
            # train
            self.train(self.loaders['train_loader_l'], self.loaders['train_loader_u'], epoch)
            # eval
            f1_macro, f1_micro = self.evaluate(self.loaders['dev_loader'], epoch)
            self.logger.info('********Dev results: Acc (MI-F1)={}*********'.format(f1_micro))
            if f1_micro > best_dev_f1:
                patience = self.patience
                self.logger.info('find new best model!')
                best_dev_f1 = f1_micro
                f1_macro, f1_micro = self.evaluate(self.loaders['test_loader'], epoch)
                self.logger.info('********Test results: F1 (MA)={}, Acc (MI-F1)={}*********'.format(f1_macro, f1_micro))
                if f1_micro > best_test_acc:
                    best_test_acc = f1_micro
                    best_test_f1 = f1_macro
            else:
                patience -= 1
                if patience == 0 or epoch == self.config.num_epoch - 1:
                    self.logger.info('No patience, best f1 is {}'.format(best_test_f1))
                    break
        self.logger.info('NUM_LABELD={}: lr={}, lr_bert={}, thr={}, lbd={}, mu={}: best_acc={}, best_MAF1={}\n'.format(self.config.num_labeled, self.config.lr_main, self.config.lr_bert, self.config.thr, self.config.lbd, self.config.mu, best_test_acc, best_test_f1))
        f = open('results_{}.txt'.format(self.config.dataset), 'a')
        f.write('NUM_LABELD={}: lr={}, lr_bert={}, thr={}, lbd={}, mu={}: best_MAF1={}, best_acc={}, best_f1={}\n'.format(self.config.num_labeled, self.config.lr_main, self.config.lr_bert, self.config.thr, self.config.lbd, self.config.mu, best_test_f1, best_test_acc, best_test_f1))
        f.close()
        exit()

    def run_base(self):
        best_test_f1 = 0.0
        best_dev_f1 = 0.0
        patience = self.patience
        for epoch in range(self.num_epoch):
            # train
            self.train_base(self.loaders['train_loader_l'], self.loaders['train_loader_u'], epoch)
            # eval
            f1_macro, f1_micro = self.evaluate(self.loaders['dev_loader'], epoch)
            self.logger.info('validating result (micro-f1):{}'.format(f1_micro))
            if f1_micro > best_dev_f1:
                patience = self.patience
                best_dev_f1 = f1_micro
                self.logger.info('find new best model!')
                f1_macro, f1_micro = self.evaluate(self.loaders['test_loader'], epoch, stage='test')
                self.logger.info('********Test results: Acc={}, F1={}************'.format(f1_micro, f1_macro))
                if f1_micro > best_test_f1:
                    best_test_f1 = f1_micro
            else:
                patience -= 1
                if patience == 0 or epoch == self.config.num_epoch - 1:
                    self.logger.info('No patience, best f1 is {}'.format(best_test_f1))
                    exit()

class ClfTrainer(BaseTrainer):
    def __init__(self, config, loaders):
        super(ClfTrainer, self).__init__(config, loaders)
        self._model = TextClassifierNoAux(config)
        self._classifier = Classifier(config)
        self._criterion_choice = nn.BCELoss()
        
        self._model.to(self._device)
        self._classifier.to(self._device)

        main_params = [] 
        bert_params = []
        for n, p in self._model.named_parameters():
            if 'bert' in n:
                bert_params.append(p)
            elif 'main' in n:
                main_params.append(p)
        
        self._aux_optimizer = getattr(optim, config.optimizer)(
            lr=config.lr_aux,
            params=self._classifier.parameters()
        )
        self._main_optimizer = getattr(optim, config.optimizer)(
            [
                    {'params': main_params, 'lr': config.lr_main},
                    {'params': bert_params, 'lr': config.lr_bert}
            ]
        )
        self._criterion = nn.CrossEntropyLoss()
        self._criterion_no_reduce = nn.CrossEntropyLoss(reduction='none')
    
    def train(self, loader_label, loader_unlabel, epoch):
        with tqdm(loader_label) as pbar:
            for batch_label in pbar:
                sents_l, sents_l_aug1, sents_l_aug2, labels = batch_label
                sents_l, sents_l_aug1, sents_l_aug2 = map(move_to_cuda, [sents_l, sents_l_aug1, sents_l_aug2])
                labels = labels.cuda()

                if self.config.dataset == 'imdb':
                    labels = labels + 1
                # step1: run and compare the outputs from two augmentations of the same labeled input
                preds1, _ = self._model(sents_l_aug1)  # (n, c)
                preds2, _ = self._model(sents_l_aug2)  # (n, c)
                aug_choice = compare_and_gen_augchoicemask(preds1, preds2, labels, metric=self.config.aux_metric).cuda()
                
                # step2: tune the auxiliary network using labeled inputs
                raw_preds, raw_hiddens = self._model(sents_l, return_hidden=True)
                aug_preds = self._classifier(raw_hiddens.detach())
                loss_aug_choice = self._criterion(aug_preds, aug_choice)
                # self._model.zero_grad()
                self._classifier.zero_grad()
                loss_aug_choice.backward(retain_graph=True)
                self._aux_optimizer.step()
                
                loss_raw = self._criterion(raw_preds, labels)
                self._model.zero_grad()
                loss_raw.backward()

                # step3: predict which augmentation is relatively weak on unlabeled inputs
                total_loss_aug = 0
                
                if epoch + 1 > self.config.warmup_epoch:
                    # only do choice inference after warmup ends
                    for _ in range(self.config.mu):
                        try:
                            batch_unlabel = next(generator)
                        except:
                            generator = iter(loader_unlabel)
                            batch_unlabel = next(generator)
                        
                        sents_u, sents_u_aug1, sents_u_aug2, _ = batch_unlabel
                        sents_u, sents_u_aug1, sents_u_aug2 = map(move_to_cuda, [sents_u, sents_u_aug1, sents_u_aug2])
                        
                        if self.config.method == 'fix_match':
                            aug_choices = torch.LongTensor([0]*32).cuda()
                        else:
                            _, hidden_u = self._model(sents_u, return_hidden=True)    
                            aug_choices = self._classifier(hidden_u.detach()).argmax(dim=-1).detach()
                            
                        preds1, _ = self._model(sents_u_aug1)
                        preds2, _ = self._model(sents_u_aug2)   # (n,c), (n,c)
                        
                        # these are logits
                        all_preds = torch.stack([preds1, preds2], dim=1) # (n, 2, c)
                        
                        # step4: calculate loss for both labeled and unlabeled sententces
                        bs = aug_choices.size(0)
                        aug_target_logits = all_preds[torch.arange(bs), aug_choices]
                        aug_target = F.softmax(aug_target_logits, dim=-1).detach()
                        
                        # hard label
                        aug_target_tensor, aug_target_indx = aug_target.topk(1, dim=-1)    # (n,), (n,)

                        # generate mask according to threshold and maximum probabilities
                        loss_mask = aug_target_tensor >  self.thr
                        # the augmentation logits not used for targets serve as preds
                        aug_pred_logits = all_preds[torch.arange(bs), 1-aug_choices]
                        
                        loss_aug_raw = self._criterion_no_reduce(aug_pred_logits, aug_target_indx.squeeze())
                        loss_aug = (loss_aug_raw * loss_mask).sum()/(self.config.mu * bs)
                        loss_aug *= self.config.lbd
                        total_loss_aug += loss_aug
                        loss_aug.backward()

                self._main_optimizer.step()
                
                pbar.set_description('Training epoch {:2d}: loss_labeled: {:.3f}, loss_aug: {:.3f}, loss_aug_choice: {:3f}'.format(epoch+1, loss_raw, total_loss_aug, loss_aug_choice))


    def evaluate(self, loader, epoch=1, stage='val'):
        all_preds, all_labels = [], []
        self.losses.reset()
        
        with tqdm(loader) as pbar:
            for (sents, _, _, labels) in pbar:
                sents = move_to_cuda(sents)
                labels = labels.cuda()

                if self.config.dataset == 'imdb':
                    labels = labels + 1
                with torch.no_grad():
                    preds, _ = self._model(sents)
                    all_preds.append(preds.argmax(-1))
                    all_labels.append(labels)
                pbar.set_description('{}ing'.format('validat' if stage=='val' else 'test'))
        
        all_preds = torch.cat(all_preds).detach().cpu().numpy()
        all_labels = torch.cat(all_labels).detach().cpu().numpy()
        f1_eval = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        return f1_eval, acc

    def run(self):
        best_test_f1 = 0.0
        best_test_acc = 0.0
        best_dev_f1 = 0.0
        patience = self.patience
        for epoch in range(self.num_epoch):
            # train
            self.train(self.loaders['train_loader_l'], self.loaders['train_loader_u'], epoch)
            # eval
            f1_macro, f1_micro = self.evaluate(self.loaders['dev_loader'], epoch)
            self.logger.info('********Dev results: Acc (MI-F1)={}*********'.format(f1_micro))
            if f1_micro > best_dev_f1:
                patience = self.patience
                self.logger.info('find new best model!')
                best_dev_f1 = f1_micro
                f1_macro, f1_micro = self.evaluate(self.loaders['test_loader'], epoch)
                self.logger.info('********Test results: Acc (MI-F1)={}, F1 (MA)={}*********'.format(f1_micro, f1_macro))
                if f1_micro > best_test_acc:
                    best_test_acc = f1_micro
                    best_test_f1 = f1_macro
            else:
                patience -= 1
                if patience == 0 or epoch == self.config.num_epoch - 1:
                    self.logger.info('No patience, best f1 is {}'.format(best_test_f1))
                    break
        self.logger.info('NUM_LABELD={}: lr={}, lr_bert={}, thr={}, lbd={}, mu={}: best_acc={}, best_MAF1={}\n'.format(self.config.num_labeled, self.config.lr_main, self.config.lr_bert, self.config.thr, self.config.lbd, self.config.mu, best_test_acc, best_test_f1))
        f = open('results_{}.txt'.format(self.config.dataset), 'a')
        f.write('NUM_LABELD={}: lr={}, lr_bert={}, thr={}, lbd={}, mu={}: best_acc={}, best_f1={}\n'.format(self.config.num_labeled, self.config.lr_main, self.config.lr_bert, self.config.thr, self.config.lbd, self.config.mu, best_test_acc, best_test_f1))
        f.close()
        exit()

class SimTrainer(BaseTrainer):
    def __init__(self, config, loaders):
        super(SimTrainer, self).__init__(config, loaders)
        self._model = TextClassifierNoAux(config)
        self._scorer = EmbeddingScorer(config)

        self._model.to(self._device)
        self._scorer.to(self._device)

        main_params = [] 
        bert_params = []
        for n, p in self._model.named_parameters():
            if 'bert' in n:
                bert_params.append(p)
            else:
                main_params.append(p)
        self._main_optimizer = getattr(optim, config.optimizer)(
            [
                {'params': main_params, 'lr': config.lr_main},
                {'params': bert_params, 'lr': config.lr_bert}
            ]
        )
        self._scorer_optimizer = getattr(optim, config.optimizer)(
            params = self._scorer.parameters(), lr=config.lr_aux
        )
    
    def train(self, loader_label, loader_unlabel, epoch):
        with tqdm(loader_label) as pbar:
            for batch_label in pbar:
                sents_l, sents_l_aug1, sents_l_aug2, labels = batch_label
                sents_l, sents_l_aug1, sents_l_aug2 = map(move_to_cuda, [sents_l, sents_l_aug1, sents_l_aug2])
                labels = labels.cuda()

                if self.config.dataset == 'imdb':
                    labels = labels + 1
                # step1: run and compare the outputs from two augmentations of the same labeled input
                raw_preds, hidden_raw = self._model(sents_l, return_hidden=True)
                preds1, hidden1 = self._model(sents_l_aug1, return_hidden=True)  # (n, c)
                preds2, hidden2 = self._model(sents_l_aug2, return_hidden=True)  # (n, c)
                aug_true_choices = compare_and_gen_augchoicemask(preds1, preds2, labels, metric=self.config.aux_metric)
                
                hidden_raw, hidden1, hidden2 = hidden_raw.detach(), hidden1.detach(), hidden2.detach()
                # step2: tune the auxiliary network using labeled inputs
                aug_choice_preds_scores = torch.cat([self._scorer(hidden_raw, hidden1), self._scorer(hidden_raw, hidden2)], dim=-1)   # (n, 2)
                bs = aug_choice_preds_scores.size(0)
                pos = aug_choice_preds_scores[torch.arange(bs), aug_true_choices]
                neg = aug_choice_preds_scores[torch.arange(bs), 1-aug_true_choices]
                self._scorer.zero_grad()
                
                loss_aug_choice = - (pos.sum() / bs - torch.logsumexp(neg, dim=0))
                loss_aug_choice.backward()
                self._scorer_optimizer.step()
                
                # step3: predict which augmentation is relatively weak on unlabeled inputs
                total_loss_aug = 0
                self._model.zero_grad()

                loss_raw = self._criterion(raw_preds, labels)
                loss_raw.backward()
                
                if epoch + 1 > self.config.warmup_epoch:
                    # only do choice inference after warmup ends
                    for _ in range(self.config.mu):
                        try:
                            batch_unlabel = next(generator)
                        except:
                            generator = iter(loader_unlabel)
                            batch_unlabel = next(generator)
                        
                        sents_u, sents_u_aug1, sents_u_aug2, _ = batch_unlabel
                        sents_u, sents_u_aug1, sents_u_aug2 = map(move_to_cuda, [sents_u, sents_u_aug1, sents_u_aug2])
                        
                        _, hidden_u = self._model(sents_u, return_hidden=True)
                        preds1, hidden1 = self._model(sents_u_aug1, return_hidden=True)
                        preds2, hidden2 = self._model(sents_u_aug2, return_hidden=True)    # (n,c), (n,h)
                        # these are logits
                        
                        hidden1, hidden2, hidden_u = hidden1.detach(), hidden2.detach(), hidden_u.detach()
                        all_preds = torch.stack([preds1, preds2], dim=1) # (n, 2, c)
                        
                        # step4: calculate loss for both labeled and unlabeled sententces
                        bs = all_preds.size(0)
                        aug1_score, aug2_score = self._scorer(hidden_u, hidden1), self._scorer(hidden_u, hidden2)
                        
                        if self.config.method == 'fix_match':
                            aug_choices = torch.LongTensor([0]*hidden1.size(0)).cuda()
                        else:
                            aug_choices = torch.argmax(torch.cat([aug1_score, aug2_score], dim=-1), dim=-1)
                        
                        aug_target_logits = all_preds[torch.arange(bs), aug_choices]
                        aug_target = F.softmax(aug_target_logits, dim=-1).detach()
                        
                        # hard label
                        aug_target_tensor, aug_target_indx = aug_target.topk(1, dim=-1)    # (n,), (n,)

                        # generate mask according to threshold and maximum probabilities
                        loss_mask = aug_target_tensor > self.thr
                        # the augmentation logits not used for targets serve as preds
                        aug_pred_logits = all_preds[torch.arange(bs), 1-aug_choices]
                        
                        loss_aug_raw = self._criterion_no_reduce(aug_pred_logits, aug_target_indx.squeeze())
                        loss_aug = (loss_aug_raw * loss_mask).sum() / (self.config.mu * bs)
                        loss_aug *= self.config.lbd
                        total_loss_aug += loss_aug
                        loss_aug.backward()
                        
                # After obtaining all logits we step once
                self._main_optimizer.step()
                
                pbar.set_description('Training epoch {:2d}: loss_labeled: {:.3f}, loss_aug: {:.3f}, loss_aug_choice: {:3f}'.format(epoch+1, loss_raw, total_loss_aug, loss_aug_choice))

    def train_base(self, loader_label, loader_unlabel, epoch):
        self.losses.reset()
        
        with tqdm(loader_label) as pbar:
            for (sents_l, _, _, labels) in pbar:
                sents_l = move_to_cuda(sents_l)
                labels = labels.cuda()

                if self.config.dataset == 'imdb':
                    labels = labels + 1
                
                raw_preds, _ = self._model(sents_l)
                loss = self._criterion(raw_preds, labels)

                self._model.zero_grad()
                loss.backward()
                self._main_optimizer.step()
                self.losses.update(loss, raw_preds.size(0))
                pbar.set_description('Training epoch {:2d}: Loss = {:.4f}'.format(epoch+1, self.losses.val))

    def evaluate(self, loader, epoch=1, stage='val'):
        all_preds, all_labels = [], []
        self.losses.reset()
        
        with tqdm(loader) as pbar:
            for (sents, _, _, labels) in pbar:
                sents = move_to_cuda(sents)
                labels = labels.cuda()

                if self.config.dataset == 'imdb':
                    labels = labels + 1
                with torch.no_grad():
                    preds, _ = self._model(sents)
                    all_preds.append(preds.argmax(-1))
                    all_labels.append(labels)
                pbar.set_description('{}ing'.format('validat' if stage=='val' else 'test'))
        
        all_preds = torch.cat(all_preds).detach().cpu().numpy()
        all_labels = torch.cat(all_labels).detach().cpu().numpy()
        f1_eval = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        return f1_eval, acc    
  
    def run(self):
        best_test_f1 = 0.0
        best_test_acc = 0.0
        best_dev_f1 = 0.0
        patience = self.patience
        for epoch in range(self.num_epoch):
            # train
            self.train(self.loaders['train_loader_l'], self.loaders['train_loader_u'], epoch)
            # eval
            f1_macro, f1_micro = self.evaluate(self.loaders['dev_loader'], epoch)
            self.logger.info('********Dev results: Acc (MI-F1)={}*********'.format(f1_micro))
            if f1_micro > best_dev_f1:
                patience = self.patience
                self.logger.info('find new best model!')
                best_dev_f1 = f1_micro
                f1_macro, f1_micro = self.evaluate(self.loaders['test_loader'], epoch, stage='test')
                self.logger.info('********Test results: Acc (MI-F1)={}, F1 (MA)={}***********'.format(f1_micro, f1_macro))
                if f1_micro > best_test_acc:
                    best_test_acc = f1_micro
                    best_test_f1 = f1_macro
            else:
                patience -= 1
                if patience == 0 or epoch == self.config.num_epoch - 1:
                    self.logger.info('No patience, best f1 is {}'.format(best_test_f1))
                    break
        self.logger.info('NUM_LABELD={}: aug={}, lr={}, lr_bert={}, thr={}, lbd={}, mu={}: best_acc={}, best_MAF1={}\n'.format(self.config.num_labeled, self.config.aug_metric, self.config.lr_main, self.config.lr_bert, self.config.thr, self.config.lbd, self.config.mu, best_test_acc, best_test_f1))
        f = open('results_{}.txt'.format(self.config.dataset), 'a')
        f.write('NUM_LABELD={}: aug={}, lr={}, lr_bert={}, thr={}, lbd={}, mu={}: best_acc={}, best_MAF1={}\n'.format(self.config.num_labeled, self.config.aug_metric, self.config.lr_main, self.config.lr_bert, self.config.thr, self.config.lbd, self.config.mu, best_test_acc, best_test_f1))
        f.close()
        exit()