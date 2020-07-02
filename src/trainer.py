import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

import os
import wandb
import inspect
import numpy as np
from tqdm import tqdm
from pprint import pprint
from copy import deepcopy
import matplotlib.pylab as plt

from .datasets import *
from .system_def import *
from .models import *


def model_to_CPU_state(net):
    cpu_model = deepcopy(net)
    cpu_model = {k:v.cpu() for k, v in cpu_model.module.state_dict().items()}
    return OrderedDict(cpu_model)

def opimizer_to_CPU_state(opt):
    cpu_opt = deepcopy(opt)
    for state in cpu_opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cpu()
    return cpu_opt.state_dict()

class SegmentationMetrics:
    def __init__(self, n_classes, int_to_lbl=None):
        self.n_classes = n_classes
        if int_to_lbl is None:
            int_to_lbl = {val:'class_'+str(val) for val in range(n_classes)}
        self.int_to_lbl = int_to_lbl
        self.confusion_matrix = np.zeros((n_classes, n_classes))
    
    # add predictions to confusion matrix
    def add_preds(self, y_true, y_pred):
        y_true = y_true.flatten().cpu().numpy()
        y_pred = y_pred.flatten().cpu().numpy()
        np.add.at(self.confusion_matrix, (y_true, y_pred), 1)
    
    # get values that we will use to compute IoU
    def get_info_from_conf(self):
        Cij = self.confusion_matrix
        Cii = np.diagonal(Cij)
        Ci = np.sum(Cij, axis=1)
        Cj = np.sum(Cij, axis=0)        
        return Ci, Cj, Cii 
    
    # Calculate IoU and accuracy and return the results as a dictionary
    def get_value(self):
        Ci, Cj, Cii = self.get_info_from_conf()
        # claculate accuracy metrics
        accuracy = np.sum(Cii)/ np.sum(Ci)
        per_class_accuracy = Cii / Ci
        mean_accuracy = np.nanmean(per_class_accuracy)

        # claculate IoU metrics
        union = Ci + Cj - Cii
        per_class_IoU = Cii/union
        mean_IoU = np.nanmean(per_class_IoU)
    
        # convert to dictionaries
        per_class_accuracy = {self.int_to_lbl[cls]:round(val,3) for cls, val in enumerate(per_class_accuracy)}
        per_class_IoU = {self.int_to_lbl[cls]:round(val,3) for cls, val in enumerate(per_class_IoU)}  
        
        # return metrics as dictionary
        metrics = {"accuracy" : round(accuracy, 3),
                  "per_class_accuracy" : per_class_accuracy,
                   "mean_accuracy" : round(mean_accuracy, 3),
                   "per_class_IoU" : per_class_IoU,
                   "mean_IoU" : round(mean_IoU, 3)}
        return metrics
    
    
class MovingMeans:
    def __init__(self, window=5):
        self.window = window
        self.values = []
        
    def add(self, val):
        self.values.append(val)
        
    def get_value(self):
        return np.convolve(np.array(self.values), np.ones((self.window,))/self.window, mode='valid')[-1]
        
class BaseTrainer:
    def __init__(self):
        self.is_grid_search = False
        self.val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.val_acc = 0.
        self.best_val_acc = 0.
        self.iters = 0
        self.epoch0 = 0
        self.epoch = 0
        self.moving_val_loss = MovingMeans(window=5)
        self.moving_val_loss.add(self.val_loss)
        
        
    
    def attr_from_dict(self, param_dict):
        for key in param_dict:
            setattr(self, key, param_dict[key])
            
    def reset(self):
        self.model.module.load_state_dict(self.org_model_state)
        self.model.to(self.device)
        self.optimizer.load_state_dict(self.org_optimizer_state)
        print(" Model and optimizer are restored to their initial states ")

    def load_session(self, restore_only_model=False):
        self.get_saved_model_path()
        if os.path.isfile(self.model_path) and self.restore_session:        
            print("Loading model from {}".format(self.model_path))
            checkpoint = torch.load(self.model_path)
            self.model.module.load_state_dict(checkpoint['state_dict'])
            self.model = self.model.to(self.device)
            if restore_only_model:
                return
            
            self.iters = checkpoint['iters']
            self.epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.model_path, checkpoint['epoch']))

        elif not os.path.isfile(self.model_path) and self.restore_session:
            print("=> no checkpoint found at '{}'".format(self.model_path))
    
    def get_saved_model_path(self):
        model_saver_dir = os.path.join(os.getcwd(), 'models')
        check_dir(model_saver_dir)
        self.model_path = os.path.join(model_saver_dir, self.model_name)
        
    def save_session(self):
        self.get_saved_model_path()
        print("Saving model as {}".format(self.model_name) )
        state = {'iters': self.iters, 'state_dict': self.best_model,
                 'optimizer': opimizer_to_CPU_state(self.optimizer), 'epoch': self.epoch,
                'parameters' : self.parameters}
        torch.save(state, self.model_path)
    def print_train_init(self):
        print("Start training with learning rate: {}".format(self.optimizer.param_groups[0]['lr']))    
        
    def lr_grid_search(self, min_pow=-5, max_pow=-1, resolution=20, n_epochs=5, 
                       random_lr=False, report_intermediate_steps=False):
        self.is_grid_search = True
        self.save_best_model = False
        self.epochs = n_epochs
        self.scheduler = None 
        pref_m = self.model_name
        self.model_name = 'grid_search'
        self.save_every = float("inf")   
        self.report_intermediate_steps = report_intermediate_steps
        if self.report_intermediate_steps:
            self.val_every = 1            
        else:
            self.log_every = float("inf")
            self.val_every = float("inf")
        
        v_losses = []
        v_accs = []
        if not random_lr:
            e = np.linspace(min_pow, max_pow, resolution)
            lr_points = 10**(e)
        else:
            lr_points = np.random.uniform(min_pow, max_pow, resolution)
            lr_points = 10**(e)
                    
        out_name = pref_m + "_grid_search_out.txt"
        with open(out_name, "w") as text_file:
            print('learning rate \t val_loss \t val_AUC', file=text_file)
        for lr in tqdm(lr_points, desc='Grid search cycles', leave=False):
            
            wandb.init(project=pref_m + '_grid_search', name=str(lr), reinit=True)     
        
            self.optimizer.param_groups[0]['lr'] = lr
            self.train()
            self.evaluate()
            v_losses.append(self.val_loss)
            v_accs.append(self.val_acc)
            self.logging({'val_loss': self.val_loss,
                          'val_acc': self.val_acc})  
            with open(out_name, "a") as text_file:
                print('{} \t {} \t {}'.format(lr,self.val_loss,self.val_acc), file=text_file)
            self.reset()
            self.val_loss = float("inf")
            self.best_val_loss = float("inf")
            self.val_acc = 0.
            self.best_val_acc = 0.
            self.iters = 0
            self.epoch0 = 0
            self.epoch = 0 
            wandb.uninit()
            
        arg_best_acc = np.argmax(v_accs)
        best_acc = v_accs[arg_best_acc]
        best_lr_acc = lr_points[arg_best_acc]

        arg_best_vloss = np.argmin(v_losses)
        best_vloss = v_losses[arg_best_vloss]
        best_lr_vloss = lr_points[arg_best_vloss]

        print("The best val_AUC is {} for lr = {}".format(best_acc, best_lr_acc))
        print("The best val_loss is {} for lr = {}".format(best_vloss, best_lr_vloss))
        
        fig, axs = plt.subplots(1,2, figsize=(15, 6))
        axs = axs.ravel()
        fig.suptitle('Grid search results')
        axs[0].plot(lr_points, v_losses)
        axs[0].scatter(best_lr_vloss, best_vloss, marker='*', c='r', s=100)
        axs[0].plot([best_lr_vloss]*2, [0, best_vloss], linestyle='--', c='r', alpha=0.5)
        axs[0].plot([lr_points[0], best_lr_vloss], [best_vloss]*2, linestyle='--', c='r', alpha=0.5)
        axs[0].set_xlabel('Learning rate')
        axs[0].set_ylabel('Validation loss')
        axs[0].set_xscale('log')
        axs[1].plot(lr_points, v_accs)
        axs[1].scatter(best_lr_acc, best_acc, marker='*', c='r', s=100)
        axs[1].plot([best_lr_acc]*2, [0, best_acc], linestyle='--', c='r', alpha=0.5)
        axs[1].plot([lr_points[0], best_lr_acc], [best_acc]*2, linestyle='--', c='r', alpha=0.5)
        axs[1].set_xlabel('Learning rate')
        axs[1].set_ylabel('Validation AUC')
        axs[1].set_xscale('log')
        plt.savefig(pref_m + '_grid_search_out.png')
              

    def get_lr_schedule(self):        
        if self.lr_scheduler == 'ReduceLROnPlateau':
            self.scheduler =  ReduceLROnPlateau(self.optimizer, 
                                                **self.ReduceLROnPlateau_params)
            if self.val_every > self.total_step:
                self.val_every = self.total_step

        elif self.lr_scheduler == 'MultiStepLR':
            self.scheduler = MultiStepLR(self.optimizer,
                                         milestones=self.MultiStepLR_schedule)
        else:
            self.scheduler = None   
            
    def logging(self, logging_dict):
        wandb.log(logging_dict, step=self.iters)  
                
        
class Trainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion, dataloaders, parameters):
        super().__init__()
        self.parameters = parameters
        training_params = parameters['training_params']
        self.who_called_me = inspect.stack()[1][3]
        self.training_params = training_params
        self.attr_from_dict(self.training_params)
        self.attr_from_dict(dataloaders)
        
        self.model = model
        self.optimizer = optimizer        
        self.criterion = criterion     
        self.n_classes = self.model.module.n_classes
        self.device = self.model.module.device        
        
        self.org_optimizer_state = opimizer_to_CPU_state(self.optimizer)
        
        self.total_step = len(self.trainloader)        
        self.get_lr_schedule()     
        
        self.report_intermediate_steps = True     
        self.org_model_state = model_to_CPU_state(self.model)
        self.best_model = deepcopy(self.org_model_state)
        
    def train(self):
        self.test_mode = False
        if not self.is_grid_search:
            self.load_session(self.restore_only_model)
        self.print_train_init()
        
        for self.epoch in tqdm(range(self.epoch0 + 1, self.epoch0 + self.epochs + 1), desc='Training', leave=False):
            for it, batch in tqdm(enumerate(self.trainloader), desc='Epoch', leave=False, total=len(self.trainloader)):
                self.iters += 1
                self.global_step(batch=batch, it=it)            
        print(" ==> Training done")
            
        if not self.is_grid_search:
            self.evaluate()
            self.save_session()
            return self.best_model
        
    def global_step(self, **kwargs):
        self.model.train() 
        self.optimizer.zero_grad()
        
        batch = kwargs['batch']
        images = batch['img']
        masks = batch['mask']
        masks = masks.to(self.device)
        outputs = self.model(images)['out']
        
        loss = self.criterion(outputs, masks)            
        loss.backward() 
        self.optimizer.step()  
        loss = loss.item()
    
        if self.iters % self.log_every == 0 or (self.iters == 1 and not self.is_grid_search):
            self.logging({'train_loss': loss})    
            self.epoch_step()   
    
    def epoch_step(self, **kwargs):        
        if self.epoch % self.val_every == 0:
            self.evaluate()
            self.logging({'val_loss': self.val_loss,
                         'val_acc': self.val_acc})
            
        if self.epoch % self.save_every == 0 and not self.is_grid_search:
            tmp_sess_name = "{}-ep_{}-it{}".format(self.model_name, 
                                                   self.epoch, self.iters)
            self.save_session()        
        
        if not self.is_grid_search and self.lr_scheduler is not None:       
            if self.lr_scheduler == 'MultiStepLR':
                self.scheduler.step()
            if self.lr_scheduler == 'ReduceLROnPlateau':
                if self.scheduler.mode == 'min':
                    self.scheduler.step(self.val_loss)
                else:
                    self.scheduler.step(self.val_acc)
                
    def evaluate(self, dataloader=None, **kwargs):
        self.model.eval()
        if dataloader == None:
            dataloader=self.valloader
            
        main_target = dataloader.dataset.main_target
        if dataloader.dataset.is_binary:
            main_int_target = 1
        else:
            main_int_target = dataloader.dataset.labels_to_int[dataloader.dataset.main_target]
            
        val_loss = 0
        val_acc = 0
        metric = SegmentationMetrics(len(dataloader.dataset.int_to_labels), 
                         int_to_lbl=dataloader.dataset.int_to_labels)
        
        valCE_weights = torch.zeros(self.n_classes)
        valCE_weights[main_int_target] = 1.
        valCE_weights = valCE_weights.to(self.device)
        val_CE = nn.CrossEntropyLoss(weight=valCE_weights)
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['img']
                masks = batch['mask']
                masks = masks.to(self.device)

                outs = self.model(images)['out']
                loss = val_CE(outs, masks).item()
                val_loss += loss                
                outs = outs.max(dim = 1)[1].detach()
                metric.add_preds(y_true=masks,y_pred=outs)

        val_loss /= len(dataloader)
        val_acc = metric.get_value()['per_class_IoU'][main_target]
        self.val_loss = val_loss
        self.val_acc = val_acc
        self.moving_val_loss.add(val_loss)
        _val_loss = self.moving_val_loss.get_value()       
        if (not self.is_grid_search) or self.report_intermediate_steps:
            self.logging(metric.get_value())
        if not self.is_grid_search:
            if self.save_best_model and val_acc > self.best_val_acc: 
                if val_loss <= self.best_val_loss:
                    self.best_model = model_to_CPU_state(self.model)
            elif not self.save_best_model:
                self.best_model = model_to_CPU_state(self.model)
            if val_acc >= self.best_val_acc:
                self.best_val_acc = self.val_acc 
            if _val_loss <= self.best_val_loss:
                self.best_val_loss = _val_loss
                
        self.model.train()
    
    def test(self, dataloader=None, **kwargs):
        self.test_mode = True
        self.restore_session = True
        self.restore_only_model = True
        self.load_session()
        self.model.eval()
        if dataloader == None:
            dataloader=self.testloader  
        main_target = dataloader.dataset.main_target            
        metric = SegmentationMetrics(len(dataloader.dataset.int_to_labels), 
                                     int_to_lbl=dataloader.dataset.int_to_labels)
        test_loss = 0
        test_acc = 0        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images = batch['img']
                masks = batch['mask']
                masks = masks.to(self.device)

                outs = self.model(images)['out']
                loss = self.criterion(outs, masks).item()
                test_loss += loss                
                outs = outs.max(dim = 1)[1]
                metric.add_preds(y_true=masks,y_pred=outs)
        
        test_loss /= len(dataloader)
        self.test_loss = test_loss
        test_acc = metric.get_value()['per_class_IoU'][main_target]
        self.test_acc = test_acc
        self.model.train()
        
        print("--"*25)
        pprint(metric.get_value())
        print("--"*25)
        print("-*"*25)
        print("Test IoU for the target {} :".format(main_target))
        print(metric.get_value()['per_class_IoU'][main_target])
        print("-*"*25)  
        