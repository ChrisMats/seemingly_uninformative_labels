#!/usr/bin/env python
# coding: utf-8

import os
import json
import wandb
import argparse

import torch
from torch import nn
from torch import optim

from src.models import *
from src.trainer import *
from src.datasets import *


def parse_arguments():
    parser = argparse.ArgumentParser(description='The main takes as \
                             argument the parameters dictionary from a json file')
    parser.add_argument('--json_path', type=dir_path, required=False, help= 'Give a valid json file')
    parser.add_argument('--checkpoint', type=str, required=False, help= 'Give a valid checkpoint name')
    parser.add_argument('--test', action='store_true', default=False, help= 'Flag for testing')
    parser.add_argument('--find_lr', action='store_true', default=False, help= 'Flag for lr finder')
    parser.add_argument('--annotator', type=int, required=False, help= 'Give the annotator ID')
    parser.add_argument('--data_location', type=str, required=False, help= 'Update the datapath')
    return parser.parse_args()

def main(args):
    # get parameters
    parameters = load_params(args)
    dataset_params = parameters['dataset_params']
    if args.data_location is not None:
        dataset_params['data_location'] = args.data_location    
    dataloader_params = parameters['dataloader_params']
    if dataset_params['dataset_location'] == 'CsawS':
        DataSet = CsawSet
    elif dataset_params['dataset_location'] == 'Cityscapes':
        DataSet = CityScapes
    elif dataset_params['dataset_location'] == 'VOC':
        DataSet = PascalVOC                 
    else:
        raise ImportError("Dataset not found")
        
    training_params = parameters['training_params']
    system_params = parameters['system_params']
    log_params = parameters['log_params']
    lr_finder_params = parameters['lr_finder']
    model_params = parameters['model_params']
    model_params['img_channels'] = DataSet.img_channels
    model_params['n_classes'] = len(DataSet.int_to_labels)
    if dataset_params['main_target'] == 'human':
        model_params['n_classes'] -= 1

    # define system
    define_system_params(system_params)
    
    # setting up parameters for ablation studies
    seed_n = None
    if not dataset_params['use_full_training_set']:
        seed_n = dataset_params['subset_n']
        
    if args.test and DataSet is CsawSet:
        if args.annotator is not None:
            dataset_params['annotator_id'] = args.annotator
        else:
            dataset_params['annotator_id'] = int(input(
                "Annotator ID not specified!\n Please give the Annotator\'s ID from the keyboard:\n "))
    else:
        dataset_params['annotator_id'] = 1

    included_labels = "all"
    if not dataset_params['is_binary']:
        old_int_to_labels = deepcopy(DataSet.int_to_labels)
        old_labels_to_int = deepcopy(DataSet.labels_to_int)
        if dataset_params['n_complementary_labels'] != 'all' or \
        dataset_params['leave_one_out']['apply']:
            if dataset_params['n_complementary_labels'] != 'all':
                training_params['model_name'] = '_'.join(['n_lbs', 
                                                          str(dataset_params['n_complementary_labels']),
                                                          training_params['model_name']])
                included_labels = [0, old_labels_to_int[dataset_params['main_target']]]
                box_of_labels = [lbl for lbl in old_int_to_labels.keys() 
                                 if lbl not in included_labels]
                included_labels += np.random.choice(box_of_labels, 
                                                        size=dataset_params['n_complementary_labels'], 
                                                        replace=False).tolist()
            if dataset_params['leave_one_out']['apply']:
                if dataset_params['leave_one_out']['label_id'] in [0, old_labels_to_int[dataset_params['main_target']]]:
                    raise IndexError("This label cannot be removed since it is the main target")
                label_ingore = DataSet.int_to_labels[dataset_params['leave_one_out']['label_id']]
                if ' ' in label_ingore or '/' in label_ingore or '\\' in label_ingore:
                    label_ingore = ''.join(filter(str.isalpha, label_ingore))
                training_params['model_name'] = '_'.join(['ignore', label_ingore, training_params['model_name']])                 
                included_labels = [lbl for lbl in old_int_to_labels.keys() 
                                 if lbl != dataset_params['leave_one_out']['label_id']]                
            model_params['n_classes'] = len(included_labels)
            
    # define dataset params and dataloaders            
    trainset = DataSet(dataset_params=dataset_params, mode='train', 
                       seed_n=seed_n, included_labels=included_labels)
    valset = DataSet(dataset_params=dataset_params, mode='eval', 
                       seed_n=seed_n, included_labels=included_labels)
    testset = DataSet(dataset_params=dataset_params, mode='test', 
                       seed_n=seed_n, included_labels=included_labels)

    if dataset_params['is_binary']:
        model_params['n_classes'] = 2
        trainset.binarize_labels()
        valset.binarize_labels()
        testset.binarize_labels()
    
    trainLoader = DataLoader(trainset, **dataloader_params['trainloader'])
    valLoader = DataLoader(valset, **dataloader_params['valloader'])
    testLoader = DataLoader(testset, **dataloader_params['testloader'])

    dataloaders = {'trainloader': trainLoader, 
                   'valloader' : valLoader,
                   'testloader' : testLoader,}
    
    # initialize logger
    if log_params['run_name'] == "DEFINED_BY_MODEL_NAME":
        log_params['run_name'] = training_params['model_name']    
    if not (args.test or args.find_lr):
        wandb.init(project=log_params['project_name'], 
                   name=log_params['run_name'], 
                   config=parameters)
    
    # Define model, criterion and optimizer
    model = SegNet(model_params)
    model = nn.DataParallel(model)
    model = model.to(model.module.device)
    
    # Define criterion and optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=training_params['learning_rate'],
                           weight_decay=training_params['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    
    # define trainer 
    trainer = Trainer(model, optimizer, criterion, dataloaders, parameters)    
    if args.test:
        trainer.test()
    elif args.find_lr:
        trainer.lr_grid_search(**lr_finder_params['grid_search_params'])
    else:
        model = trainer.train()
        trainer.test()
    
    
if __name__ == '__main__':
    args = parse_arguments()
    main(args)
    