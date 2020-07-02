import os
import json
import torch
import pickle

def save_json(data, fname):
    with open(fname, 'w') as wfile:  
        json.dump(data, wfile)
        
def load_json(fname):
    with open(fname, "r") as rfile:
        data = json.load(rfile)
    return data

def save_pickle(data, fname):
    with open(fname, 'wb') as wfile:
        pickle.dump(data, wfile, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_pickle(fname):
    with open(fname, 'rb') as rfile:
        data = pickle.load(rfile)
    return data

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dir_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotAFileError(string)

def get_saved_model_path(checkpoint_name):
    path = os.path.join(os.getcwd(), 'models')
    if not os.path.exists(path):
        raise IOError("Checkpoint path {} does not exist".format(path))
    else:
        return os.path.join(path, checkpoint_name) 
    
def load_params(args):
    if args.json_path is not None:
        return load_json(args.json_path)
    elif args.checkpoint is not None:
        checkpoint_path = get_saved_model_path(args.checkpoint)
        checkpoint = torch.load(checkpoint_path)
        return checkpoint['parameters']
    else:
        raise IOError("Please define the training paramenters")
        
       