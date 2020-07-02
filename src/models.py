import torch
from torch import nn
from collections import OrderedDict
from torch.nn import functional as F
from torchvision.models.segmentation.segmentation import _segm_resnet

from .system_def import get_device

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = get_device()
    
    def attr_from_dict(self, param_dict):
        for key in param_dict:
            setattr(self, key, param_dict[key])
            
    def get_out_channels(self, m):
        def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())
        c=children(m)
        if len(c)==0: return None
        for l in reversed(c):
            if hasattr(l, 'num_features'): return l.num_features
            res = self.get_out_channels(l)
            if res is not None: return res
            
    def get_submodel(self, m, min_layer=None, max_layer=None):
        return list(m.children())[min_layer:max_layer]
    
    def freeze_bn(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.BatchNorm2d):
                layer.eval()
                
    def unfreeze_bn(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.BatchNorm2d):
                layer.train()
                
    def freeze_submodel(self, submodel=None):
        submodel = self if submodel is None else submodel
        for param in submodel.parameters():
            param.requires_grad = False
            
    def unfreeze_submodel(self, submodel=None):
        submodel = self if submodel is None else submodel
        for param in submodel.parameters():
            param.requires_grad = True

    def initialize_norm_layers(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.BatchNorm2d) or isinstance(layer,  nn.GroupNorm):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()  

    def freeze_norm_layers(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.BatchNorm2d) or isinstance(layer,  nn.GroupNorm):
                layer.eval()  
                
    def init_weights(self, submodel=None):
        submodel = self if submodel is None else submodel
        for layer in submodel.modules():
            if isinstance(layer,  nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data)
                
    def BN_to_GN(self, submodel=None, num_groups=32, keep_stats=True):
        def get_atr(m,n):
            try:
                a = getattr(m, n)
                return a
            except:
                return m[int(n)]
        def recur_depth(submodel,lname, n=0, keep_stats=True, num_groups=32):
            if n < len(lname)-1:
                return recur_depth(get_atr(submodel,lname[n]),
                                   lname, n=n+1, keep_stats=keep_stats, num_groups=num_groups)
            else:
                old_l = getattr(submodel, lname[n])
                nc = old_l.num_features
                new_l = nn.GroupNorm(num_groups=num_groups, num_channels=nc)
                if keep_stats:
                    new_l.weight = old_l.weight
                    new_l.bias = old_l.bias
                setattr(submodel, lname[n], new_l)
                
        submodel = self if submodel is None else submodel
        for name, module in submodel.named_modules():
            if isinstance(module,  nn.BatchNorm2d):
                recur_depth(submodel,name.split('.'), keep_stats=keep_stats, num_groups=num_groups)

    def print_trainable_params(self, submodel=None):
        submodel = self if submodel is None else submodel
        for name, param in submodel.named_parameters():
            if param.requires_grad:
                print(name)  
                
class SegNet(BaseModel):
    def __init__(self, model_params):
        super().__init__()
        
        self.model_params = model_params
        self.attr_from_dict(model_params)         
        
        fcn_backbone = _segm_resnet(name=self.segmentation_type, 
                                    backbone_name=self.backbone_type, 
                                    num_classes=self.n_classes, 
                                    aux=False, 
                                    pretrained_backbone=self.pretrained)    
        if self.img_channels == 1:
            pre_weight = nn.Parameter(fcn_backbone.backbone.conv1.weight[:,:1].clone().detach())
            fcn_backbone.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), 
                                                    stride=(2, 2), padding=(3, 3), bias=False)
            fcn_backbone.backbone.conv1.weight = pre_weight
        if self.freeze_backbone:
            self.freeze_submodel(fcn_backbone.backbone)
        self.backbone = fcn_backbone
        
        if self.goup_norm['replace_with_goup_norm']:
            self.BN_to_GN(num_groups=self.goup_norm['num_groups'],
                          keep_stats=self.goup_norm['keep_stats'])
            
    def forward(self, x):
        x = self.backbone(x)
        
        return x