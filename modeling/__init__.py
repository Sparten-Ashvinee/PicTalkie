# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .example_model import ResNet18
from Model.clip.modeling_clip import CLIPModel

def build_model(cfg):
    #model = ResNet18(cfg.MODEL.NUM_CLASSES)
    model = CLIPModel()
    return model
