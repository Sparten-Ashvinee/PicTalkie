# encoding: utf-8
from .example_model import ResNet18
from Model.clip.modeling_clip import CLIPModel

def build_model(cfg):
    #model = ResNet18(cfg.MODEL.NUM_CLASSES)
    model = CLIPModel()
    return model
