from utils.preprocess import img2tensor
from utils.plot import visualiser
from pytorch_grad_cam import GradCAM
import torch

class ImpGradCam():
    def __init__(self, model, inp_path,target_layers,targets=None):
        self.model = model
        self.inp_path = inp_path
        self.target_layers = target_layers
        self.targets=targets
    
    def saliency(self):
        inp, baseline = img2tensor(self.inp_path)
        cam = GradCAM(model=self.model, target_layers=self.target_layers)
        grayscale_cam = cam(input_tensor=inp, targets=self.targets)
        return grayscale_cam[0]
    
    def visualise(self):
        visualiser(self.saliency(),'GradCam')

