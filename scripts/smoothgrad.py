from utils.preprocess import img2tensor
from utils.common import saliency_tensor
from utils.plot import visualiser
from captum.attr import NoiseTunnel
import importlib


class ImpSmoothGrad():
    def __init__(self, model, inp_path, method_name):
        self.model = model
        self.inp_path = inp_path
        self.method_name=method_name
        if self.method_name=='GradCAM':
            module=importlib.import_module("pytorch_grad_cam")
        else:
            module=importlib.import_module("captum.attr")
        self.method=getattr(module,self.method_name)
    
    def attribute(self):
        inp, baseline = img2tensor(self.inp_path)
        m=self.method(self.model)
        sg = NoiseTunnel(m)
        attributions_sg = sg.attribute(inp, target=0)
        return attributions_sg
    
    def saliency(self):
        return saliency_tensor(self.attribute())
    
    def visualise(self):
        visualiser(self.saliency(), "SmoothGrad + "+self.method_name)