from utils.preprocess import img2tensor
from utils.common import saliency_tensor
from utils.plot import visualiser
from captum.attr import GuidedBackprop


class ImpGuidedBackprop():
    def __init__(self, model, inp_path):
        self.model = model
        self.inp_path = inp_path
    
    def attribute(self):
        inp, baseline = img2tensor(self.inp_path)
        gb = GuidedBackprop(self.model)
        attributions_gb = gb.attribute(inp, target=0)
        return attributions_gb
    
    def saliency(self):
        return saliency_tensor(self.attribute())
    
    def visualise(self):
        visualiser(self.saliency(), "GuidedBackprop")
