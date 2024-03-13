from utils.preprocess import img2tensor
from utils.common import saliency_tensor
from utils.plot import visualiser
from captum.attr import DeepLift


class ImpDeepLift():
    def __init__(self, model, inp_path):
        self.model = model
        self.inp_path = inp_path
    
    def attribute(self):
        inp, baseline = img2tensor(self.inp_path)
        dl = DeepLift(self.model)
        attributions_dl, delta = dl.attribute(inp, baseline, target=0, return_convergence_delta=True)
        return attributions_dl
    
    def saliency(self):
        return saliency_tensor(self.attribute())
    
    def visualise(self):
        visualiser(self.saliency(), "DeepLift")

        







