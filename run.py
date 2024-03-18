from scripts import ImpDeepLift,ImpGuidedBackprop,ImpSmoothGrad,ImpGradCam, ImpIntegratedGradients
import torchvision.models as models
from PIL import Image
import numpy as np
import torch


# model=models.inception_v3(pretrained=True)
model = models.resnet50(pretrained=True).cuda()
input_path = "doberman.png"
input_matrix = np.array(Image.open(input_path), dtype=np.float32) / 255

ig = ImpSmoothGrad(model, input_matrix, 'DeepLift')
ig.visualise()
# gb = ImpGuidedBackprop(model, input_path)
# gb.visualise()
# sg= ImpSmoothGrad(model, input_path,'DeepLift')
# sg.visualise()

# target_layers = [model.layer4[-1]]
# gc=ImpGradCam(model,input_path,target_layers)
# gc.visualise()
