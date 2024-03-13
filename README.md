## Instructions for Setup and Use

```bash
conda create --name interpret python==3.8
conda activate interpret
pip install torch==2.2.0 torchvision==0.17.0
```

Setting up [Captum](!https://github.com/pytorch/captum/tree/master) and [GradCam](!https://github.com/jacobgil/pytorch-grad-cam)
```bash
conda install captum -c pytorch
pip install nomkl
pip install saliency
pip install grad-cam
```

To Run DeepLift or GuidedBackprop make changes in `run.py`
 Example:
```python
from scripts import ImpDeepLift
import torchvision.models as models

if __name__ == "__main__":
 
    model=models.inception_v3(pretrained=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    input_path = "doberman.png"

    dl = ImpDeepLift(model, input_path)
    dl.visualise()

```
Use `saliency()` instead of `visualise` if you want to return the saliency tensor instead of the plot

To implement SmoothGrad with say DeepLift (for example)

``python
    sg= ImpSmoothGrad(model, input_path,'DeepLift')
    sg.visualise()
```




