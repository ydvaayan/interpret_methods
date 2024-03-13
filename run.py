from scripts import ImpDeepLift,ImpGuidedBackprop,ImpSmoothGrad,ImpGradCam, ImpIntegratedGradients
import torchvision.models as models
if __name__ == "__main__":
 
    # model=models.inception_v3(pretrained=True)
    model = models.resnet50(pretrained=True)
    input_path = "doberman.png"

    ig = ImpIntegratedGradients(model, input_path)
    print(ig.saliency())
    ig.visualise()
    # gb = ImpGuidedBackprop(model, input_path)
    # gb.visualise()
    # sg= ImpSmoothGrad(model, input_path,'DeepLift')
    # sg.visualise()

    # target_layers = [model.layer4[-1]]
    # gc=ImpGradCam(model,input_path,target_layers)
    # gc.visualise()
