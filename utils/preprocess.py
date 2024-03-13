from PIL import Image
import numpy as np
from torchvision.transforms import transforms
import torch

def img2tensor(img_path):
    
    image = Image.open(img_path)
    image_np = np.array(image)

    mean = np.mean(image_np, axis=(0, 1)) / 255.0
    std = np.std(image_np, axis=(0, 1)) / 255.0

    
    transform = transforms.Compose([
        transforms.Resize((299, 299)), 
        transforms.ToTensor(),          
        transforms.Normalize(mean=mean, std=std)  
    ])

    tensor_image = transform(image).unsqueeze(0)  
    inp = tensor_image

    baseline = torch.zeros( tensor_image.shape)

    return inp,baseline

