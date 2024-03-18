import numpy as np
import saliency.core as saliency
import torch, torchvision
from matplotlib import pyplot as plt

def normalize_scores(scores):

    scores_np = scores.cpu().detach().numpy()  # Convert to NumPy array
    scores_normalized = (scores_np - np.min(scores_np)) / (np.max(scores_np) - np.min(scores_np))
    return scores_normalized

def saliency_tensor(attributions):
  # Load your attribution scores and input image
  attribution_scores = attributions.squeeze().permute(1,2,0).contiguous() # Your attribution scores, shape: (height, width, channels)
  # Normalize scores
  normalized_scores = normalize_scores(attribution_scores)
  # Generate saliency map
  saliency_map =  1-saliency.VisualizeImageGrayscale(normalized_scores)
  return saliency_map

def imshow(imgs, title):
    img = torchvision.utils.make_grid(imgs.cpu().data)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img.cpu() * std + mean  # Apply the reverse formula
    return img

def get_output(model, input_batch):
    model.eval()
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output, dim=1)

    return output[0], probabilities

def get_label(probabilities):
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    mask = torch.argmax(probabilities, dim=1).detach()
    return mask, [categories[t] for t in mask] 