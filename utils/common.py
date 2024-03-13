import numpy as np
import saliency.core as saliency


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