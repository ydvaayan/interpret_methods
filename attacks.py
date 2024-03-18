import torch
import torchattacks
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchattacks import __all__ as attack_list
import torchvision.models as models
import torchvision
from utils.preprocess import load_preprocess, load_model, load_image_loader
from utils.common import get_output, get_label, denormalize

import os
import numpy as np
from captum.attr import IntegratedGradients

    
def plot_images(input_images, adv_images, input_labels, adv_labels, adv_saliencies, original_saliencies, attack_name):

    iterable = zip(input_images, adv_images, input_labels, adv_labels, adv_saliencies, original_saliencies)
    for input_image, adv_image, input_label, adv_label, adv_saliency, original_saliency in iterable:
        plt.subplot(2, 2, 1)
        plt.imshow(denormalize(adv_image).permute(1, 2, 0))
        plt.title(f'Adv Image: {adv_label}')

        plt.subplot(2, 2, 2)
        noise = (adv_image-input_image).squeeze().abs().mean(0)
        noise /= noise.max()

        plt.imshow(noise.cpu().detach().numpy(), cmap="gray")
        plt.title(f'Noise: {attack_name}')

        plt.subplot(2, 2, 3)
        plt.imshow(denormalize(adv_image).permute(1, 2, 0))
        plt.title(f'Original Explanation: {input_label}')

        plt.subplot(2, 2, 4)
        plt.imshow(adv_saliency, cmap="gray")
        plt.title(f'Adv Explanation')

        plt.show()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Model
    MODEL_NAME = "inceptionv3"

    model = load_model(MODEL_NAME)
    loader = load_image_loader(MODEL_NAME)
    preprocesser = load_preprocess(MODEL_NAME)

    input_images = preprocesser(loader([os.path.join("images", x) for x in os.listdir("images")]).to(device))

    # Getting the output and the labels
    outputs, probabilities = get_output(model, input_images)
    input_labels, label_outputs = get_label(probabilities)

    attack_list = ["PGDL2", "PIFGSM", "PIFGSMPP", "CW"]
    ig = IntegratedGradients(model)
    # Attacking the model
    for attack_name in attack_list:
        print(attack_name)
        attack = getattr(torchattacks, attack_name)
        atk = attack(model)
        # atk = torchattacks.PGDL2(model, eps=1.0, alpha=0.01, steps = 100, random_start=False, eps_for_division=1e-10)
        # atk = torchattacks.OnePixel(model=model, pixels=100, steps=20, popsize=10, inf_batch=128)
        # atk = torchattacks.PIFGSM(model=model, max_epsilon=0.02, num_iter_set=100, momentum=0.2, amplification=2.0, prob=0.1)
        # atk = torchattacks.PIFGSMPP(model=model, max_epsilon=0.02, num_iter_set=100, momentum=0.2, amplification=2.0, prob=0.1, project_factor=0.8)
        # atk = torchattacks.Pixle(model)

        # atk.set_mode_targeted_least_likely(1)
        atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        adv_images = atk(input_images, input_labels)

        adv_outputs, adv_probabilities = get_output(model, adv_images)
        adv_input_labels, adv_label_outputs = get_label(adv_probabilities)
        
        adv_saliency_maps = [ig.attribute(adv_image.unsqueeze(0), target=adv_input_label, n_steps=32).abs().squeeze().mean(0).cpu()
                            for adv_image, adv_input_label in zip(adv_images, adv_input_labels)]

        original_saliency_maps = [ig.attribute(input_image.unsqueeze(0), target=input_label, n_steps=32).abs().squeeze().mean(0).cpu()
                                  for input_image, input_label in zip(input_images, input_labels)]
        
        print(adv_label_outputs)

        plot_images(input_images, adv_images, label_outputs, adv_label_outputs, adv_saliency_maps, original_saliency_maps, attack_name)
        # plt.imshow(denormalize(input_images.squeeze()).permute(1,2,0))
        # plt.imshow(denormalize(adv_images.squeeze()).permute(1,2,0))
        # plt.show()