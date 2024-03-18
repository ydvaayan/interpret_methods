import matplotlib.pyplot as plt
import numpy as np

def visualiser(sal_tensor,name):
    plt.figure(figsize=(5, 5))

    plt.imshow(np.abs(sal_tensor), cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.title(name)
    plt.axis('off')
    plt.show()

    # plt.savefig("output.png")