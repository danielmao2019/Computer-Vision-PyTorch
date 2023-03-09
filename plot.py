import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import imageio


# for filepath in glob.glob("saved_models/LeNet_MNIST_Multi_4/tensors/inner_products_weights/inner_products*.txt"):
#     inner_products = np.loadtxt(filepath)
#     plt.figure()
#     plt.hist(inner_products, bins=100, range=[-200, +100])
#     epoch = filepath.split('/')[-1].split('.')[0].split('_')[-1]
#     plt.title(f"{epoch=}")
#     image_path = os.path.join("saved_models/LeNet_MNIST_Multi_4/images/inner_products_weights",
#         f"checkpoint_{epoch}.pt.png",
#     )
#     plt.savefig(image_path)


images = []
for filepath in glob.glob("saved_models/LeNet_MNIST_Multi_4/images/inner_products_weights/checkpoint_*.pt.png"):
    images.append(imageio.imread(filepath))
imageio.mimsave("saved_models/LeNet_MNIST_Multi_4/images/inner_products_weights/inner_products_weights.gif", images)
