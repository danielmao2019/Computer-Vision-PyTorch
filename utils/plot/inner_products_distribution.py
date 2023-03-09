import numpy as np
import matplotlib.pyplot as plt
import glob
import os


for filepath in glob.glob("saved_models/LeNet_MNIST_Multi_4/tensors/inner_products_weights/inner_products*.txt"):
    inner_products = np.loadtxt(filepath)
    plt.figure()
    plt.ylim([0, 60000])
    plt.xlabel("Inner Product")
    plt.ylabel("Frequency")
    plt.hist(inner_products, bins=100)
    epoch = filepath.split('/')[-1].split('.')[0].split('_')[-1]
    plt.title(f"Histogram of Inner Products on Training Set" + '\n' + '*'*int(epoch)/5)
    image_path = os.path.join("saved_models/LeNet_MNIST_Multi_4/images/inner_products_weights",
        f"checkpoint_{epoch}.pt.png",
    )
    plt.savefig(image_path)
