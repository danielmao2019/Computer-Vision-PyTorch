import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm


filepath_list = sorted(glob.glob("saved_models/LeNet_MNIST_Multi_3/tensors/inner_products_weights/inner_products_checkpoint*.pt.txt"))
for filepath in tqdm(filepath_list):
    inner_products = np.loadtxt(filepath)
    fig, ax_list = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    for ax in ax_list:
        ax.set_ylim([0, 60000])
        ax.set_xlabel("Inner Product")
        ax.set_ylabel("Frequency")
    ax_list[0].hist(inner_products, bins=100, range=[-100, +100])
    ax_list[1].hist(inner_products, bins=100, range=[-0.1, +0.1])
    epoch = filepath.split('/')[-1].split('.')[0].split('_')[-1]
    fig.suptitle(f"Histogram of Inner Products on Training Set" + '\n' + '*' * (int(epoch) // 5))
    image_path = os.path.join("saved_models/LeNet_MNIST_Multi_3/images/inner_products_weights",
        f"checkpoint_{epoch}.pt.png",
    )
    plt.savefig(image_path)
