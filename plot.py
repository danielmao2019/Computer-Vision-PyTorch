import numpy as np
import matplotlib.pyplot as plt
import glob
import os


filepath_list = glob.glob("saved_tensors/inner_products_weights/inner_products*.txt")
for filepath in filepath_list:
    inner_products = np.loadtxt(filepath)
    plt.figure()
    plt.hist(inner_products, bins=100)
    plt.savefig(os.path.join("saved_images", "inner_products_weights",
        f"checkpoint_{filepath.split('/')[-1].split('.')[0].split('_')[-1]}.pt.png"
    ))
