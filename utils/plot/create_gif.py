import glob
import imageio


images = []
for filepath in glob.glob("saved_models/LeNet_MNIST_Multi_4/images/inner_products_weights/checkpoint_*.pt.png"):
    images.append(imageio.imread(filepath))
imageio.mimsave("saved_models/LeNet_MNIST_Multi_4/images/inner_products_weights/inner_products_weights.gif", images)
