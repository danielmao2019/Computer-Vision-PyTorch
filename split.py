import numpy as np
import os


parent_dir = os.path.join("saved_tensors", "inner_products_weights")
inner_products = np.loadtxt(os.path.join(parent_dir, "inner_products_checkpoint_200.pt.txt"))
for threshold in [1.0e-09, 1.0e-07, 1.0e-05, 1.0e-03, 1.0e-01]:
    np.savetxt(fname=os.path.join(parent_dir, f"easy_cp_200_th_1.0e-{int(-np.log10(threshold)):02d}.txt"),
        X=np.where(inner_products >= -threshold)[0].astype(np.int64), fmt='%d')
    np.savetxt(fname=os.path.join(parent_dir, f"hard_cp_200_th_1.0e-{int(-np.log10(threshold)):02d}.txt"),
        X=np.where(inner_products < -threshold)[0].astype(np.int64), fmt='%d')

threshold = 0
np.savetxt(fname=os.path.join(parent_dir, f"easy_cp_200_th_0.txt"), X=np.where(inner_products >= -threshold)[0].astype(np.int64), fmt='%d')
np.savetxt(fname=os.path.join(parent_dir, f"hard_cp_200_th_0.txt"), X=np.where(inner_products < -threshold)[0].astype(np.int64), fmt='%d')
