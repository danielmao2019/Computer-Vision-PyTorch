import numpy as np

inner_products = np.loadtxt("saved_tensors/inner_products_checkpoint_200.pt.txt")
for threshold in [1.0e-09, 1.0e-07]:
    np.savetxt(fname=f"saved_tensors/easy_cp_200_th_1.0e-{int(-np.log10(threshold)):02d}.txt", X=np.where(inner_products >= -threshold)[0].astype(np.int64), fmt='%d')
    np.savetxt(fname=f"saved_tensors/hard_cp_200_th_1.0e-{int(-np.log10(threshold)):02d}.txt", X=np.where(inner_products < -threshold)[0].astype(np.int64), fmt='%d')

threshold = 0
np.savetxt(fname=f"saved_tensors/easy_cp_200_th_0.txt", X=np.where(inner_products >= -threshold)[0].astype(np.int64), fmt='%d')
np.savetxt(fname=f"saved_tensors/hard_cp_200_th_0.txt", X=np.where(inner_products < -threshold)[0].astype(np.int64), fmt='%d')
