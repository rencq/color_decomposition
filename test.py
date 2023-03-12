import numpy as np
a = np.load("/home/ubuntu/Rencq/nerf_data/nerf_360_real/vasedeck/poses_bounds.npy")
np.savetxt("pose.txt",a)