import os.path

import numpy as np

from data.llff import LLFFDataset

filedir = "/home/ubuntu/Rencq/nerf_data/nerf_360_real/vasedeck/"
outdir = "/home/ubuntu/Rencq/out_image/"

llffdataset = LLFFDataset(filedir,downsample=4,spheric_poses=True)
for i in range(llffdataset.poses.shape[0]):
    outpath_pose = os.path.join(outdir,f"IMG_{8361+i}.txt")
    tmp = np.array([[0.,0.,0.,1.]])
    poses = np.concatenate([llffdataset.poses[..., 1:2], llffdataset.poses[..., :1], -llffdataset.poses[..., 2:3],llffdataset.posesses[...,3]], -1)
    pose_tmp = np.concatenate((llffdataset.poses[i,...],tmp),axis=0)
    np.savetxt(outpath_pose,pose_tmp,header='extrinsic',comments='')
for i in range(llffdataset.poses.shape[0]):
    outpath_pose = os.path.join(outdir, f"IMG_{8361 + i}.txt")
    K = [[llffdataset.focal[0],0.0,llffdataset.img_wh[0]/2],[0.,llffdataset.focal[1],llffdataset.img_wh[1]/2],[0.,0.,1.]]
    with open(outpath_pose,"a") as file:
        file.write("\n")
        file.write('intrinsic\n')
        for i in range(len(K)):
            for j in range(len(K[i])):
                file.write(str(K[i][j]))
                if j != len(K[i])-1:
                    file.write(" ")
            file.write("\n")
        file.write("\n")
        d = (llffdataset.near_far[1]-llffdataset.near_far[0])/512
        file.write(str(llffdataset.near_far[0]))
        file.write(" ")
        file.write(str(d))

"""save nearfar"""
outpath_nearfar = os.path.join(outdir,f"nearfar.txt")
np.savetxt(outpath_nearfar,llffdataset.near_far)
"""save focal"""
outpath_focal = os.path.join(outdir,f"focal.txt")
np.savetxt(outpath_focal,llffdataset.focal)
