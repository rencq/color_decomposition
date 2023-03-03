import os
import sys
import open3d as o3d
import imageio
import numpy as np
import torch
from tqdm.auto import  trange
from .color_decomposition import color_decomposition,plot_color_decomposition_idx,plt_color_decomposition



@torch.no_grad()
def write_point_cloud(test_dataset, tensorf, args, renderer, savePath=None, N_vis=5, N_samples=-1, white_bg=False,
               ndc_ray=False, palette=None, new_palette=None,device='cuda',filename=None):

    '''
    point_cloud
    '''

    point_clouds = []

    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
    test_rays = test_dataset.all_rays[0::img_eval_interval]
    pbar = trange(len(test_rays), file=sys.stdout, position=0, leave=True)
    for idx in pbar:
        samples = test_rays[idx]

        # W, H = test_dataset.img_wh

        rays = samples.view(-1, samples.shape[-1])

        res = renderer(rays, tensorf, chunk=4096, N_samples=N_samples, ndc_ray=ndc_ray, white_bg=white_bg,
                       device=device,
                       ret_opaque_map=True, palette=palette, new_palette=new_palette)


        depth_map = res['depth_map']
        point_cloud = rays[...,:3] + rays[...,3:6] * torch.reshape(depth_map,(-1,1))

        point_clouds.append(point_cloud)
        if idx == 2:
            break

    point_clouds = torch.tensor(point_clouds)

    out_filepath = './logs/point_cloud'
    if not os.path.exists(out_filepath):
        os.makedirs(out_filepath)

    if savePath is not  None:
        out_filepath = os.path.join(out_filepath,savePath)

    if not os.path.exists(out_filepath):
        os.makedirs(out_filepath)
    points_data = torch.reshape(point_clouds,(-1,3))
    if filename is not  None:
        np.savetxt(f'{out_filepath}/{filename}',points_data.cpu().numpy())
    else:
        np.savetxt(f'{out_filepath}/point_clouds.txt',points_data.cpu().numpy())

    return point_clouds


@torch.no_grad()
def read_point_cloud(filepath,filename,format='xyz'):

    pcd = o3d.io.read_point_cloud(f'{filepath}/{filename}',format=format)

    # o3d.visualization.draw([pcd])
    return pcd

@torch.no_grad()
def write_point_cloud_with_color_decomposition(test_dataset, tensorf, args, renderer,eps=0.2, savePath=None, N_vis=-1, N_samples=-1, white_bg=False,
               ndc_ray=False, palette=None, new_palette=None,device='cuda',filename=None):

    '''
    point_cloud
    '''

    point_clouds_idx = []
    print(f'all_rays =============> N =  {test_dataset.all_rays.shape[0]}')
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
    test_rays = test_dataset.all_rays[0::img_eval_interval]
    print(f'sample_rays =============> N =  {test_rays.shape[0]}')
    pbar = trange(len(test_rays), file=sys.stdout, position=0, leave=True)

    palette_number = palette.shape[0]
    for i in range(palette_number):
        point_clouds_idx.append([])

    for idx in pbar:
        samples = test_rays[idx]

        # W, H = test_dataset.img_wh

        rays = samples.view(-1, samples.shape[-1]).to(device)

        res = renderer(rays, tensorf, chunk=4096, N_samples=N_samples, ndc_ray=ndc_ray, white_bg=white_bg,
                       device=device,
                       ret_opaque_map=True, palette=palette, new_palette=new_palette)

        rgb_map = res['rgb_map']
        depth_map = res['depth_map']
        true_idx = plt_color_decomposition(rgb_map,palette_rgb=palette,eps=eps)
        point_cloud = rays[...,:3] + rays[...,3:6] * torch.reshape(depth_map,(-1,1))
        for i in range(palette_number):
            point_cloud_tmp = point_cloud[true_idx[i]]
            point_clouds_idx[i].append(point_cloud_tmp)
    for i in range(palette_number):
        point_clouds_idx[i] = torch.reshape(torch.cat(point_clouds_idx[i],dim=0),(-1,3))


    out_filepath = './logs/point_cloud'
    if not os.path.exists(out_filepath):
        os.makedirs(out_filepath)

    if savePath is not  None:
        out_filepath = os.path.join(out_filepath,savePath)

    if not os.path.exists(out_filepath):
        os.makedirs(out_filepath)

    if filename is not  None:
        for i in range(palette_number):
            np.savetxt(f'{out_filepath}/{filename}_{i}',point_clouds_idx[i].cpu().numpy())

    else:
        for i in range(palette_number):
            np.savetxt(f'{out_filepath}/point_clouds_{i}.txt',point_clouds_idx[i].cpu().numpy())

    return point_clouds_idx