# %%
import os
import sys
import cv2
sys.path.insert(0, './')
import torch
import numpy as np
import imageio
import glob
from einops import rearrange
from matplotlib import pyplot as plt


# %%
from engine.trainer import Trainer
from engine.eval import evaluation_path
from data import dataset_dict
from utils.opt import config_parser
from utils.vis import plot_palette_colors, visualize_depth_numpy, visualize_palette_components_numpy
from utils.color import rgb2hex, hex2rgb
from utils.ray import get_rays, ndc_rays_blender


# %% md
## Utils
# %%
def print_divider():
    print()



def render_one_view(test_dataset, tensorf, c2w, renderer, N_samples=-1,
                    white_bg=False, ndc_ray=False, new_palette=None, palette=None, device='cuda'):
    torch.cuda.empty_cache()

    near_far = test_dataset.near_far

    if palette is None and hasattr(tensorf, 'get_palette_array'):
        palette = tensorf.get_palette_array().cpu()

    # 测试数据图像的宽高
    W, H = test_dataset.img_wh

    # 旋转矩阵
    c2w = torch.FloatTensor(c2w)
    # 根据方向和旋转矩阵 得到光线位置和方向
    rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)

    if ndc_ray:
        rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
    rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
    # render_one_view(ds_test_dataset, model, c2w, trainer.renderer, palette=torch.from_numpy(new_palette),
    # N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=trainer.device)
    # trainer.render 等于 chunkify_render
    res = renderer(rays, tensorf, chunk=2048, N_samples=N_samples, new_palette=new_palette, palette=palette,
                   ndc_ray=ndc_ray, white_bg=white_bg, device=device, ret_opaque_map=True)

    rgb_map = res['rgb_map']
    depth_map = res['depth_map']

    rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

    rgb_map = (rgb_map.numpy() * 255).astype('uint8')

    depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

    is_vis_plt = (palette is not None) and ('opaque_map' in res)
    plt_decomp = None
    # 获得调色板分解图
    if is_vis_plt:
        opaque = rearrange(res['opaque_map'], '(h w) c-> h w c', h=H, w=W).cpu()
        plt_decomp = visualize_palette_components_numpy(opaque.numpy(), palette.numpy())
        plt_decomp = (plt_decomp * 255).astype('uint8')

    return rgb_map, depth_map, plt_decomp


# %%

# %% md
## Config
# %%
# Make paths accessible by this notebook
path_redirect = [
    # option name, path in the config, redirected path
    ('palette_path', '../data_palette', './data_palette')
]
# %%
run_dir = './logs/fern/'
ckpt_path = None
out_dir = os.path.join(run_dir, 'demo_out')

print('Run dir:', run_dir)
print('Demo output dir:', out_dir)
# %% md
## Load and Setup
# %%
# Read args
parser = config_parser()
# 对args.txt里的参数进行获取
config_path = os.path.join(run_dir, 'args.txt')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        args, remainings = parser.parse_known_args(args=[], config_file_contents=f.read())

        # override ckpt path
        if ckpt_path is not None:
            setattr(args, 'ckpt', ckpt_path)

        # redirect path
        for entry in path_redirect:
            setattr(args, entry[0], getattr(args, entry[0]).replace(entry[1], entry[2]))

        print('Args loaded:', args)
else:
    print(f'ERROR: cannot read args in {run_dir}.')
print_divider()

# Setup trainer
print('Initializing trainer and model...')
ckpt_dir = os.path.join(run_dir, 'checkpoints')
tb_dir = os.path.join(run_dir, 'tensorboard')
# 训练器
trainer = Trainer(args, run_dir, ckpt_dir, tb_dir)
# 模型
model = trainer.build_network()
model.eval()
print_divider()

# Create downsampled dataset
dataset = dataset_dict[args.dataset_name]
ds_test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train * 2., is_stack=True)
print('Downsampled dataset loaded')

# %% md
## Palette Editing
# %%
palette_prior = np.load('palette_rgb.npy')
palette = model.renderModule.palette.get_palette_array().detach().cpu().numpy()
# %%
print('Initial palette prior:')
plot_palette_colors(palette_prior,'palette_prior_image.jpg')
print(palette)
plot_palette_colors(palette,'palette_image.jpg')
print(palette_prior)
# %%

palette_num = palette.shape[0]

palette_max = 50

palette_color = np.load('./data_palette/fern/rgb_palette.npy')
print("palette_color = \n",palette_color)
plot_palette_colors(palette_color,'palette_color.jpg')

new_palette = torch.ones((palette_num,palette_max,palette_num,3))
new_palette[...,:,:] = palette.copy()
print(new_palette)

# %%
print('Optimized palette:')
new_palette = palette.clip(0., 1.)

# new_palette = new_palette.clip(0.5, 0.7)
print(new_palette)
plot_palette_colors(new_palette,'new_palette_image.jpg')
# %%



print('Palette for rendering:')


def render_one():
    render_cam_idx = 1

    c2w = ds_test_dataset.poses[render_cam_idx]
    white_bg = ds_test_dataset.white_bg
    ndc_ray = args.ndc_ray

    with torch.no_grad():
        rgb, depth, plt_decomps = render_one_view(ds_test_dataset, model, c2w, trainer.renderer,
                                                  palette=torch.from_numpy(palette),
                                                  new_palette=torch.from_numpy(new_palette),
                                                  N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=trainer.device)

    fig, axes = plt.subplots(1, 2, figsize=(16, 16))
    axes[0].set_axis_off()
    axes[0].imshow(rgb)
    axes[1].set_axis_off()
    axes[1].imshow(depth)

    fig, axes = plt.subplots(1, 1, figsize=(16, 8))
    axes.set_axis_off()
    axes.imshow(plt_decomps)

# Run the cells below to save this editing

'''Modify this to name this editing'''
edit_name = 'red_chair'
def save_palette():

    assert edit_name

    out_fn = f'rgb_palette{"_" + edit_name if edit_name else ""}'
    out_path = os.path.join(out_dir, f'{out_fn}.npy')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if os.path.exists(out_path):
        print('Error: file exists. Please specify another `edit_name`.')
    else:
        np.save(out_path, new_palette)
        print('Save palette to', out_path)


# %%
    '''Choose between 'test' / 'path' '''
def save():
    cam_poses = 'test'

    save_dir = os.path.join(out_dir, f'render_{cam_poses}{"_" + edit_name if edit_name else ""}')

    if os.path.exists(save_dir):
        print('Error: directory exists. Please specify another `edit_name`.')
    else:
        c2ws = trainer.test_dataset.poses if cam_poses == 'test' else trainer.test_dataset.render_path
        if cam_poses == 'test' and args.dataset_name == 'llff':
            c2ws = c2ws[::8, ...]
        white_bg = trainer.test_dataset.white_bg
        ndc_ray = trainer.args.ndc_ray

        print('Save renderings to', save_dir)
        print('=== render path ======>', c2ws.shape)
        with torch.no_grad():
            evaluation_path(trainer.test_dataset, model, c2ws, trainer.renderer, save_dir,
                            palette=torch.from_numpy(palette), new_palette=torch.from_numpy(new_palette),
                            N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, save_video=True, device=trainer.device)
    # %%

# %%
