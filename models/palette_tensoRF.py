import math
import random

import torch
import torch.nn.functional as F
from .tensoRF import TensorVMSplit
from .tensorBase import positional_encoding, RenderBufferProp
from .loss import soft_L0_norm
from .palette import FreeformPalette
from point_Model.point import point_cloud,point_cloud_classical,point_cloud_classical_num
from point_engine.train import get_class_index
'''
主要做调色板混合
'''
class PLTRender(torch.nn.Module):
    '''
    Color decomposition scheme: alpha blending
    '''
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128, alpha_blend=False, palette=None, learn_palette=False, palette_init='userinput', soft_l0_sharpness=24.,**kwargs):
        super().__init__()

        len_palette = len(palette)

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        self.n_dim = 3 + len_palette
        self.learn_palette = learn_palette
        self.soft_l0_sharpness = soft_l0_sharpness
        self.net1 = None
        self.net2 = None
        #调色板的训练
        if not learn_palette:
            self.palette = FreeformPalette(len_palette, is_train=False, initial_palette=palette)
        else:
            self.palette = FreeformPalette(len_palette, is_train=True, palette_init_scheme=palette_init, random_std=0.1, initial_palette=palette)
            
        self.render_buf_layout = [
            RenderBufferProp('rgb', 3, False, 'RGB'),
            RenderBufferProp('opaque', len_palette, True)]

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, len_palette - 1)  #alpha从2开始 所以比调色板长度少一维
        torch.nn.init.constant_(layer3.bias, 0)
        self.mlp = torch.nn.Sequential(layer1, torch.nn.LeakyReLU(inplace=True),
                                       layer2, torch.nn.LeakyReLU(inplace=True),
                                       layer3)
        
        self.n_dim += 1
        self.render_buf_layout.append(RenderBufferProp('sparsity_norm', 1, False))

        self.alpha_blend = alpha_blend
        if not alpha_blend:
            self.n_dim += 1
            self.render_buf_layout.append(RenderBufferProp('convexity_residual', 1, False))

    def weights_from_alpha_blending(self, logits):
        opaque = torch.sigmoid(logits)
        log_opq = F.logsigmoid(logits)
        log_wa = torch.cumsum(F.logsigmoid(torch.neg(logits)), dim=-1)
        #alpha混合计算公式
        w_0 = opaque[..., :1]
        w_a = torch.exp(log_wa[..., :-1] + log_opq[..., 1:])
        w_last = torch.exp(log_wa[..., -1:])
        bary_coord = torch.cat((w_0, w_a, w_last), dim=-1)
        # bary_coord guarantee sum to 1
        # assert torch.allclose(bary_coord.sum(dim=-1), torch.ones(()), atol=1e-3)
        return bary_coord, opaque

    #这里对网络计算
    def forward(self, pts, viewdirs, features, is_train=False, **kwargs):
        #pts xyz  (sample_num , 3)  features (sample_num,27)
        if kwargs['is_choose'] == True:
            if ('net1' in kwargs and kwargs['net1']):
                self.net1 = kwargs['net1']
                self.net1.to(pts.device)
            if ( 'net2' in kwargs and kwargs['net2']):
                self.net2 = kwargs['net2']

            if self.net1 and self.net2:
                pts_choice = []
                for i in range(len(self.net2)):
                    self.net2[f'model{i}'].to(pts.device)
                    pts_choice.append(get_class_index(self.net2[f'model{i}'](self.net1(pts.type(torch.float64)))))

        indata = [features, viewdirs]
        if self.feape > 0:
            indata.append(positional_encoding(features, self.feape))
        if self.viewpe > 0:
            indata.append(positional_encoding(viewdirs, self.viewpe))
        #这里卷积
        h = self.mlp(torch.cat(indata, dim=-1))

        palette = self.palette
        if not is_train and 'palette' in kwargs:
            palette = kwargs['palette'].to(pts.device)

            if 'new_palette' in kwargs and kwargs['new_palette'] :
                new_palette = kwargs['new_palette']

            #调试  以50%的概率选择调色盘
            # x = random.uniform(0,1)
            # if(x<0.5):
            #     palette = palette
            # else:
            #     palette = new_palette
            assert isinstance(palette, torch.Tensor)

        conv_residual = None
        if self.alpha_blend:
            bary_coord, opaque = self.weights_from_alpha_blending(h)
            sparsity_weight = torch.exp(-torch.linspace(0, 1., bary_coord.shape[-1])).to(bary_coord.device)
        else:
            opaque = torch.sigmoid(h)
            bary_coord = torch.cat([opaque, F.relu(1.0 - opaque.sum(-1, keepdim=True))], -1)
            sparsity_weight = torch.ones(bary_coord.shape[1]).to(bary_coord.device)
            conv_residual = torch.abs(1. - torch.sum(bary_coord, dim=-1, keepdim=True))

        if kwargs['is_choose'] == True:
            palette_all = torch.ones(size=(pts.shape[0],len(palette),3),dtype=torch.float64).to(pts.device)
            for i in range(len(palette)):
                index = pts_choice[i][1] > 0.9
                index2 = pts_choice[i][0][index].type(torch.long)
                tt = new_palette[i][index2]
                palette_all[index,i,:] = tt
                palette_all[index==False,i,:] = new_palette[i][-1]
            rgb = torch.sum(bary_coord.reshape(bary_coord.shape[0],len(palette),1) * palette_all,dim=1)
        else:
            rgb = bary_coord @ palette  # operator overload

        sparsity_weight = sparsity_weight.unsqueeze(0)
        sparsity = torch.sum(sparsity_weight * soft_L0_norm(bary_coord, scale=self.soft_l0_sharpness), dim=-1, keepdim=True)
        
        rend_buf = [rgb, bary_coord]
        rend_buf.append(sparsity)
        if conv_residual is not None:
            rend_buf.append(conv_residual)

        return torch.cat(rend_buf, dim=-1)

#tensorBase父类调用了子类的init_render_func
class PaletteTensorVM(TensorVMSplit):
    def init_render_func(self, shadingMode='PLT_Direct', pos_pe=6, view_pe=6, fea_pe=6, featureC=128,
                         palette=None, learn_palette=False, palette_init='userinput', soft_l0_sharpness=24., **kwargs):
        
        print('[init_render_func]', f"shadingMode={shadingMode}", f"pos_pe={pos_pe}", f"view_pe={view_pe}", f"fea_pe={fea_pe}",
                f"learn_palette={learn_palette}", f"palette_init={palette_init}")

        if shadingMode == 'PLT_AlphaBlend':
            alpha_blend = True 
        elif shadingMode == 'PLT_Direct':
            alpha_blend = False
        else:
            raise NotImplementedError


        return PLTRender(self.app_dim, view_pe, fea_pe, featureC, alpha_blend, palette, learn_palette, palette_init, soft_l0_sharpness,**kwargs).to(self.device)
    
    def get_palette_array(self):
        return self.renderModule.palette.get_palette_array()
    