import torch.nn as nn
import functools
import torch
import functools
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torch.autograd import Variable
from .vgg import VGG
import os
import torchvision.models.vgg as models

class MUST(nn.Module):
    def __init__(self, dataroot, input_dim, dim, style_dim, n_downsample, n_res, mlp_dim, activ='relu', pad_type='reflect'):
        super(MUST, self).__init__()
        # appearance encoder
        input_dim = 3
        SP_input_nc = 8
        self.enc_appearance = AppearanceEncoder(dataroot, 3, input_dim, dim, int(style_dim/SP_input_nc), norm='none', activ=activ, pad_type=pad_type) # layer: ch: 64,128,256,512, size: 1x2 (mean,var)

        # pose encoder
        input_dim = 19 # ch: 19->8
        self.enc_pose = PoseEncoder(3, n_res, input_dim, dim, 'in', activ, pad_type=pad_type, convGroup=1) # ch: 512, size: 32
        input_dim = 3
        dim = 512

        #generator
        self.gen = Generator(3, n_res, dim, input_dim, norm='in', activ=activ, pad_type=pad_type, convgroup=1)

    def forward(self, Pose, Img, Sem):
        Pose = F.interpolate(Pose, (256,256))
        Pose_code = self.enc_pose(Pose)
        Appearance_code = self.enc_appearance(Img, Sem)
        images_recon = self.gen(Appearance_code, Pose_code)
        images_recon = F.interpolate(images_recon, (256,176))
        return images_recon


class AppearanceEncoder(nn.Module):
    def __init__(self, dataroot, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(AppearanceEncoder, self).__init__()
        # self.vgg = models.vgg19(pretrained=True).features
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load(os.path.join(dataroot, 'vgg19-dcbb9e9d.pth')))
        self.vgg = vgg19.features

        for param in self.vgg.parameters():
            param.requires_grad_(False)

        self.cha_atten1 = ChannelAttention(dim)
        self.conv1 = Conv2dBlock(dim, dim//8, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type) # ch: 64->8, size: 256->256
        dim = dim*2
        self.cha_atten2 = ChannelAttention(dim)
        self.conv2 = Conv2dBlock(dim, dim//8, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type) # ch: 128->16, size: 128->128
        dim = dim*2
        self.cha_atten3 = ChannelAttention(dim)
        self.conv3 = Conv2dBlock(dim, dim//8, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type) # ch: 256->32, size: 64->64
        dim = dim*2
        self.cha_atten4 = ChannelAttention(dim)
        self.conv4 = Conv2dBlock(dim, dim//8, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)  # ch: 512->64, size: 32->32
        dim = dim*2

        self.padding = nn.ReflectionPad2d(50)


    def get_features(self, image, model, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1'}
        features = {}
        x = image
        # model._modules is a dictionary holding each module in the model
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def get_statistics(self, x, model='global'):
        if model == 'global':
            n,c,h,w = x.size()
            x = x.view(n*c, h*w)
            mean_x = torch.mean(x, 1, keepdim=True)
            var_x = torch.var(x, 1, keepdim=True)
            x = torch.cat([mean_x, var_x], 1)
            x = x.view(n, c, 1, 2)

        return x

    def must_module(self, x):
        out = {}

        sty_fea = self.get_features(x, self.vgg)
        x = sty_fea['conv1_1'] # 64*256*256
        x = self.cha_atten1(x)
        x = self.conv1(x)
        out['layer4'] = self.get_statistics(x, 'global') # n,c,1,2

        x = sty_fea['conv2_1'] # 128*128*128
        x = self.cha_atten2(x)
        x = self.conv2(x)
        out['layer3'] = self.get_statistics(x, 'global') # n,c,1,2

        x = sty_fea['conv3_1'] # 256*64*64
        x = self.cha_atten3(x)
        x = self.conv3(x)
        out['layer2'] = self.get_statistics(x, 'global') # n,c,1,2

        x = sty_fea['conv4_1'] # 512*32*32
        x = self.cha_atten4(x)
        x = self.conv4(x)
        out['layer1'] = self.get_statistics(x, 'global') # n,c,1,2

        return out

    def forward(self, x, sem):
        for i in range(sem.size(1)):
            semi = sem[:, i, :, :]
            semi = torch.unsqueeze(semi, 1)
            semi = semi.repeat(1, x.size(1), 1, 1)
            xi = x.mul(semi)
            xi = F.interpolate(xi, (256,256)) # size: 256,176->256,256
            
            n,c,h,w = xi.size()
            xi = xi.view(n,c,64,32,32)
            xi = xi[:,:,torch.randperm(64),:,:]
            xi = xi.view(n,c,h,w)

            if torch.rand(1) > 0.7:
                in_out = torch.rand(1)
                if in_out > 0.5:
                    xi_crop = xi[:,:,50:h-50,50:w-50]
                    xi = F.interpolate(xi_crop, (256,256))
                else:
                    xi_padding = self.padding(xi)
                    xi = F.interpolate(xi_padding, (256,256))

    
            #sem's dims: 0 is background, 1 and 3 are pants and skirts, 2 is hair, 4 is face, 5 is upper clothes, 6 is arms, 7 is legs.
            #For clothes style transfer, you can replace some part in the "sem" with that in another person image's appearance latent code.

            if i is 0:
                out = self.must_module(xi)
            else:
                out_ = self.must_module(xi)
                for layer in out:
                    out[layer] = torch.cat([out[layer], out_[layer]], dim=1)
        
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PoseEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type, convGroup=8):
        super(PoseEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type, convGroup=1)] # ch: 8->64, size: 256->256
        # downsampling blocks
        for i in range(n_downsample): # n_downsample=3
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type, convGroup=1)] # ch: 64->128->256->512, size: 256->128->64->32
            dim *= 2
        for i in range(n_downsample-1): # n_downsample=2
            self.model += [Conv2dBlock(dim, dim, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type, convGroup=1)] # ch: 512->512->512, size: 32->32->32

        self.model = nn.Sequential(*self.model)
        self.output_dim = dim  # output_dim： 512

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='in', activ='relu', pad_type='zero', convgroup=1):
        super(Generator, self).__init__()

        self.model = nn.ModuleList()
        # upsampling blocks
        dim = 512
        for i in range(n_upsample): # n_upsample: 3, ch: 512->256->128->64, size: 32->64->128->256
            self.model += [SMResBlocks(dim, dim // 2, True, norm=norm, activation=activ, pad_type=pad_type, convgroup=convgroup)]
            dim //= 2
        # ch: 64->64, size: 256->256
        self.model += [SMResBlocks(dim, dim, False, norm=norm, activation=activ, pad_type=pad_type, convgroup=convgroup)]

        self.model += [Conv2dBlock(dim, output_dim, 1, 1, 0, norm='none', activation='tanh', pad_type=pad_type)] # ch: 64->3, size: 256->256

    def forward(self, appearance_code, x):
        for i,layer in enumerate(self.model):
            if i < 4:
                x = layer(appearance_code['layer'+str(i+1)], x)
            else:
                x = layer(x)
        return x


class SMResBlocks(nn.Module):
    def __init__(self, fin, fout, up_flag, norm='bn', activation='relu', pad_type='zero', convgroup=1):
        super(SMResBlocks, self).__init__()

        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        self.conv_0 = nn.Conv2d(fin, fmiddle, 3, 1, 1, groups=convgroup)
        self.conv_1 = nn.Conv2d(fmiddle, fout, 3, 1, 1, groups=convgroup)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False, groups=convgroup)

        if norm == 'sn':
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        self.norm_0 = SMResBlock(fin, fin)
        self.norm_1 = SMResBlock(fin, fout)

        if self.learned_shortcut:
            self.norm_s = SMResBlock(fin, fin)

        self.up_flag = up_flag
        if self.up_flag is True:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
    
    def shortcut(self, x, statistics_code):
        if self.learned_shortcut:
            x_s = self.conv_s(self.actvn(self.norm_s(x, statistics_code)))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.relu(x)

    def forward(self, statistics_code, x):        
        x_s = self.shortcut(x, statistics_code)
        dx = self.conv_0(self.actvn(self.norm_0(x, statistics_code)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, statistics_code)))
        out = x_s + dx
        if self.up_flag is True:
            out = self.up(out)
        return out
        

class SMResBlock(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(SMResBlock, self).__init__()

        self.param_free_norm = nn.InstanceNorm2d(in_nc, affine=False)
        self.statis_affine = MLP(2*in_nc, 2*out_nc, 2*in_nc, 3, norm='none', activ='relu')
        
    def forward(self, x, statistics_code):       
        normalized = self.param_free_norm(x) # n,c,h,w

        n,c,h,w = statistics_code.size()
        statistics_code = statistics_code.view(statistics_code.size(0), -1)
        statistics_code = self.statis_affine(statistics_code)
        statistics_code = statistics_code.view(n,-1,h,w)

        gamma = statistics_code[:,:,:,1].unsqueeze(3)
        beta = statistics_code[:,:,:,0].unsqueeze(3)

        out = normalized * gamma + beta

        return out


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', in_norm_learnable=False, groupnormnum=8, convGroup=1):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            if in_norm_learnable == True:
                self.norm = nn.InstanceNorm2d(norm_dim, affine=True)
            else:
                self.norm = nn.InstanceNorm2d(norm_dim, affine=False)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'group':
            self.norm = nn.GroupNorm(groupnormnum, norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, groups=convGroup))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, groups=convGroup)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

