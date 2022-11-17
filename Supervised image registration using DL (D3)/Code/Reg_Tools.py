import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
# from .. import default_unet_features
# from . import layers

def Torchinterp(src, phiinv):  #src:[b, 1, 64, 64, 64]     phiinv: [b, 64, 64, 64, 3]
    if(src.shape[-3]==1 and src.shape[-4]==1):
        src = src.squeeze(-3)
        phiinv = phiinv[...,0:2].squeeze(-4)
    mode='bilinear'
    shape = phiinv.shape[1:-1] 
    # normalize deformation grid values to [-1, 1] 
    for i in range(len(shape)):
        phiinv[...,i] = 2 * (phiinv[...,i] / (shape[i] - 1) - 0.5)
    return nnf.grid_sample(src, phiinv, align_corners=False,mode = mode, padding_mode= 'zeros')

def get_grid2(imagesize, device):
    size = (imagesize,imagesize,imagesize)
    # create sampling grid
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.to(device)
    return grid



######  3D network  #####

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,  #(100, 100)
                 infeats=None,  # 2
                 nb_features=None,  #[[16, 64], [64, 64, 16, 16]]
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super(Unet, self).__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features #[16, 64], [64, 64, 16, 16]
        nb_dec_convs = len(enc_nf)  #2
        final_convs = dec_nf[nb_dec_convs:]  #[16, 16]
        dec_nf = dec_nf[:nb_dec_convs]  #[64, 64]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1  #3
        self.nb_levels = 3

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels  #max_pool : [2, 2, 2]

        # cache downsampling / upsampling operations
        # MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        # self.pooling = [MaxPooling(s) for s in max_pool]
        # self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        MPooling = getattr(nn, 'Conv%dd' % ndims)
        MUpsampliing = getattr(nn, 'ConvTranspose%dd' % ndims)
        # self.pooling = []
        # self.upsampling = []
        self.pooling = nn.ModuleList()
        self.upsampling = nn.ModuleList()

        # configure encoder (down-sampling path)
        prev_nf = infeats #2
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]  #enc_nf [16,64]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            self.pooling.append(MPooling(prev_nf,prev_nf,kernel_size=4,stride=2,padding=1))
            nn.init.xavier_uniform_(self.pooling[-1].weight)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)  #encoder_nfs:[2, 16, 64]  -> [64, 16,  2]
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]  #dec_nf : [64, 64]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            self.upsampling.append(MUpsampliing(prev_nf,prev_nf,kernel_size=4,stride=2,padding=1))
            nn.init.xavier_uniform_(self.upsampling[-1].weight)

            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf
    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)
            # if(level<len(self.pooling)-1):
            #     x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            x = self.upsampling[level](x)
            if not self.half_res or level < (self.nb_levels - 2):
                # if(level<len(self.upsampling)-1):
                #     x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)
        
        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)
        return x

class VxmDense(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super(VxmDense, self).__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, 3, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.flow.weight)
        # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))


        # # configure optional resize layers (downsize)
        # if not unet_half_res and int_steps > 0 and int_downsize > 1:
        #     self.resize = layers.ResizeTransform(int_downsize, ndims)
        # else:
        #     self.resize = None

        # # resize to full res
        # if int_steps > 0 and int_downsize > 1:
        #     self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        # else:
        #     self.fullsize = None

        # # configure bidirectional training
        # self.bidir = bidir


    def forward(self, x):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        # [1, 1, 160, 224],[1, 1, 160, 224] ----> [1, 2, 160, 224]
        # print (source.shape)
        # print (target.shape)
        # x = torch.cat([source, target], dim=0)
        x = self.unet_model(x)   #[1, 16, 160, 224]
        x = self.flow(x)
        # print(x.shape)
        return x
        # print("self.unet_model", x.requires_grad)

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1, bias = True)
        nn.init.xavier_uniform_(self.main.weight)
        self.activation = nn.PReLU(out_channels)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out






def loadParameter(net, checkpoint_path, device = None):
    if(device):
        # Load saved parameters
        checkpoint = torch.load(checkpoint_path,map_location = device)
        net.load_state_dict(checkpoint['model_state_dict'])
        # optimizer = torch.optim.Adam(net.parameters())
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("lambda storage, loc: storage.cuda(1)")
        checkpoint = torch.load(checkpoint_path,map_location = lambda storage, loc: storage.cuda(1))
        net.load_state_dict(checkpoint['model_state_dict'])

def getUnetT3D():
    net = VxmDense(inshape = (128,128,128),
     nb_unet_features= [[16,64],[64,64,16,16]],
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res= False)
    return net