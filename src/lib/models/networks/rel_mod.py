from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F


class BaseRelMod(nn.Module):

    def __init__(self,
                 in_channels=1024,
                 feat_dim=1024,
                 stages=3,
                 stride=8,
                 use_pos=False,
                 use_gt=False,
                 ignore_min=0,
                 add_type='sum',
                 ):
        super(BaseRelMod, self).__init__()
        self.stages = stages
        self.stride = stride
        self.feat_dim = feat_dim
        self.in_channels = in_channels
        self.use_pos = use_pos
        self.use_gt = use_gt
        self.ignore_min = ignore_min
        self.add_type = add_type
        assert self.stages > 0
        self.fcs = []
        self.kws = []
        self.qws = []
        self.vws = []
        self.us = []
        for i in range(self.stages):
            if i == 0:
                self.fcs.append(nn.Linear(self.in_channels, self.feat_dim))
            else:
                self.fcs.append(nn.Linear(self.feat_dim, self.feat_dim))
            self.kws.append(nn.Linear(self.feat_dim, self.feat_dim))
            self.qws.append(nn.Linear(self.feat_dim, self.feat_dim))
            self.vws.append(nn.Linear(self.feat_dim, self.feat_dim))
        if self.use_pos:
            self.wgs = []
            for i in range(self.stages):
                self.wgs.append(nn.Linear(self.feat_dim, 1))
            self.wgs = nn.ModuleList(self.wgs)
        self.fcs = nn.ModuleList(self.fcs)
        self.kws = nn.ModuleList(self.kws)
        self.qws = nn.ModuleList(self.qws)
        self.vws = nn.ModuleList(self.vws)
        if self.feat_dim != self.in_channels:
            self.adjust = nn.Linear(self.feat_dim, self.in_channels)
        if self.add_type == 'group':
            self.aff_alpha = nn.Parameter(torch.Tensor(self.groups, 2))
        if self.add_type == 'cond':
            self.aff_alpha = nn.Linear(self.feat_dim, self.groups * 2)
        self.downsample = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=self.stride, stride=self.stride, padding=0)
        self.upsample = nn.ConvTranspose2d(
            self.in_channels, self.in_channels, kernel_size=self.stride, stride=self.stride, padding=0)

    def init_weights(self):
        fcs = [self.kws, self.qws, self.fcs, self.vws]
        if self.use_pos:
            fcs.append(self.wgs)
        for em in fcs:  # linear layers
            for ei in em:
                nn.init.kaiming_uniform_(ei.weight, a=1)
                nn.init.constant_(ei.bias, 0)
        convs = [[self.downsample, self.upsample]]
        for em in convs:  # conv layers
            for ei in em:
                nn.init.normal_(ei.weight, std=0.01)
                nn.init.constant_(ei.bias, 0)
        if self.feat_dim != self.in_channels:
            nn.init.normal_(self.adjust.weight, std=0.01)
            nn.init.constant_(self.adjust.bias, 0)
        # for u in self.us:  # single parameter
        #     nn.init.normal_(u, std=0.01)
        if hasattr(self, 'adjust'):
            nn.init.kaiming_uniform_(self.adjust.weight, a=1)
            nn.init.constant_(self.adjust.bias, 0)
        if self.add_type == 'group':
            for p in [self.aff_alpha]:
                nn.init.normal_(p, std=0.01)
        if self.add_type == 'cond':
            for ei in [self.aff_alpha]:
                nn.init.kaiming_uniform_(ei.weight, a=1)
                nn.init.constant_(ei.bias, 0)

    def feat_aff(self, feats, ref_feats, qw, kw, position_embedding=None, gw=None, add='avg'):

        if position_embedding is not None:
            assert gw is not None
            # position_feat_1 = F.relu(gw(position_embedding))
            # # aff_weight, [num_rois, group, num_nongt_rois, 1]
            # aff_weight = position_feat_1.permute(2, 1, 3, 0)
            # # aff_weight, [num_rois, group, num_nongt_rois]
            # aff_weight = aff_weight.squeeze(3)

        # def ignore_part(aff_softmax_):
        #     aff_flatten: torch.Tensor = aff_softmax_.reshape(
        #         -1, aff_softmax_.shape[-1])
        #     ref_num = aff_flatten.size(1)
        #     ref_num_dist = max(int(ref_num * (1 - self.ignore_min)), 1)
        #     scores, inds = aff_flatten.topk(ref_num_dist, dim=1)
        #     pinds = torch.arange(aff_flatten.size(0)).to(
        #         aff_flatten.device) * ref_num
        #     pinds = pinds.unsqueeze(-1) + inds
        #     pinds = pinds.reshape(-1)
        #     mask = aff_flatten.new_zeros(aff_flatten.shape).reshape(-1)
        #     mask[pinds] = 1
        #     mask = mask.reshape(*aff_softmax_.shape)
        #     aff_softmax_ = F.softmax(aff_softmax_, dim=2) * mask
        #     k = aff_softmax_.sum(dim=2).unsqueeze(-1)
        #     s = 1. / k
        #     aff_softmax_ = aff_softmax_ * s
        #     return aff_softmax_

        q_feats = qw(feats)
        k_feats = kw(ref_feats)
        aff = torch.bmm(q_feats, k_feats.transpose(1, 2))

        aff_scale = (1.0 / math.sqrt(float(self.feat_dim))) * aff
        # aff_scale, [batch_size, num_pts, num_pts_ref]

        if position_embedding is not None:
            # weighted_aff = (aff_weight + 1e-6).log() + aff_scale
            weighted_aff = aff_scale
        else:
            weighted_aff = aff_scale

        # if outter_aff is not None and add != 'inner_only':
        #     if add == 'avg':

        #         aff_softmax = F.softmax(weighted_aff, dim=2)
        #         outter_aff_softmax = F.softmax(outter_aff, dim=2)

        #         aff_softmax_ = (
        #             outter_aff_softmax + aff_softmax) / 2.

        #     elif add == 'sum':
        #         aff = weighted_aff + outter_aff

        #         if self.ignore_min > 0:
        #             aff_softmax_ = ignore_part(aff)
        #         else:
        #             aff_softmax_ = F.softmax(aff, dim=2)

        #     elif add == 'max':
        #         aff_softmax_ = torch.max(weighted_aff, outter_aff)
        #         aff_softmax_ = F.softmax(aff_softmax_, dim=2)
        #     elif add == 'group':
        #         aff_alpha = self.aff_alpha.softmax(dim=1)
        #         aff_softmax_ = weighted_aff * \
        #             aff_alpha[:, 0].reshape(
        #                 1, -1, 1) + outter_aff * aff_alpha[:, 1].reshape(1, -1, 1)
        #         if self.ignore_min > 0:
        #             aff_softmax_ = ignore_part(aff_softmax_)
        #         else:
        #             aff_softmax_ = F.softmax(aff_softmax_, dim=2)
        #     elif add == 'cond':
        #         aff_alpha = self.aff_alpha(feats).reshape(-1, self.groups, 2)
        #         aff_alpha = aff_alpha.softmax(dim=2)
        #         aff_softmax_ = weighted_aff * \
        #             aff_alpha[:, :, 0].unsqueeze(
        #                 2) + outter_aff * aff_alpha[:, :, 1].unsqueeze(2)
        #         aff_softmax_ = F.softmax(aff_softmax_, dim=2)
        # else:
        #     aff_softmax_ = weighted_aff
        #     if self.ignore_min > 0:
        #         aff_softmax_ = ignore_part(aff_softmax_)
        #     else:
        #         aff_softmax_ = F.softmax(aff_softmax_, dim=2)
        # aff_softmax_reshape = aff_softmax_.reshape(
        #     aff_softmax_.shape[0] * aff_softmax_.shape[1], aff_softmax_.shape[2])

        return aff_scale

    def build_grid(resolution):
        ranges = [np.linspace(0., 1., num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        grid = np.concatenate([grid, 1.0 - grid], axis=-1)
        grid = torch.from_numpy(grid).view(1, -1, 4)
        return grid

    def cal_pos_embed(self, h, w, device='cuda'):
        if not hasattr(self, 'grids'):
            self.grids = {}
        if (h, w) not in self.grids:
            self.grids[(h, w)] = self.build_grid((h, w))
        pos_embed = self.grids[(h, w)].to(device)
        return pos_embed

    def forward(self, cur, ref, return_ref=False):
        bs, c, h, w = cur.shape
        assert ref.size(0) % bs == 0
        rep = ref.size(0) // bs
        cur_x = self.downsample(cur)
        h_d, w_d = cur_x.shape[-2:]
        cur_x = cur_x.view(bs, c, -1).permute(0, 2, 1)
        ref_x = self.downsample(ref).view(
            bs * rep, c, -1).permute(0, 2, 1).reshape(bs, -1, c)
        x_cur = cur_x
        x_ref = ref_x
        affs = []
        for i in range(self.stages):
            gw = self.wgs[i] if self.use_pos else None
            x_cur = F.relu(self.fcs[i](x_cur))
            x_ref = F.relu(self.fcs[i](x_cur))
            pos_emb = None
            if self.use_pos:
                pos_emb = self.cal_pos_embed(h, w, device=x_cur.device)
            aff_1 = self.feat_aff(
                x_cur, x_ref, self.qws[i], self.kws[i], pos_emb, gw, add=self.add_type)
            aff_2 = self.feat_aff(
                x_ref, x_ref, self.qws[i], self.kws[i], pos_emb, gw, add=self.add_type)
            affs.append(aff_1)
            # output_t, [batch_size, num_pts, feat_dim]
            output_cur = torch.bmm(aff_1, x_ref)

            # linear_out, [batch_size, num_pts, feat_dim]
            linear_cur = self.vws[i](output_cur)

            x_cur = x_cur + linear_cur

            if i != self.stages - 1 and not return_ref:
                # linear_out, [batch_size, num_pts_ref, feat_dim]
                output_ref = torch.bmm(aff_2, x_ref)

                # linear_out, [batch_size, num_pts_ref, feat_dim]
                linear_ref = self.vws[i](output_ref)

                x_ref = x_ref + linear_ref
        if self.feat_dim != self.in_channels:
            x_cur = self.adjust(x_cur)
            if return_ref:
                x_ref = self.adjust(x_ref)
        x_cur = self.upsample(x_cur.view(
            bs, h_d, w_d, c).permute(0, 3, 1, 2))
        if return_ref:
            x_ref = self.upsample(x_ref.view(
                bs * rep, h_d, w_d, c).permute(0, 3, 1, 2))
            return x_cur, x_ref, affs
        return x_cur, affs
