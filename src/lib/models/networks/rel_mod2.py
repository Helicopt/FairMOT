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

from .rel_mod import BaseRelMod


class RelModV2(BaseRelMod):

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
            # x_cur = F.relu(self.fcs[i](x_cur))
            # x_ref = F.relu(self.fcs[i](x_cur))
            x_cur = F.normalize(x_cur)
            x_ref = F.normalize(x_ref)
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
            linear_cur = self.adjust(linear_cur)
            if return_ref:
                linear_ref = self.adjust(linear_ref)
        linear_cur = self.upsample(linear_cur.view(
            bs, h_d, w_d, c).permute(0, 3, 1, 2))
        x_cur = cur + linear_cur
        if return_ref:
            linear_ref = self.upsample(linear_ref.view(
                bs * rep, h_d, w_d, c).permute(0, 3, 1, 2))
            x_ref = ref + linear_ref
            return x_cur, x_ref, affs
        return x_cur, affs
