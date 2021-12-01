# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import torch
import os.path as osp
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from lib.core.config import MP_DATA_DIR
from lib.models.spin import Regressor, hmr
from lib.utils.geometry import rot6d_to_rotmat

from human_motion_prior.train.motion_prior import MotionPrior, ContinousRotReprDecoder


class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x):
        n,t,f = x.shape
        x = x.permute(1,0,2) # NTF -> TNF
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t,n,f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1,0,2) # TNF -> NTF
        return y


class VIBE(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(MP_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(VIBE, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

        self.conv1 = nn.Conv1d(2048, 512, kernel_size=3, stride=2, padding=1)
        self.drop1 = nn.Dropout()
        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, stride=2, padding=1)
        self.drop2 = nn.Dropout()
        self.drop3 = nn.Dropout()
        self.feat2mpose = nn.Linear(1024, 256)
        self.feat2beta = nn.Linear(2048, 10)
        self.feat2rot = nn.Linear(2048, 1)
        
        self.convert_mat = torch.tensor([[[1, 0, 0],
                                          [0, -1, 0],
                                          [0, 0, -1]]],
                                        dtype=torch.float32, requires_grad=False).cuda()

    def forward(self, input, motion_prior=None, vposer=None, J_regressor=None):
        # input size NTF
        batch_size, seqlen = input.shape[:2]
        convert_mat = self.convert_mat.repeat([batch_size*seqlen, 1, 1])

        feature = self.encoder(input)
        feature = feature.reshape(-1, feature.size(-1))

        feature = self.drop3(feature)
        beta = self.feat2beta(feature.reshape(batch_size, seqlen, -1)[:, 0, :])
        beta = beta.unsqueeze(1).repeat([1, seqlen, 1]).reshape(-1, 10)

        feat_batch = feature.reshape(batch_size, seqlen, -1).transpose(1, 2)
        feat_batch = self.drop1(self.conv1(F.relu(feat_batch)))
        feat_batch = self.drop2(self.conv2(F.relu(feat_batch)))
        mpose = self.feat2mpose(feat_batch.reshape(batch_size, -1))
        motion_rec = motion_prior.decode(mpose, output_type='cont')
       
        pose_body = vposer.decode(motion_rec[:, :, :32].reshape(batch_size * 128, -1),
                                          output_type='cont').reshape(batch_size, 128, -1)
        root_orient_cont = motion_rec[:, :, 32:32+6]
        pose_body = pose_body[:, :seqlen, :].reshape(batch_size * seqlen, -1)
        root_orient_cont = root_orient_cont[:, :seqlen, :].reshape(batch_size * seqlen, -1)

        ################# OUTPUT GLOBAL ORIENT ##################
        # TODO: fix this dummy solution
        zeros1 = torch.zeros((batch_size, 1), device=input.device, dtype=input.dtype, requires_grad=False)
        zeros2 = torch.zeros((batch_size, 1), device=input.device, dtype=input.dtype, requires_grad=False)
        rot = self.feat2rot(feature.reshape(batch_size, seqlen, -1)[:, 0, :])
        rot_y = torch.cat([zeros1, rot, zeros2], dim=1)
        rot_y = MotionPrior.aa2matrot(rot_y)
        rot_y = rot_y.reshape(batch_size, 1, 3, 3).repeat([1, seqlen, 1, 1]).reshape(-1, 3, 3)

        root_orient_mat = rot6d_to_rotmat(root_orient_cont).view(-1, 3, 3)
        root_orient_mat = torch.bmm(convert_mat, root_orient_mat)      # adjust orient to dataset
        root_orient_mat_rect = torch.bmm(rot_y, root_orient_mat)
        root_orient = root_orient_mat_rect[:, :3, :2].reshape(batch_size * seqlen, 6)

        init_pose = torch.cat([root_orient, pose_body], dim=-1)
        smpl_output = self.regressor(feature, init_pose=init_pose, init_shape=beta, J_regressor=J_regressor, rot=None)

        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smpl_output


