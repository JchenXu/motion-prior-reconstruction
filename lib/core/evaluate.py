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

import cv2
import torch
import shutil
import logging
import numpy as np
import os.path as osp
import tqdm

from lib.core.config import MP_DATA_DIR
from lib.utils.utils import move_dict_to_device, AverageMeter
from lib.utils.renderer import Renderer
from lib.utils.demo_utils import convert_crop_cam_to_orig_img

from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    compute_similarity_transform_torch,
)

logger = logging.getLogger(__name__)
class Evaluator():
    def __init__(
            self,
            test_loader,
            model,
            motion_prior,
            vposer,
            render=False,
            device=None,
    ):
        self.test_loader = test_loader
        self.model = model
        self.model.eval()
        self.motion_prior = motion_prior
        self.vposer = vposer
        self.render = render
        self.device = device

        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts', 'pred_theta'])

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def validate(self, target):
        if self.evaluation_accumulators is not None:
            for k,v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        J_regressor = torch.from_numpy(np.load(osp.join(MP_DATA_DIR, 'J_regressor_h36m.npy'))).float()

        move_dict_to_device(target, self.device)
        # <=============
        with torch.no_grad():
            inp = target['features']

            preds = self.model(inp, motion_prior=self.motion_prior, vposer=self.vposer, J_regressor=J_regressor)

            # convert to 14 keypoint format for evaluation
            # if self.use_spin:
            n_kp = preds[-1]['kp_3d'].shape[-2]
            pred_j3d = preds[-1]['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
            target_j3d = target['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
            pred_verts = preds[-1]['verts'].view(-1, 6890, 3).cpu().numpy()
            pred_theta = preds[-1]['theta'].view(pred_verts.shape[0], -1).cpu().numpy()
            target_theta = target['theta'].view(-1, 85).cpu().numpy()


            self.evaluation_accumulators['pred_verts'].append(pred_verts)
            self.evaluation_accumulators['pred_theta'].append(pred_theta)
            self.evaluation_accumulators['target_theta'].append(target_theta)

            self.evaluation_accumulators['pred_j3d'].append(pred_j3d)
            self.evaluation_accumulators['target_j3d'].append(target_j3d)
        # =============>

    def render_func(self, target, pred_theta, pred_verts):
        target['img_array'] = target['img_array'].cpu().numpy()
        target['bbox'] = target['bbox'].cpu().numpy()
        orig_height, orig_width = target['img_array'].shape[2:4]

        renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=False)

        img_array = target['img_array'].reshape(-1, orig_height, orig_width, 3)
        bbox = target['bbox'].reshape(img_array.shape[0], -1)

        for ii in range(img_array.shape[0]):
            bbox_ii = bbox[ii:ii + 1]
            bbox_ii[:, 2:] = bbox_ii[:, 2:] * 1.2
            img = img_array[ii:ii + 1]
            cam = pred_theta[ii:ii + 1, :3]

            orig_cam = convert_crop_cam_to_orig_img(
                cam=cam,
                bbox=bbox_ii,
                img_width=orig_width,
                img_height=orig_height
            )

            img = renderer.render(
                img,
                pred_verts[ii],
                cam=orig_cam[0],
                color=[1.0, 1.0, 0.9],
                mesh_filename=None
            )[0]

            h, w = img.shape[:2]
            new_h, new_w = int(h / 2), int(w / 2)
            new_h, new_w = new_h if new_h % 2 == 0 else new_h - 1, new_w if new_w % 2 == 0 else new_w - 1  # for ffmpeg
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def evaluate(self, target):
        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = np.vstack(v)

        pred_j3ds = self.evaluation_accumulators['pred_j3d']
        target_j3ds = self.evaluation_accumulators['target_j3d']

        pred_j3ds = torch.from_numpy(pred_j3ds).float()
        target_j3ds = torch.from_numpy(target_j3ds).float()

        # print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
        pred_pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
        target_pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0


        pred_j3ds -= pred_pelvis
        target_j3ds -= target_pelvis

        # Absolute error (MPJPE)
        errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        # S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
        S1_hat = torch.zeros_like(target_j3ds, device=target_j3ds.device, dtype=target_j3ds.dtype)
        for i in range(target_j3ds.shape[0]):
            S1_hat[i] = compute_similarity_transform_torch(pred_j3ds[i], target_j3ds[i])
        errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        pred_verts = self.evaluation_accumulators['pred_verts']
        pred_theta = self.evaluation_accumulators['pred_theta']
        target_theta = self.evaluation_accumulators['target_theta']

        m2mm = 1000

        pve = compute_error_verts(target_theta=target_theta, pred_verts=pred_verts) * m2mm
        accel = compute_accel(pred_j3ds) * m2mm
        accel_err = compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds) * m2mm
        mpjpe = errors * m2mm
        pa_mpjpe = errors_pa * m2mm

        return pve, accel, accel_err, mpjpe, pa_mpjpe, errors_pa.shape[0]

    def run(self):
        pves, accels, accel_errs, mpjpes, pa_mpjpes = [], [], [], [], []
        for i, target in enumerate(tqdm.tqdm(self.test_loader)):
            self.validate(target)
            pve, accel, accel_err, mpjpe, pa_mpjpe, num = self.evaluate(target)
            pves.append(pve)
            accels.append(accel)
            accel_errs.append(accel_err)
            mpjpes.append(mpjpe)
            pa_mpjpes.append(pa_mpjpe)

        eval_dict = {
            'mpjpe': np.concatenate(mpjpes, axis=0).mean(),
            'pa-mpjpe': np.concatenate(pa_mpjpes, axis=0).mean(),
            'pve': np.concatenate(pves, axis=0).mean(),
            'accel': np.concatenate(accels, axis=0).mean(),
            'accel_err': np.concatenate(accel_errs, axis=0).mean(),
        }
        log_str = ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
        print(log_str)

