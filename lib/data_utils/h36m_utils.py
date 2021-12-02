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

import sys
sys.path.append('.')

import os
import cv2
import torch
import joblib
import argparse
import numpy as np
import pickle as pkl
import os.path as osp
from tqdm import tqdm

from lib.models import spin
from lib.data_utils.kp_utils import *
from lib.core.config import MP_DB_DIR, MP_DATA_DIR
from lib.utils.smooth_bbox import get_smooth_bbox_params
from lib.data_utils.occ_utils import load_occluders
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14
from lib.data_utils.feature_extractor import extract_features
from lib.utils.geometry import batch_rodrigues, rotation_matrix_to_angle_axis

NUM_JOINTS = 24
VIS_THRESH = 0.3
MIN_KP = 6
H36M = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

def read_data(folder, set, debug=False):

    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'shape': [],
        'pose': [],
        'bbox': [],
        'img_name': [],
        'features': [],
    }

    occluders = load_occluders('./data/VOC2012')

    model = spin.get_pretrained_hmr()

    h36m_data = os.path.join(folder, 'h36m_p1_{}.npz'.format(set))
    imgnames = h36m_data['imgname']
    centers = h36m_data['center']
    scales = h36m_data['scale']
    Ss = h36m_data['S']
    poses = h36m_data['pose']
    shapes = h36m_data['shape']
    parts = h36m_data['part']
    roots = h36m_data['root']

    J_regressor = None

    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    if set == 'test' or set == 'validation':
        J_regressor = torch.from_numpy(np.load(osp.join(MP_DATA_DIR, 'J_regressor_h36m.npy'))).float()

    start_idx, end_idx = None, None
    for i, img in enumerate(tqdm(imgnames)):
        img_ = img.split('/')[1].split('.')
        user_action = img_[0]
        camera, frm_id = img_[1].split('_')

        if start_idx is None:
            start_idx = i
            start_user_action = user_action
            start_camera = camera
        elif user_action != start_user_action or camera != start_camera:
            end_idx = i

        if start_idx is None or end_idx is None:
            continue

        num_frames = end_idx - start_idx
        pose = torch.from_numpy(poses[start_idx:end_idx]).float()
        shape = torch.from_numpy(shapes[start_idx:end_idx]).float()
        # trans = torch.from_numpy(trans['trans'][p_id]).float()
        j2d = parts[start_idx:end_idx]        # (765, 3, 18) 3dpw
        j2d = j2d[:, H36M, :]

        # ======== Align the mesh params ======== #

        output = smpl(betas=shape, body_pose=pose[:,3:], global_orient=pose[:,:3])
        # verts = output.vertices
        j3d = output.joints

        if J_regressor is not None:
            vertices = output.vertices
            J_regressor_batch = J_regressor[None, :].expand(vertices.shape[0], -1, -1).to(vertices.device)
            j3d = torch.matmul(J_regressor_batch, vertices)
            j3d = j3d[:, H36M_TO_J14, :]

        img_paths = []
        for i_frame in range(start_idx, end_idx):
            img_path = osp.join(folder, imgnames[i_frame])
            img_paths.append(img_path)

        bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(j2d, vis_thresh=VIS_THRESH, sigma=8)

        # process bbox_params
        c_x = bbox_params[:,0]
        c_y = bbox_params[:,1]
        scale = bbox_params[:,2]
        w = h = 150. / scale
        w = h = h * 1.1
        bbox = np.vstack([c_x,c_y,w,h]).T

        # process keypoints
        j2d[:, :, 2] = j2d[:, :, 2] > 0.3  # set the visibility flags
        # Convert to common 2d keypoint format
        perm_idxs = get_perm_idxs('h36m', 'common')
        j2d = j2d[:, perm_idxs]

        img_paths_array = np.array(img_paths)[time_pt1:time_pt2]
        dataset['vid_name'].append(np.array([f'{start_user_action}_{start_camera}']*num_frames)[time_pt1:time_pt2])
        dataset['frame_id'].append(np.arange(0, num_frames)[time_pt1:time_pt2])
        dataset['img_name'].append(img_paths_array)
        dataset['joints3D'].append(j3d.numpy()[time_pt1:time_pt2])
        dataset['joints2D'].append(j2d[time_pt1:time_pt2])
        dataset['shape'].append(shape.numpy()[time_pt1:time_pt2])
        dataset['pose'].append(pose.numpy()[time_pt1:time_pt2])
        dataset['bbox'].append(bbox)

        features = extract_features(model, occluders, img_paths_array, bbox,
                                    kp_2d=j2d[time_pt1:time_pt2], debug=debug, dataset='h36m', scale=1.2)
        dataset['features'].append(features)

        start_idx, end_idx = None, None

    for k in dataset.keys():
        dataset[k] = np.concatenate(dataset[k])
        print(k, dataset[k].shape)

    # Filter out keypoints
    indices_to_use = np.where((dataset['joints2D'][:, :, 2] > VIS_THRESH).sum(-1) > MIN_KP)[0]
    for k in dataset.keys():
        dataset[k] = dataset[k][indices_to_use]

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/h36m')
    args = parser.parse_args()

    debug = False

    dataset = read_data(args.dir, 'train', debug=debug)
    joblib.dump(dataset, osp.join(MP_DB_DIR, 'h36m_p1_train_50fps_occ_db.pt'))

    dataset = read_data(args.dir, 'train', debug=debug)
    joblib.dump(dataset, osp.join(MP_DB_DIR, 'h36m_p1_train.pt'))

    dataset = read_data(args.dir, 'test', debug=debug)
    joblib.dump(dataset, osp.join(MP_DB_DIR, 'h36m_p1_test.pt'))
