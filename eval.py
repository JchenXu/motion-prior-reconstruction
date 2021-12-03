# This code is developed based on VIBE <https://github.com/mkocabas/VIBE>

import os
import torch

from lib.dataset import ThreeDPW_eval3dpw
from lib.models import VIBE
from lib.core.evaluate import Evaluator
from lib.core.config import parse_args, MP_DATA_DIR
from torch.utils.data import DataLoader

from human_motion_prior.train.motion_prior import MotionPrior, ContinousRotReprDecoder
from human_motion_prior.tools.model_loader import load_vposer
from configer import Configer

def main(cfg):
    print('...Evaluating on 3DPW test set...')

    model = VIBE(
        n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        seqlen=cfg.DATASET.SEQLEN,
        hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
        pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR,
        add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
        bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
        use_residual=cfg.MODEL.TGRU.RESIDUAL,
    ).to(cfg.DEVICE)

    if cfg.TRAIN.PRETRAINED != '' and os.path.isfile(cfg.TRAIN.PRETRAINED):
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED)
        best_performance = checkpoint['performance']
        tmp = {}
        for k, v in checkpoint['gen_state_dict'].items():
            if "init_shape" == k or "init_cam" == k:
                continue

            tmp[k] = v
        model.load_state_dict(tmp)
        print(f'==> Loaded pretrained model from {cfg.TRAIN.PRETRAINED}...')
        print(f'Performance on 3DPW test set {best_performance}')
    else:
        print(f'{cfg.TRAIN.PRETRAINED} is not a pretrained model!!!!')
        exit()

    test_db = ThreeDPW_eval3dpw(set='test', seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)

    test_loader = DataLoader(
        dataset=test_db,
        batch_size=1,
        shuffle=False,
    )

    mp_path = os.path.join(MP_DATA_DIR, cfg.TRAIN.MP_PATH)
    ini_path = os.path.join(MP_DATA_DIR, cfg.TRAIN.MP_INI)
    ps = Configer(default_ps_fname=ini_path)  # This is the default configuration

    mp = MotionPrior(num_neurons=ps.num_neurons, latentD=ps.latentD, latentD_t=ps.latentD_t, dense_freq=ps.dense_freq, block_size=5, frequency_num=20, frame_num=ps.frame_num, use_cont_repr=ps.use_cont_repr)
    state_dict = torch.load(mp_path, map_location='cpu')
    new_state_dict = {k[7:]:v for k, v in state_dict.items()}
    mp.load_state_dict(new_state_dict, strict=False)
    mp = mp.eval().cuda()

    mp_dir = cfg.TRAIN.MP_DIR
    expr_dir = 'human_motion_prior/models/pre_trained/vposer_v1_0'
    vposer, _ = load_vposer(os.path.join(mp_dir, expr_dir), vp_model='snapshot')
    vposer = vposer.eval().cuda()

    Evaluator(
        model=model,
        motion_prior=mp,
        vposer=vposer,
        device=cfg.DEVICE,
        test_loader=test_loader,
    ).run()


if __name__ == '__main__':
    cfg, cfg_file = parse_args()

    main(cfg)
