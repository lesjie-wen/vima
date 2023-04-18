import os
import pickle
import cv2
import numpy as np


def gray_2_img(v1:np.ndarray):
    img = np.float32(v1)
    img = np.where(img == 1, 128, img)
    img = np.where(img == 3, 255, img)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return rgb_img
def parse_traj(traj_pkl:dict):
    scene = traj_pkl['prompt_assets']['scene']
    # rgb = scene['rgb']
    # segm = scene['segm']
    # img = np.transpose(np.float32(rgb['top']),(1,2,0))
    #
    # rgb_img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('./1.jpg',rgb_img)
    for k, v in scene.items():
        if not isinstance(v, dict):
            continue
        for k1, v1 in v.items():
            if k == 'segm':
                if k1 == 'obj_info':
                    continue
                rgb_img = gray_2_img(v1)
                # img = np.float32(v1)
                # img = np.where(img == 1, 128, img)
                # img = np.where(img == 3, 255, img)
                # rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = np.transpose(np.float32(v1),(1,2,0))
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'../../vima/data/{k}_{k1}.jpg', rgb_img)


def obs_2_img(seg_pkl:dict):
    for k, v in seg_pkl.items():
        for idx in range(seg_pkl[k].shape[0]):
            rgb_img = gray_2_img(seg_pkl[k][idx])
            cv2.imwrite(f'../../vima/data/obs_seg_{k}_{idx}.jpg', rgb_img)
