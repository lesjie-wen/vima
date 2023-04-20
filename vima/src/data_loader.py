import os
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from einops import rearrange
from tqdm import tqdm

class VimaDataset(Dataset):
    def __init__(self,
                 task_type='rearrange_then_restore',
                 root_dir=''):  # 数据集根目录
        self.root_dir = root_dir
        self.task = task_type
        self.actions_max_len = 0
        self.data = list()
        super().__init__()
        # for task in os.listdir(root_dir):
        task_path = os.path.join(root_dir, task_type)
        assert os.path.isdir(task_path)

        # self.data[task] = list()

        task_all = os.listdir(task_path)
        task_all.sort()

        # 存在meta信息的时候自动去掉最后一个
        task_all = task_all[:-1]

        for each in tqdm(task_all[0:13]):

            path = os.path.join(task_path, each)
            #
            # img_folder_path = dict()
            # img_folder_path['rgb_front'] = os.path.join(path, "rgb_front")
            # img_folder_path['rgb_top'] = os.path.join(path, "rgb_top")
            #
            # assert os.path.isdir(img_folder_path['rgb_front'])
            # assert os.path.isdir(img_folder_path['rgb_top'])

            action_file_path = os.path.join(path, "action.pkl")
            obs_file_path = os.path.join(path, "obs.pkl")
            traj_file_path = os.path.join(path, "trajectory.pkl")

            with open(action_file_path, 'rb') as f:
                action_data = pickle.load(f)
            with open(obs_file_path, 'rb') as f:
                obs_data = pickle.load(f)
            with open(traj_file_path, 'rb') as f:
                traj_data = pickle.load(f)

            self.actions_max_len = max(self.actions_max_len, action_data['pose0_position'].shape[0])

            d = self._parse_pkl(action_data, traj_data, obs_data, path)

            self.data.append(d)

    def __getitem__(self, index):
        data = self.data[index]
        action = {k: torch.tensor(v) for k,v in data['action'].items()}
        prompt = data['prompt']
        obs_img = {k: torch.tensor(v) for k,v in data['rgb_dict'].items()}
        traj_meta = data['traj_meta']
        segm = {k: torch.tensor(v) for k,v in data['segm'].items()}
        end_effector = torch.tensor(data['end_effector'])

        if self.actions_max_len + 1 > end_effector.shape[0]:
            end_effector = torch.cat([end_effector, torch.zeros(self.actions_max_len + 1 - end_effector.shape[0])], dim=0)
        # all_dict = self.padding_dim({
        #     'rgb_dict': obs_img,
        #     'segm': segm,
        #     'action': action,
        # })

        meta_info = dict()
        meta_info['action_bounds'] = traj_meta['action_bounds']
        meta_info['n_objects'] = traj_meta['n_objects']
        meta_info['obj_id_to_info'] = list(traj_meta['obj_id_to_info'].keys())
        # img_paths = sorted(os.listdir(data['img_folder']))
        # imgs = []
        # for img_path in img_paths:
        #     img_path = os.path.join(data['img_folder'], img_path)
        #     img = Image.open(img_path)
        #     img = img.convert('RGB')
        #     img = torch.tensor(np.array(img)).permute(2, 0, 1)
        #     imgs.append(img)

        return {
            # 'target_action': all_dict['action'],
            # 'obs_img': all_dict['rgb_dict'],
            # 'segm': all_dict['segm'],

            'target_action': action,
            'obs_img': obs_img,
            'segm': segm,
            "prompt": prompt,
            'end_effector': end_effector,
            'total_steps': traj_meta['steps'],
            'prompt_assets': traj_meta['prompt_assets'],
            'meta_info': meta_info,
            # 'traj_meta': traj_meta
            # 'traj_meta': traj_meta,
        }

    def __len__(self):
        return len(self.data)

    def _parse_pkl(self, action, traj_meta, obs, path):
        d = dict()
        rgb_dict = {"front": [], "top": []}
        n_rgb_frames = len(os.listdir(os.path.join(path, f"rgb_front")))
        for view in ["front", "top"]:
            for idx in range(n_rgb_frames):
                # load {idx}.jpg using PIL
                rgb_dict[view].append(
                    rearrange(
                        np.array(
                            Image.open(os.path.join(path, f"rgb_{view}", f"{idx}.jpg")),
                            copy=True,
                            dtype=np.uint8,
                        ),
                        "h w c -> c h w",
                    )
                )
        rgb_dict = {k: np.stack(v, axis=0) for k, v in rgb_dict.items()}
        segm = obs.pop("segm")
        end_effector = obs.pop("ee")
        d['rgb_dict'] = rgb_dict
        d['segm'] = segm
        d['end_effector'] = end_effector
        d['action'] = action
        d['prompt'] = traj_meta['prompt']
        d['prompt_assets'] = traj_meta['prompt_assets']
        d['traj_meta'] = traj_meta
        return d

# 由于不同实例对应的动作、分割以及图片的数量维度(有的任务可能只需要3步就能完成有的可能需要6步)不一致因此将其扩充成统一尺寸
    def padding_dim(self, data):

        keys = ['rgb_dict', 'segm', 'action']
        for key in keys:
            max_len = self.actions_max_len
            first_k = list(data[key].keys())[0]
            s = data[key][first_k].shape
            if key != 'action':
                max_len += 1
            if s[0] < max_len:
                data[key] = {
                    k: torch.cat([data[key][k], torch.zeros(max_len - s[0], *data[key][k].shape[1:])], dim=0)
                    for k in data[key]}
            else:
                break
        return data

def vima_collate(batch):
    res = {k: list() for k in batch[0].keys()}
    for each in batch:
        for k in res.keys():
            if isinstance(each[k], dict):
                res[k].append(each[k].values())
            else:
                res[k].append(each[k])
    return res


if __name__ == '__main__':
    dataset = VimaDataset(root_dir="/Users/lesjie/PycharmProjects/vima/vimadata/vima_v6")
    print(dataset.actions_max_len)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=vima_collate)

    for batch in dataloader:
        action_batch = batch['target_action']
        obs_batch = batch['obs_img']
        # do something with the batch
