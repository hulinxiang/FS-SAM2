# -*- coding: utf-8 -*-
import os
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torchvision import tv_tensors
from PIL import Image
import numpy as np

from training.dataset.coco_pair_dataset_online import COCOPairOnlineRawDataset

class DatasetCOCO2017PairOnline(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, img_size, use_original_imgsize, seed=None,
                 pos_prob=1.0, min_mask_pixels=1500, cat_names="", repeat_times=1, include_crowd=False):
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.use_original_imgsize = use_original_imgsize
        self.benchmark = 'coco2017p'
        self.nclass = 80
        self.shot = 1
        self.seed = seed if seed is not None else 0

        if split == 'trn':
            images_dir = os.path.join(datapath, 'train2017', 'train2017')
            ann_path = os.path.join(datapath, 'annotations_trainval2017', 'annotations', 'instances_train2017.json')
        else:
            images_dir = os.path.join(datapath, 'val2017', 'val2017')
            ann_path = os.path.join(datapath, 'annotations_trainval2017', 'annotations', 'instances_val2017.json')

        self.raw = COCOPairOnlineRawDataset(
            images_dir=images_dir,
            ann_path=ann_path,
            pos_prob=pos_prob,
            min_mask_pixels=min_mask_pixels,
            cat_names=cat_names,
            seed=self.seed,
            include_crowd=include_crowd,
            repeat_times=repeat_times,
        )

        # 基于 COCO API 构建 cat_id -> [0..N-1] 连续索引（COCO instances 通常是 80 类）
        coco = self.raw.coco
        cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda x: x['id'])
        self.cat_id_to_idx = {c['id']: i for i, c in enumerate(cats)}
        self.nclass = len(self.cat_id_to_idx)  # 一般为 80
        self.class_ids = list(range(self.nclass))


    def set_epoch(self, epoch):
        # 控制随机性
        self.raw.rng.seed(self.seed + (epoch if epoch is not None else 0))

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        video, seg_loader = self.raw.get_video(idx)
        cid = int(video.video_name.split('_')[-1])       # COCO 类别 id
        cls_idx = self.cat_id_to_idx[cid]                # 连续索引 [0..79]
        f0, f1 = video.frames[0], video.frames[1]  # 帧0=支持图(A)，帧1=查询图(B)
        img0 = Image.open(f0.image_path).convert('RGB')
        img1 = Image.open(f1.image_path).convert('RGB')

        m0 = seg_loader.load(0).get(1, None)  # 支持图 mask（必须有）
        m1 = seg_loader.load(1).get(1, None)  # 查询图 mask（正样本才有）

        # 仅接受正样本对（co-seg同类配对）
        if m1 is None or (not bool(m0.any().item())):
            return self.__getitem__((idx + 1) % len(self))

        m0 = torch.from_numpy(np.array(m0, dtype=np.uint8))
        m1 = torch.from_numpy(np.array(m1, dtype=np.uint8))

        # 查询图 resize + transform
        q_img = img1.resize((self.img_size, self.img_size))
        if not self.use_original_imgsize:
            q_mask = F.interpolate(m1.unsqueeze(0).unsqueeze(0).float(), (self.img_size, self.img_size), mode='nearest').squeeze()
        else:
            q_mask = m1
        q_mask = tv_tensors.Mask(q_mask.float())
        q_img, q_mask = self.transform(q_img, q_mask)

        # 支持图 resize + transform
        s_img = img0.resize((self.img_size, self.img_size))
        s_mask = tv_tensors.Mask(F.interpolate(m0.unsqueeze(0).unsqueeze(0).float(), (self.img_size, self.img_size), mode='nearest').squeeze())
        s_img, s_mask = self.transform(s_img, s_mask)

        batch = {
            'query_name': os.path.basename(f1.image_path),
            'query_img': q_img,
            'query_mask': (q_mask > 0).to(torch.uint8),

            'support_names': [os.path.basename(f0.image_path)],
            'support_imgs': torch.stack([s_img]),
            'support_masks': torch.stack([(s_mask > 0).to(torch.uint8)]),

            'class_id': torch.tensor(cls_idx),  # 若需要真实类id，可扩展返回
        }
        return batch