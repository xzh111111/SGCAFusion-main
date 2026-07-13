import os
import torch
import pickle
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms.functional as TF  # ← 新增第8行
import random                                     # ← 新增第9行


class CustomDataset(Dataset):
    def __init__(self, root, image_numbers, transform=None):
        self.root = root
        self.image_numbers = image_numbers
        self.transform = transform

    def __len__(self):
        return len(self.image_numbers)

    def __getitem__(self, idx):
        ir_path = os.path.join(self.root, "infrared/train/", f"{idx + 1}.jpg")
        vis_path = os.path.join(self.root, "visible/train/", f"{idx + 1}.jpg")
        visNF_path = os.path.join(self.root, "visible_focus_near/train/", f"{idx + 1}.jpg")
        visFF_path = os.path.join(self.root, "visible_focus_far/train/", f"{idx + 1}.jpg")

        ir_img = Image.open(ir_path).convert("L")
        vis_img = Image.open(vis_path).convert("L")
        visNF_img = Image.open(visNF_path).convert("L")
        visFF_img = Image.open(visFF_path).convert("L")

        if self.transform:
            ir_img = self.transform(ir_img)
            vis_img = self.transform(vis_img)
            visNF_img = self.transform(visNF_img)
            visFF_img = self.transform(visFF_img)

        # ↓ 新增：配对数据增强（四张图使用完全相同的随机变换）
        if random.random() > 0.5:
            ir_img    = TF.hflip(ir_img)
            vis_img   = TF.hflip(vis_img)
            visNF_img = TF.hflip(visNF_img)
            visFF_img = TF.hflip(visFF_img)

        if random.random() > 0.5:
            ir_img    = TF.vflip(ir_img)
            vis_img   = TF.vflip(vis_img)
            visNF_img = TF.vflip(visNF_img)
            visFF_img = TF.vflip(visFF_img)

        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            ir_img    = TF.rotate(ir_img, angle)
            vis_img   = TF.rotate(vis_img, angle)
            visNF_img = TF.rotate(visNF_img, angle)
            visFF_img = TF.rotate(visFF_img, angle)
        # ↑ 新增结束

        return ir_img, vis_img, visNF_img, visFF_img