import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

class OpenPoseImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512):
        self.dataset_dir = img_dir
        self.img_size = img_size
        
        # Load the JSONL file
        jsonl_path = os.path.join(img_dir, 'dataset.jsonl')
        with open(jsonl_path, 'r') as f:
            self.data = [json.loads(line.strip()) for line in f]
        
        print('OpenPoseImageDataset: ', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            json_data = self.data[idx]

            # Load the main image
            img_path = os.path.join(self.dataset_dir, json_data['image'])
            img = Image.open(img_path)
            img = c_crop(img)
            img = img.resize((self.img_size, self.img_size))
            # support gray scale images as well
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)

            # Load the conditioning image
            hint_path = os.path.join(self.dataset_dir, json_data['conditioning_image'])
            hint = Image.open(hint_path)
            hint = c_crop(hint)
            hint = hint.resize((self.img_size, self.img_size))
            hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
            hint = hint.permute(2, 0, 1)
            
            # Get the prompt text
            prompt = json_data['text']
            return img, hint, prompt

        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.data) - 1))


def openpose_dataset_loader(train_batch_size, num_workers, **args):
    dataset = OpenPoseImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
