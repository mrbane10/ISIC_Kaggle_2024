import numpy as np
import torch
import lightning as L
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import Resize, Normalize, Compose, ToTensor, CenterCrop, InterpolationMode
import pandas as pd
from sklearn.model_selection import train_test_split as ttp
from meta_features import *
from sklearn.preprocessing import StandardScaler


transform_ =\
    Compose([
            ToTensor(),
            Resize([256, 256], antialias=True,
                  interpolation=InterpolationMode.BICUBIC),
            CenterCrop([224, 224]),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class Dataset_Generator(Dataset):
    def __init__(self, data_dir, meta, label=None):
        super().__init__()
        self.data_dir = data_dir
        self.label = label
        self.meta = meta

    def __getitem__(self, idx):
        _dir = self.data_dir[idx]
        img = rgb_loader(_dir)
        img = transform_(img)
        meta = self.meta[idx].astype('float32')
        if self.label is None:
            return img, meta
        else:
            label = self.label[idx]
            return img, meta, label

    def __len__(self):
        return len(self.data_dir)

class DataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config['batch_size']
        self.dataset_dir = config['dataset_dir']
        self.num_imgs = config['num_imgs']
        train_dir = f'{self.dataset_dir}/train'
        scaler = StandardScaler()

        #dataframe
        df = pd.read_csv(
            '../../datasets/medical/skin cancer/isic/train-metadata.csv', low_memory=False)
        df = df.dropna(subset=['age_approx'], axis=0)
        df_sub = df.sample(self.num_imgs, random_state=0)

        df_norm = df_sub[df_sub.target == 0]
        df_anorm = df[df.target == 1]

        norm_meta = scaler.fit_transform(df_norm[features_cols])
        anorm_meta = scaler.transform(df_anorm[features_cols])

        #image, meta
        train_dir_ = list(map(lambda x: f'{train_dir}/image/{x}.jpg',
                                  df_norm.isic_id))
        self.train_dir, test_norm_dir,\
            self.meta_train, meta_norm_test = ttp(train_dir_, norm_meta,
                                                       test_size = 400, random_state=0)

        test_anorm_dir = list(map(lambda x: f'{train_dir}/image/{x}.jpg',
                                    df_anorm.isic_id))

        self.meta_test = np.concatenate([meta_norm_test, anorm_meta], axis=0)
        self.test_dir = test_norm_dir + test_anorm_dir
        self.test_label = [0] * len(test_norm_dir) + [1] * len(test_anorm_dir)


        #dataset
        self.ds_train = Dataset_Generator(self.train_dir, self.meta_train)
        self.ds_test = Dataset_Generator(self.test_dir, self.meta_test, self.test_label)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size,
                          num_workers=4, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size,
                          num_workers=4, persistent_workers=True, shuffle=False)
