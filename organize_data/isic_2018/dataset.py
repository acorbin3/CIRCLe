# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class ISIC2018SkinDataset():
    def __init__(self, df, transform=None):
        """
        Args:
            df: Main dataframe with all the data organized already
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.loc[self.df.index[idx], 'image_path']
        image = Image.open(img_name)

        mask_name = self.df.loc[self.df.index[idx], 'mask_path']
        mask = Image.open(mask_name)

        label = self.df.loc[self.df.index[idx], 'label_encoded']
        fitzpatrick = self.df.loc[self.df.index[idx], 'fizpatrick_skin_type']
        if self.transform:
            image = self.transform(image)

        return image, mask, label, fitzpatrick


def get_isic_2018_dataloaders(isic_df, batch_size=8, shuffle=True):
    all_domains = [1, 2, 3, 4, 5, 6]

    # group index based on FSK. Split into 80/20 for training, test. then 50/50 for test and validation
    print("Splitting up the dataset into train,test, validation datasets")
    grouped = isic_df.groupby("fizpatrick_skin_type")
    group_indexes = grouped.indices

    train_indexes = []
    test_indexes = []
    val_indexes = []

    for group, index_list in group_indexes.items():
        index_train, index_test, _, _ = train_test_split(index_list, index_list, test_size=0.2, random_state=42)
        train_indexes += list(index_train)

        index_test, index_val, _, _ = train_test_split(index_test, index_test, test_size=0.5, random_state=42)
        test_indexes += list(index_test)
        val_indexes += list(index_val)

        print("fizpatrick_skin_type:", group, len(index_list))
        print(f"\t train {len(index_train)}")
        print(f"\t test {len(index_test)}")
        print(f"\t val {len(index_val)}")

    print(f"total_train: {len(train_indexes)} {len(train_indexes) / len(isic_df) * 100}")
    print(f"total_test: {len(test_indexes)} {len(test_indexes) / len(isic_df) * 100}")
    print(f"total_val: {len(val_indexes)} {len(val_indexes) / len(isic_df) * 100}")

    train = isic_df.iloc[train_indexes]
    test = isic_df.iloc[test_indexes]
    val = isic_df.iloc[val_indexes]

    print(f"train size: {len(train)}")
    print(f"test size: {len(test)}")
    print(f"val size: {len(val)}")

    for s in all_domains:
        print("\ttrain: skin type", s, ":", len(train[train['fizpatrick_skin_type'] == s]))

    print("----")
    for s in all_domains:
        print("\ttest: skin type", s, ":", len(test[test['fizpatrick_skin_type'] == s]))

    print("----")
    for s in all_domains:
        print("\tval: skin type", s, ":", len(val[val['fizpatrick_skin_type'] == s]))

    print("train size:", len(train))
    print("val size:", len(val))
    print("train skin types:", train.fizpatrick_skin_type.unique())
    print("val skin types:", val.fizpatrick_skin_type.unique())
    label_codes = sorted(list(train['label'].unique()))
    print("train skin conditions:", len(label_codes))
    label_codes1 = sorted(list(val['label'].unique()))
    print("val skin conditions:", len(label_codes1))

    transformed_train = ISIC2018SkinDataset(
        df=train,
        transform=transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    )

    transformed_val = ISIC2018SkinDataset(
        df=val,
        transform=transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    transformed_test = ISIC2018SkinDataset(
        df=test,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        transformed_train,
        batch_size=batch_size,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        transformed_val,
        batch_size=batch_size,
        shuffle=shuffle, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        transformed_test,
        batch_size=batch_size,
        shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader