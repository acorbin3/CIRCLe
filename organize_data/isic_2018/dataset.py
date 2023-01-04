# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from pathlib import Path

import torch
from PIL import Image, ImageFile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import requests
import zipfile
import os
from sklearn.preprocessing import LabelEncoder

from organize_data.transforms import Compose, RandomRotation, RandomHorizontalFlip, Resize, PILToTensor, Normalize, \
    ConvertImageDtype

import torchvision.transforms as transforms


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
        # image.save("test_image_before_transform.png")

        mask_name = self.df.loc[self.df.index[idx], 'mask_path']
        mask = Image.open(mask_name)
        #mask.save("test_mask_before_transform.png")

        label = self.df.loc[self.df.index[idx], 'label_encoded']
        image_ita = self.df.loc[self.df.index[idx], 'ita']
        fitzpatrick = self.df.loc[self.df.index[idx], 'fizpatrick_skin_type'] - 1  # This is to have a range starting at zero
        if self.transform:
            image, mask = self.transform(image, mask)

        # print(f"dataset fetch image: {image.shape} {image.dtype}")
        #to_pil = transforms.ToPILImage()

        # pil_image = to_pil(image.type(torch.float32))
        # pil_image.save("test_fetch_image.png")

        #pil_image = to_pil(mask.type(torch.float32))
        #pil_image.save("test_after_mask_transform.png")

        return image, label, fitzpatrick, mask, image_ita
        # return image, label, fitzpatrick


def download_isic_2018_datasets():
    """
    Locations to get the ISIC 2018 task 3 dataset
    !wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip
    !wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip
    """
    # check to see if we already have downloaded the zip files and extracted the zip files

    image_count = len(list(Path("ISIC_2018").glob("**/*.jpg")))
    if image_count == 0:
        images_url = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip"
        print("Downloading isic2018 images")
        download_and_extract(images_url, "ISIC_2018/")
        print("Downloading isic2018 images. Complete!")
    else:
        print("isic2018 images already downloaded")

    image_count = len(list(Path("ISIC_2018/masks").glob("**/*.png")))
    if image_count == 0:
        masks_url = "https://isic2018task3masks.s3.amazonaws.com/isic_2018_mask_results1_2022_12_29.zip"
        print("Downloading isic2018 masks")
        download_and_extract(masks_url, "ISIC_2018/masks", create_root_dir=False)
        print("Downloading isic2018 masks. Complete!")

        print("Resizing masks")
        # the masks need to be resized to (600,450) to match the original image sizes.
        target_size = (600, 450)
        mask_directory = "ISIC_2018/masks"
        # Iterate over all the files in the directory
        for file in os.listdir(mask_directory):
            # Skip files that are not images
            if not file.endswith(".jpg") and not file.endswith(".png"):
                continue

            # Open the image
            im = Image.open(os.path.join(mask_directory, file))

            # Resize the image
            im_resized = im.resize(target_size)

            # Save the resized image
            im_resized.save(os.path.join(mask_directory, file))
        print("Resizing masks. Complete!")
    else:
        print("isic 2018 masks already downladed")

    gt_csv_count = len(list(Path("ISIC_2018_GT").glob("*.csv")))
    if gt_csv_count == 0:
        print("Donloading isic 2018 ground truth classification data")
        ground_truth_url = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip"
        download_and_extract(ground_truth_url, "ISIC_2018_GT/")
    else:
        print("isic 2018 ground truth classification data already downlaoded")


def download_and_extract(images_url, directory, create_root_dir=True):
    response = requests.get(images_url)
    open("temp.zip", "wb").write(response.content)
    # Unzip the file to the destination directory
    with zipfile.ZipFile("temp.zip", "r") as zip_ref:
        if create_root_dir:
            zip_ref.extractall(directory)
        else:
            zip_ref.extractall(directory, pwd=None, members=zip_ref.infolist())
    os.remove("temp.zip")


def get_cached_dataframe():
    """
    This function is supposed to read a saved version of a dataframe that has Fitzpatrick skin type data pre-calculated
    for the ISIC 2018 Task 3 dataset. The ground truth is already embedded into this dataframe and doesnt beed to be
    extracted from ISIC2018_Task3_Training_GroundTruth.zip
    """

    print("Creating dataframe")
    orig_images = []
    masks_images = []

    # get the file names of the images
    for file in Path("ISIC_2018").glob("**/*.jpg"):
        orig_images.append(file)

        # no masks for Task 3, for now use empty strings
        masks_images = [""] * len(orig_images)

    # Creating the main dataframe
    print("\t Looking for cached dataframe")
    isic_df = pd.DataFrame()
    for file in Path(".").glob("**/*.csv"):
        if "isic_2018" in file.name and "saved_data" in file.name:
            print("\t\t", file)
            isic_df = pd.read_csv(file)
            isic_df = isic_df.fillna("")
    isic_df["image_path"] = orig_images
    isic_df["mask_path"] = masks_images
    isic_df["image_id"] = isic_df["image_path"].apply(lambda x: x.name.split('.')[0])

    # Clean up columns for further process
    # replaces 1's with column name
    isic_df.loc[isic_df["MEL"] == 1.0, ["MEL"]] = "MEL"
    isic_df.loc[isic_df["NV"] == 1.0, ["NV"]] = "NV"
    isic_df.loc[isic_df["AKIEC"] == 1.0, ["AKIEC"]] = "AKIEC"
    isic_df.loc[isic_df["BKL"] == 1.0, ["BKL"]] = "BKL"
    isic_df.loc[isic_df["BCC"] == 1.0, ["BCC"]] = "BCC"
    isic_df.loc[isic_df["DF"] == 1.0, ["DF"]] = "DF"
    isic_df.loc[isic_df["VASC"] == 1.0, ["VASC"]] = "VASC"

    # Clear out the 0's
    isic_df.loc[isic_df["MEL"] == 0, ["MEL"]] = ""
    isic_df.loc[isic_df["NV"] == 0, ["NV"]] = ""
    isic_df.loc[isic_df["AKIEC"] == 0, ["AKIEC"]] = ""
    isic_df.loc[isic_df["BKL"] == 0, ["BKL"]] = ""
    isic_df.loc[isic_df["BCC"] == 0, ["BCC"]] = ""
    isic_df.loc[isic_df["DF"] == 0, ["DF"]] = ""
    isic_df.loc[isic_df["VASC"] == 0, ["VASC"]] = ""

    encoder = LabelEncoder()
    isic_df["label"] = isic_df[["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]].apply(lambda x: "".join(x), axis=1)
    isic_df["label_encoded"] = encoder.fit_transform(isic_df["label"])

    mask_directory = str(list(Path("ISIC_2018/masks").glob("**/*.png"))[0].parent)

    isic_df["mask_path"] = isic_df["image_id"].apply(lambda x: f"{mask_directory}/{x}.png")
    print("Creating dataframe. Complete!")
    return isic_df


def get_isic_2018_dataloaders(isic_df, batch_size=32, shuffle=True):
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
        transform=Compose([
            # RandomRotation(degrees=15),
            # RandomHorizontalFlip(),
            Resize(size=(128, 128)),
            PILToTensor(),
            ConvertImageDtype(torch.float),
            # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # this seems to really mess up the colors of the base image
        ])
    )

    transformed_val = ISIC2018SkinDataset(
        df=val,
        transform=Compose([
            Resize(size=(128, 128)),
            PILToTensor(),
            ConvertImageDtype(torch.float),
            # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # this seems to really mess up the colors of the base image
        ])
    )

    transformed_test = ISIC2018SkinDataset(
        df=test,
        transform=Compose([
            Resize(size=(128, 128)),
            PILToTensor(),
            ConvertImageDtype(torch.float),
            # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # this seems to really mess up the colors of the base image
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