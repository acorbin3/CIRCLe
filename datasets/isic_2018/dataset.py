# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from pathlib import Path

import cv2
import torch
from PIL import Image, ImageFile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import requests
import zipfile
import os
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

from datasets.transforms import Compose, RandomRotation, RandomHorizontalFlip, Resize, PILToTensor, Normalize, \
    ConvertImageDtype

import torchvision.transforms as transforms
from torch.utils.data import Dataset


class ISIC2018SkinDataset(Dataset):
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

        transformed_image_name = self.df.loc[self.df.index[idx], 'transformed_path']
        transformed_image = Image.open(transformed_image_name)

        label = self.df.loc[self.df.index[idx], 'label_encoded']

        # Not using mask anymore since we are using cached transformed images
        if self.transform:
            image, transformed_image = self.transform(image, transformed_image, target_is_mask=False)

        # print(f"dataset fetch image: {image.shape} {image.dtype}")
        # to_pil = transforms.ToPILImage()

        # pil_image = to_pil(image.type(torch.float32))
        # pil_image.save("test_fetch_image.png")

        # pil_image = to_pil(mask.type(torch.float32))
        # pil_image.save("test_after_mask_transform.png")

        return image, label, transformed_image
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

    # cached image transforms
    transform_image_count = len(list(Path("ISIC_2018/transformed").glob("**/*.jpg")))
    if transform_image_count == 0:
        print("Downloading transformed iaages")
        transform_image_url = "https://isic2018task3masks.s3.amazonaws.com/transformed_images_2023_01_08.zip"
        download_and_extract(transform_image_url, "ISIC_2018/transformed/")


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
    for file in Path("ISIC_2018/ISIC2018_Task3_Training_Input").glob("**/*.jpg"):
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
    transform_directory = str(list(Path("ISIC_2018/transformed/ISIC_2018/transformed/").glob("**/*.jpg"))[0].parent)

    isic_df["mask_path"] = isic_df["image_id"].apply(lambda x: f"{mask_directory}/{x}.png")
    isic_df["transformed_path"] = isic_df["image_id"].apply(lambda x: f"{transform_directory}/{x}.jpg")
    print("Creating dataframe. Complete!")
    return isic_df


def compute_img_mean_std(isic_df, image_size):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """

    img_h, img_w = image_size, image_size
    imgs = []
    means, stdevs = [], []

    for index, row in isic_df.iterrows():
        # for i in tqdm(range(len(image_paths))):
        img = cv2.imread(str(row["image_path"]))
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means, stdevs


def get_isic_2018_dataloaders(isic_df, batch_size=32, image_size=128, shuffle=True):
    all_domains = [1, 2, 3, 4, 5, 6]

    # group index based on FSK. Split into 80/20 for training, test. then 50/50 for test and validation
    print("Splitting up the dataset into train,test, validation datasets based on the skin condition")
    grouped = isic_df.groupby(["label", "fizpatrick_skin_type"])
    group_indexes = grouped.indices

    train_indexes = []
    test_indexes = []
    val_indexes = []

    for group, index_list in group_indexes.items():
        # print(f"index_list: {len(index_list)}")
        index_test = []
        index_val = []
        if len(index_list) > 1:
            index_train, index_test, _, _ = train_test_split(index_list, index_list, test_size=0.2, random_state=42)
            train_indexes += list(index_train)
        elif len(index_list) == 1:
            train_indexes += list(index_list)

        # print(f"index_train: {len(index_train)} index_test: {len(index_test)}")

        # todo - next block of code should be fixed once we figure out validation issues
        if len(index_test) > 1 and False:
            index_test, index_val, _, _ = train_test_split(index_test, index_test, test_size=0.5, random_state=42)
            test_indexes += list(index_test)
            val_indexes += list(index_val)
        elif len(index_test) == 1:
            test_indexes += list(index_test)
        test_indexes += list(index_test)
        val_indexes += list(index_test)

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

    # TODO - need to add more training data for the classes that are not balanced
    conditions = isic_df["label"].unique().tolist()

    counts = train["label"].value_counts()

    print("Before augmentation")

    print(train["label"].value_counts(normalize=True, sort=False).mul(100).round(2))
    ros = RandomOverSampler()
    train, _ = ros.fit_resample(train, train["label"])
    pd.options.display.max_columns = None

    print(train.head())

    # max = counts.max()
    # for c in conditions:
    #    value = counts[c]
    #    ratio = int(max / value)
    #    print(c, value, ratio)
        # This is to oversampled, but this approach didnt seem to work well
        # if False:
        # train = train.append([train.loc[train['label'] == c, :]] * (ratio - 1),
        #                     ignore_index=True)
    print("After augmentation")
    print(train["label"].value_counts())
    print(train["label"].value_counts(normalize=True, sort=False).mul(100).round(2))

    """NV
    5361
    MEL
    888
    BKL
    876
    BCC
    410
    AKIEC
    260
    VASC
    113
    DF
    91
    """

    print(f"train size: {len(train)}")
    print(f"test size: {len(test)}")
    print(f"val size: {len(val)}")
    print(f"dataset sizes:{len(val) + len(test) + len(train)}. df size: {len(isic_df)}")

    conditions = isic_df["label"].unique().tolist()
    print("----")
    print_splits(all_domains, conditions, train)

    print("----")
    print_splits(all_domains, conditions, test)

    print("----")
    print_splits(all_domains, conditions, val)

    print("train size:", len(train))
    print("val size:", len(val))
    print("train skin types:", train.fizpatrick_skin_type.unique())
    print("val skin types:", val.fizpatrick_skin_type.unique())
    label_codes = sorted(list(train['label'].unique()))
    print("train skin conditions:", len(label_codes))
    label_codes1 = sorted(list(val['label'].unique()))
    print("val skin conditions:", len(label_codes1))

    # compute mean and standard deviation:
    # Next line can be uncommented if there is a new dataset. This really only needs to be done 1 time

    # normMean, normStd = compute_img_mean_std(isic_df, image_size)
    normMean = [0.76308113, 0.54567724, 0.57007957]
    normStd = [0.14093588, 0.15261903, 0.1699746]

    transformed_train = ISIC2018SkinDataset(
        df=train,
        transform=Compose([
            RandomRotation(degrees=15),
            RandomHorizontalFlip(),
            Resize(size=(image_size, image_size)),
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Normalize(normMean, normStd)
            # Now using computed mean and std above^ Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    transformed_val = ISIC2018SkinDataset(
        df=val,
        transform=Compose([
            Resize(size=(image_size, image_size)),
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Normalize(normMean, normStd)
            # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    transformed_test = ISIC2018SkinDataset(
        df=test,
        transform=Compose([
            Resize(size=(image_size, image_size)),
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Normalize(normMean, normStd)
            # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        transformed_train,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=2)

    val_loader = torch.utils.data.DataLoader(
        transformed_val,
        batch_size=batch_size,
        shuffle=False, drop_last=False, pin_memory=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        transformed_test,
        batch_size=batch_size,
        shuffle=False, drop_last=False, pin_memory=True, num_workers=2)

    return train_loader, val_loader, test_loader


def print_splits(all_domains, conditions, df_collection):
    print(df_collection['label'].value_counts())
    for s in all_domains:
        skin_type_len = len(df_collection[df_collection['fizpatrick_skin_type'] == s])
        print(f"\t skin type {s} : {skin_type_len} ({skin_type_len / len(df_collection) * 100:.2f})")
        for c in conditions:
            condition_len = len(df_collection[df_collection['label'] == c])
            print(f"\t\t{c}: {condition_len} ({condition_len / skin_type_len * 100:.2f})")
