import os
from enum import Enum

import numpy as np
import pandas as pd
import PIL
import torch
from torchvision import transforms

_CLASSNAMES = []

# TODO
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class HarborDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Harbor.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        # img_width: int = 384
        # img_height: int = 288
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        # for classname in self.classnames_to_use:
        #     classpath = os.path.join(self.source, classname, self.split.value)
        #     maskpath = os.path.join(self.source, classname, "ground_truth")
        #     anomaly_types = os.listdir(classpath)

        #     imgpaths_per_class[classname] = {}
        #     maskpaths_per_class[classname] = {}

        #     for anomaly in anomaly_types:
        #         anomaly_path = os.path.join(classpath, anomaly)
        #         anomaly_files = sorted(os.listdir(anomaly_path))
        #         imgpaths_per_class[classname][anomaly] = [
        #             os.path.join(anomaly_path, x) for x in anomaly_files
        #         ]

        #         if self.train_val_split < 1.0:
        #             n_images = len(imgpaths_per_class[classname][anomaly])
        #             train_val_split_idx = int(n_images * self.train_val_split)
        #             if self.split == DatasetSplit.TRAIN:
        #                 imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
        #                     classname
        #                 ][anomaly][:train_val_split_idx]
        #             elif self.split == DatasetSplit.VAL:
        #                 imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
        #                     classname
        #                 ][anomaly][train_val_split_idx:]

        #         if self.split == DatasetSplit.TEST and anomaly != "good":
        #             anomaly_mask_path = os.path.join(maskpath, anomaly)
        #             anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
        #             maskpaths_per_class[classname][anomaly] = [
        #                 os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
        #             ]
        #         else:
        #             maskpaths_per_class[classname]["good"] = None

        for classname in self.classnames_to_use:
            data_path = r'/home/jacob/data/LTD Dataset/Image Dataset'

            csv_path = ''
            if self.split == DatasetSplit.TRAIN:
                csv_path = r'/home/jacob/code/harbor-synthetic/src/data/split/harbor_train_0001.csv'
            elif self.split == DatasetSplit.TEST:
                # csv_path = r'/home/jacob/code/harbor-synthetic/src/data/split/harbor_appearance_test.csv'
                csv_path = r'/home/jacob/code/harbor-synthetic/src/data/split/harbor_appearance_test_1.csv'
            else:
                raise ValueError()
            df = pd.read_csv(csv_path)

            imgpaths_per_class[classname] = {}
            # classpath = os.path.join(self.source, classname, self.split.value)
            # anomaly_types = os.listdir(classpath)
            # anomaly_types = ['anomaly_type']
            # TODO only good types
            anomaly_types = ['harbor_anomaly1']
            for anomaly in anomaly_types:
                df.img_path = data_path + os.sep + df.img_path

                maskpaths_per_class[classname] = {}
                if self.split == DatasetSplit.TRAIN:
                    imgpaths_per_class[classname][anomaly] = df.img_path.tolist()
                    maskpaths_per_class[classname]["good"] = None
                elif self.split == DatasetSplit.TEST:
                    gt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,]
                    imgpaths_per_class[classname]['good'] = df.img_path[~np.array(gt, dtype=bool)].to_list()
                    imgpaths_per_class[classname][anomaly] = df.img_path[np.array(gt, dtype=bool)].to_list()

                    maskpaths_per_class[classname]['good'] = imgpaths_per_class[classname]['good']
                    maskpaths_per_class[classname][anomaly] = imgpaths_per_class[classname][anomaly]

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
