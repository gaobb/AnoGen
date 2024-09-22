import glob
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

import random
import cv2


from data.data_utils import perlin_noise


class MVTecTestDataset(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.is_train = is_train
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        if self.is_train:
            dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
            dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
            dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

            fill_color = (114, 114, 114)
            # rotate_90
            if self.rotate_90:
                degree = np.random.choice(np.array([0, 90, 180, 270]))
                image = image.rotate(
                    degree, fillcolor=fill_color, resample=Image.BILINEAR
                )
            # random_rotate
            if self.random_rotate > 0:
                degree = np.random.uniform(-self.random_rotate, self.random_rotate)
                image = image.rotate(
                    degree, fillcolor=fill_color, resample=Image.BILINEAR
                )

            # perlin_noise implementation
            aug_image, aug_mask = perlin_noise(image, dtd_image, aug_prob=1.0)
            aug_image = self.final_preprocessing(aug_image)

            image = self.final_preprocessing(image)
            return {"img_aug": aug_image, "img_origin": image, "mask": aug_mask}
        else:
            image = self.final_preprocessing(image)
            dir_path, file_name = os.path.split(self.mvtec_paths[index])
            base_dir = os.path.basename(dir_path)
            if base_dir == "good":
                mask = torch.zeros_like(image[:1])
            else:
                mask_path = os.path.join(dir_path, "../../ground_truth/")
                mask_path = os.path.join(mask_path, base_dir)
                mask_file_name = file_name.split(".")[0] + "_mask.png"
                mask_path = os.path.join(mask_path, mask_file_name)
                mask = Image.open(mask_path)
                mask = self.mask_preprocessing(mask)
                mask = torch.where(
                    mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
                )
            return {"img": image, "mask": mask}


class MVTecDataset(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.is_train = is_train
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png"))
            self.mask_preprocessing = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(
                        size=(self.resize_shape[1], self.resize_shape[0]),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        if self.is_train:
            dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
            dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
            dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

            fill_color = (114, 114, 114)
            # rotate_90
            if self.rotate_90:
                degree = np.random.choice(np.array([0, 90, 180, 270]))
                image = image.rotate(
                    degree, fillcolor=fill_color, resample=Image.BILINEAR
                )
            # random_rotate
            if self.random_rotate > 0:
                degree = np.random.uniform(-self.random_rotate, self.random_rotate)
                image = image.rotate(
                    degree, fillcolor=fill_color, resample=Image.BILINEAR
                )

            # perlin_noise implementation
            aug_image, aug_mask = perlin_noise(image, dtd_image, aug_prob=1.0)
            aug_image = self.final_preprocessing(aug_image)
            image = self.final_preprocessing(image)
            print(aug_mask)
            print(aug_mask.min(), aug_mask.max())
            return {"img_aug": aug_image, "img_origin": image, "mask": aug_mask}
        else:
            image = self.final_preprocessing(image)
            dir_path, file_name = os.path.split(self.mvtec_paths[index])
            base_dir = os.path.basename(dir_path)
            if base_dir == "good":
                mask = torch.zeros_like(image[:1])
            else:
                mask_path = os.path.join(dir_path, "../../ground_truth/")
                mask_path = os.path.join(mask_path, base_dir)
                mask_file_name = file_name.split(".")[0] + "_mask.png"
                mask_path = os.path.join(mask_path, mask_file_name)
                mask = Image.open(mask_path)
                mask = self.mask_preprocessing(mask)
                mask = torch.where(
                    mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
                )
            return {"img": image, "mask": mask}


class MVTecDataset_new(Dataset):
    def __init__(
        self,
        is_train,
        mvtec_dir,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        obj_name = None,
        anomaly_source_path_DM = None,
        sample_pro = 0.5
    ):
        super().__init__()
        self.resize_shape = resize_shape
        self.is_train = is_train

        self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png"))
        self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
        self.rotate_90 = rotate_90
        self.random_rotate = random_rotate

        # 自己的dm
        self.obj_name=obj_name
        self.anomaly_source_path = anomaly_source_path_DM
        self.anomaly_source_paths = os.listdir(self.anomaly_source_path+obj_name+"/images")
        self.anomaly_source_masks = os.listdir(self.anomaly_source_path+obj_name+"/masks") 
        self.sample_pro = sample_pro

        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):
        return len(self.mvtec_paths)

    def __getitem__(self, index):
        if torch.rand(1).numpy()[0] > self.sample_pro:
            index = torch.randint(0, len(self.mvtec_paths), (1,)).item()
            image = Image.open(self.mvtec_paths[index]).convert("RGB")
            image = image.resize(self.resize_shape, Image.BILINEAR)

            dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
            dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
            dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)

            fill_color = (114, 114, 114)
            # rotate_90
            if self.rotate_90:
                degree = np.random.choice(np.array([0, 90, 180, 270]))
                image = image.rotate(
                    degree, fillcolor=fill_color, resample=Image.BILINEAR
                )
            # random_rotate
            if self.random_rotate > 0:
                degree = np.random.uniform(-self.random_rotate, self.random_rotate)
                image = image.rotate(
                    degree, fillcolor=fill_color, resample=Image.BILINEAR
                )

            # perlin_noise implementation
            aug_image, aug_mask = perlin_noise(image, dtd_image, aug_prob=1.0)
            aug_image = self.final_preprocessing(aug_image)

            image = self.final_preprocessing(image)
            return {"img_aug": aug_image, "img_origin": image, "mask": aug_mask, "dtd": True}

        else:
            selected = random.sample(self.anomaly_source_paths, 1)[0]
            idx = int(selected[:3])
            image = Image.open(self.mvtec_paths[idx]).convert("RGB")
            image = image.resize(self.resize_shape, Image.BILINEAR)
            # image = cv2.imread(self.image_paths[idx])
            # image = cv2.resize(image, (256, 256))
            mask_idx = None

            if self.obj_name == 'screw':
                mask_idx = selected[:7]+".png"
            else:
                mask_idx = selected[4:7]+"-"+selected[13:]
            

            augmented_image = Image.open(self.anomaly_source_path + self.obj_name+"/images/"+selected).convert("RGB")
            augmented_mask = Image.open(self.anomaly_source_path + self.obj_name+"/masks/"+mask_idx).convert('L')
            augmented_image = augmented_image.resize(self.resize_shape, Image.BILINEAR)
            augmented_mask = augmented_mask.resize(self.resize_shape, Image.BILINEAR)

            fill_color = (114, 114, 114)
            if self.rotate_90:
                degree = np.random.choice(np.array([0, 90, 180, 270]))
                image = image.rotate(degree, fillcolor=fill_color, resample=Image.BILINEAR)
                augmented_image = augmented_image.rotate(degree, fillcolor=fill_color, resample=Image.BILINEAR)
                augmented_mask = augmented_mask.rotate(degree, fillcolor=0, resample=Image.BILINEAR)
            if self.random_rotate > 0:
                degree = np.random.uniform(-self.random_rotate, self.random_rotate)
                image = image.rotate(degree, fillcolor=fill_color, resample=Image.BILINEAR)
                augmented_image = augmented_image.rotate(degree, fillcolor=fill_color, resample=Image.BILINEAR)
                augmented_mask = augmented_mask.rotate(degree, fillcolor=0, resample=Image.BILINEAR)
            
            augmented_image= self.final_preprocessing(augmented_image)
            image = self.final_preprocessing(image)

            augmented_mask = np.array(augmented_mask)
            augmented_mask = augmented_mask.astype(np.float32)
            augmented_mask = augmented_mask / 255
            augmented_mask = np.expand_dims(augmented_mask, axis=0)


            sample = {'img_origin': image, "mask": augmented_mask, 
                    'img_aug': augmented_image, 'dtd':False}

            return sample