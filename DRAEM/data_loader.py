import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
import random
import time

no_rotation_category = [
    "capsule",
    "metal_nut",
    "pill",
    "toothbrush",
    "transistor",
]
slight_rotation_category = [
    "wood",
    "zipper",
    "cable",
]
rotation_category = [
    "bottle",
    "grid",
    "hazelnut",
    "leather",
    "tile",
    "carpet",
    "screw",
]


class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample


   
class MVTecDRAEMTrainDataset(Dataset):
    def __init__(self, root_dir, anomaly_source_path, resize_shape, obj_name, anomaly_source_path_DM):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.obj_name=obj_name
        self.anomaly_source_path = anomaly_source_path_DM

        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))
        self.anomaly_source_paths_initial = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))


        self.anomaly_source_paths = os.listdir(self.anomaly_source_path+obj_name+"/images")
        # self.anomaly_source_paths = random.sample(self.anomaly_source_paths, int(len(self.anomaly_source_paths)*number))
        self.anomaly_source_masks = os.listdir(self.anomaly_source_path+obj_name+"/masks")

        print(len(self.anomaly_source_paths))
        if obj_name == 'capsule': # ok
            for x in self.anomaly_source_paths:
                if 'poke' in x:
                    self.anomaly_source_paths.remove(x)
        if obj_name == 'cable': # ok
            for x in self.anomaly_source_paths:
                if 'cable_swap' in x:
                    self.anomaly_source_paths.remove(x)
        print(len(self.anomaly_source_paths))
            

        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))

        # self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug
    
    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)
        # if self.obj_name in slight_rotation_category:
        #     rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        #     image = rot(image=image)
        # elif self.obj_name in rotation_category:
        #     degree = np.random.choice(np.array([0, 90, 180, 270]))
        #     rot = iaa.Sequential([iaa.Affine(rotate=degree)])
        #     image = rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        if torch.rand(1).numpy()[0] > 0.5:
            idx = torch.randint(0, len(self.image_paths), (1,)).item()
            anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths_initial), (1,)).item()
            image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths_initial[anomaly_source_idx])


            sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx, 'dtd':True}
            
            return sample

        else:
            selected = random.sample(self.anomaly_source_paths, 1)[0]
            idx = int(selected[:3])
            image = cv2.imread(self.image_paths[idx])
            image = cv2.resize(image, (256, 256))
            mask_idx = None

            if self.obj_name == 'screw':
                mask_idx = selected[:7]+".png"
            else:
                mask_idx = selected[4:7]+"-"+selected[13:]
            
            # print(idx, selected, mask_idx, self.image_paths[idx])
            
            if torch.rand(1).numpy()[0] > 0.5:
                augmented_image = cv2.imread(self.anomaly_source_path+self.obj_name+"/images/"+selected)
                anomaly_mask = cv2.imread(self.anomaly_source_path+self.obj_name+"/masks/"+mask_idx)
                # anomaly_mask_grabcut = self.grabCut(augmented_image, anomaly_mask)
                has_anomaly = np.array([1.0],dtype=np.float32)
            else:
                anomaly_mask = np.zeros_like(image, dtype=np.float32)
                # anomaly_mask_grabcut = np.zeros_like(image, dtype=np.float32)
                augmented_image = image
                has_anomaly = np.array([0.0],dtype=np.float32)
            
            anomaly_mask = cv2.cvtColor(anomaly_mask, cv2.COLOR_BGR2GRAY)
            anomaly_mask = np.expand_dims(anomaly_mask, axis=0)
            # anomaly_mask_grabcut = cv2.cvtColor(anomaly_mask_grabcut, cv2.COLOR_BGR2GRAY)
            # anomaly_mask_grabcut = np.expand_dims(anomaly_mask_grabcut, axis=0)


            if torch.rand(1).numpy()[0] > 0.7:
                angle = random.randint(-90, 90)
                center = (128, 128)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
                augmented_image = cv2.warpAffine(augmented_image, rotation_matrix, (image.shape[1], image.shape[0]))
                anomaly_mask = cv2.warpAffine(anomaly_mask, rotation_matrix, (anomaly_mask.shape[1], anomaly_mask.shape[0]), flags=cv2.INTER_NEAREST)
                # anomaly_mask_grabcut = cv2.warpAffine(anomaly_mask_grabcut, rotation_matrix, (anomaly_mask.shape[1], anomaly_mask.shape[0]), flags=cv2.INTER_NEAREST)
            # if self.obj_name in slight_rotation_category:
            #     angle = random.randint(-90, 90)
            #     center = (128, 128)
            #     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            #     image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            #     augmented_image = cv2.warpAffine(augmented_image, rotation_matrix, (image.shape[1], image.shape[0]))
            #     anomaly_mask = cv2.warpAffine(anomaly_mask, rotation_matrix, (anomaly_mask.shape[1], anomaly_mask.shape[0]), flags=cv2.INTER_NEAREST)
            # elif self.obj_name in rotation_category:
            #     angle = np.random.choice(np.array([0, 90, 180, 270]))
            #     center = (128, 128)
            #     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            #     image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            #     augmented_image = cv2.warpAffine(augmented_image, rotation_matrix, (image.shape[1], image.shape[0]))
            #     anomaly_mask = cv2.warpAffine(anomaly_mask, rotation_matrix, (anomaly_mask.shape[1], anomaly_mask.shape[0]), flags=cv2.INTER_NEAREST)
                
            # _, anomaly_mask = cv2.threshold(anomaly_mask, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
            # _, anomaly_mask_grabcut = cv2.threshold(anomaly_mask_grabcut, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
            
            image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
            augmented_image = np.array(augmented_image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
            anomaly_mask = np.array(anomaly_mask).reshape((anomaly_mask.shape[0], anomaly_mask.shape[1], anomaly_mask.shape[2])).astype(np.float32) / 255.0
            # anomaly_mask_grabcut = np.array(anomaly_mask_grabcut).reshape((anomaly_mask.shape[0], anomaly_mask.shape[1], anomaly_mask.shape[2])).astype(np.float32) / 255.0
            
            augmented_image = np.transpose(augmented_image, (2, 0, 1))
            image = np.transpose(image, (2, 0, 1))

            sample = {'image': image, "anomaly_mask": anomaly_mask, 
                    'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx, 'dtd':False}

            return sample
        
