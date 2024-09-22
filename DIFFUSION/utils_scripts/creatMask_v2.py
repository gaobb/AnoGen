#根据目标生成mask
from PIL import Image, ImageDraw
import random
import os
import cv2
import numpy as np
import json

def random_mask(template, info):
    size = (256, 256)
    n = 0
    
    while True:
        img = np.zeros((size[1], size[0]), dtype=np.uint8)
        width = random.randint(int(info[0][0]*256), int(info[0][1]*256))
        height = random.randint(int(info[1][0]*256), int(info[1][1]*256))
        x = random.randint(0, 256-width)
        y = random.randint(0, 256-height)

        cv2.rectangle(img, (x, y), (x+width, y+height), (255, 255, 255), -1)
        init_size = float(np.count_nonzero(img))

        if info[2]:
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.minAreaRect(contours[0])
            M = cv2.getRotationMatrix2D(rect[0], random.randint(0, 180), 1) 
            img = cv2.warpAffine(img, M, img.shape[::-1])
        
        common_mask = img & template
        common_size = np.count_nonzero(common_mask) / init_size
        if common_size > 0.75:
            break
        else:
            n += 1
            if n%10000==0:
                print(n,x,y, common_size, np.count_nonzero(common_mask), float(np.count_nonzero(img)))
                print(width, height, img.mean())
    
    # print('Hit {:3d} times to sucess'.format(n))
    return img, n



if __name__ == '__main__':
    json_path="./utils_scripts/visa_range.json"
    data_path = "/apdcephfs/private_laurelgui/data/visa_pytorch"

    with open(json_path, "r") as file:
        mask_info = json.load(file)

    classes = os.listdir(data_path)
    #遍历每一个类别
    for c in classes:
        root_path = os.path.join(data_path, c, "train/good")
        number = len(os.listdir(root_path))
        #遍历每一张图像
        for i in range(1000):
            path = os.path.join(root_path, "{:04d}.JPG".format(i))
            mask_path = os.path.join("visa_masks", c, "{:03d}".format(i))
            if not os.path.exists(mask_path):
                os.makedirs(mask_path)
        
            print(path)
            img = cv2.imread(path)
            img = cv2.resize(img, (256, 256))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if thresh[0][0] == 255:
                thresh = cv2.bitwise_not(thresh)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            for j in range(2):
                mask, n = random_mask(closing, [[0.05, 0.1], [0.05, 0.1], False])
                print("generating {:03d}-{}: for {:03d} times".format(i, j, n))
                save_path = os.path.join(mask_path, "{:03d}.png".format(j))
                cv2.imwrite(save_path, mask)
