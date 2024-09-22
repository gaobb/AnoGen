# 根据模生成mask
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
    json_path="./utils_scripts/mvtec_range.json"
    template_path = "./images/mask_template"

    with open(json_path, "r") as file:
        mask_info = json.load(file)
    
    for name in mask_info.keys():
        for catgory in mask_info[name].keys():
            template = cv2.imread(os.path.join(template_path, name, catgory,'mask.png'))
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            template = cv2.resize(template, (256, 256))
            
            for i in range(100):
                mask, n = random_mask(template, mask_info[name][catgory])
                save_path = os.path.join('mvtec_masks', name, catgory)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_path+("/{:03d}.png".format(i+1)), mask)
                print("Generating {}-{}-{:3d}:cost {} times".format(name, catgory, i, n))


