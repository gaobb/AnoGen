# 固定随机数种子，从mvtec测试集的异常图像随机采样， 得到训练集，并将采样结果保存为list
import random
from PIL import Image
import os
import json
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', action='store', type=str, required=True, default='/apdcephfs/private_laurelgui/data/mvtec')
parser.add_argument('--k_shot', action='store', type=int, required=True, default=3)
parser.add_argument('--trainset_path', action='store', type=str, required=True, default='mvtec_train_data')
parser.add_argument('--save_dict', action='store', type=str, default=None)
parser.add_argument('--seed', action='store', type=int, default=0)
args = parser.parse_args()

random.seed(args.seed)
names = os.listdir(args.dataset_path)
# names = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 
#         'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
save_sicts = {}

for name in names:
    root_path = os.path.join(args.dataset_path, name)
    img_path = os.path.join(root_path, 'test')
    mask_path = os.path.join(root_path, 'ground_truth')

    # 遍历每一种缺陷
    categories = os.listdir(mask_path)
    for c in categories:
        print("creating--", name, "--", c)
        # 生成目标图像名称
        total_num = len(os.listdir(os.path.join(img_path, c)))
        numbers = random.sample(range(total_num), args.k_shot)
        targets = ["{:03d}.png".format(number) for number in numbers]
        mask_targets = ["{:03d}_mask.png".format(number) for number in numbers]
        # 遍历选中的3张图像
        for target in targets:
            # 存图像
            target_path = os.path.join(img_path, c, target)
            img = Image.open(target_path)
            save_img_path = os.path.join(args.trainset_path, name, c)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
            img.save(os.path.join(save_img_path, target))
        for target in mask_targets:
            # 存mask
            target_path = os.path.join(mask_path, c, target)
            img = Image.open(target_path)
            save_mask_path = os.path.join(args.trainset_path, name, c+'_mask')
            if not os.path.exists(save_mask_path):
                os.makedirs(save_mask_path)
            img.save(os.path.join(save_mask_path, target))
        # 存储当前类的选择结果
        save_sicts[name+"_"+c] = targets

if args.trainset_path != None:
    json_data = json.dumps(save_sicts)
    with open("train_list", 'w') as f:
        f.write(json_data)
    f.close()

        
