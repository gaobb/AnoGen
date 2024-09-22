import argparse
import os
import shutil
import warnings

import torch
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset, MVTecDataset_new
from eval import evaluate
from model.destseg import DeSTSeg
from model.losses import cosine_similarity_loss, focal_loss, l1_loss
import numpy as np
import random

warnings.filterwarnings("ignore")


def train(args, category, rotate_90=False, random_rotate=0):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"{args.run_name_head}_{args.steps}_{category}"
    if os.path.exists(os.path.join(args.log_path, run_name + "/")):
        shutil.rmtree(os.path.join(args.log_path, run_name + "/"))

    # visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

    model = DeSTSeg(dest=True, ed=True).cuda()

    seg_optimizer = torch.optim.SGD(
        [
            {"params": model.segmentation_net.res.parameters(), "lr": args.lr_res},
            {"params": model.segmentation_net.head.parameters(), "lr": args.lr_seghead},
        ],
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )
    de_st_optimizer = torch.optim.SGD(
        [
            {"params": model.student_net.parameters(), "lr": args.lr_de_st},
        ],
        lr=0.4,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

    sample_pro = 0.5
    if args.mode == 1:  # mode为1，则不适用dtd
        sample_pro = 1

    dataset = MVTecDataset_new(
        is_train=True,
        mvtec_dir=args.mvtec_path + category + "/train/good/",
        resize_shape=RESIZE_SHAPE,
        normalize_mean=NORMALIZE_MEAN,
        normalize_std=NORMALIZE_STD,
        dtd_dir=args.dtd_path,
        rotate_90=rotate_90,
        random_rotate=random_rotate,
        anomaly_source_path_DM=args.anomaly_source_path,
        obj_name=category,
        sample_pro=sample_pro
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    global_step = 0

    flag = True

    while flag:
        for _, sample_batched in enumerate(dataloader):
            seg_optimizer.zero_grad()
            de_st_optimizer.zero_grad()
            img_origin = sample_batched["img_origin"].cuda()
            img_aug = sample_batched["img_aug"].cuda()
            mask = sample_batched["mask"].cuda()
            dtd = sample_batched["dtd"].cuda()

            if global_step < args.de_st_steps:
                model.student_net.train()
                model.segmentation_net.eval()
            else:
                model.student_net.eval()
                model.segmentation_net.train()

            output_segmentation, output_de_st, output_de_st_list = model(
                img_aug, img_origin
            )

            mask = F.interpolate(
                mask,
                size=output_segmentation.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.where(
                mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
            )
            # import ipdb
            # ipdb.set_trace()
            # 弱监督mask
            # print(output_segmentation)
            max_p = output_segmentation.detach()
            loss_mask = (max_p > args.p_cutoff)
            reverse_anomaly_mask = 1.- mask
            loss_mask = torch.logical_or(loss_mask, reverse_anomaly_mask).float()
            # dtd决定mask
            # dtd = dtd.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # fina_mask = torch.where(dtd, mask, loss_mask)

            cosine_loss_val = cosine_similarity_loss(output_de_st_list)
            focal_loss_val = focal_loss(output_segmentation, mask, gamma=args.gamma)
            l1_loss_val = l1_loss(output_segmentation, mask)
            
            # import ipdb
            # ipdb.set_trace()

            if global_step < args.de_st_steps:
                total_loss_val = cosine_loss_val
                total_loss_val.backward()
                de_st_optimizer.step()
            else:
                focal_loss_val = focal_loss_val * loss_mask
                l1_loss_val = l1_loss_val

                total_loss_val = focal_loss_val.mean() + l1_loss_val.mean()

                total_loss_val.backward()
                seg_optimizer.step()

            global_step += 1

            # visualizer.add_scalar("cosine_loss", cosine_loss_val, global_step)
            # visualizer.add_scalar("focal_loss", focal_loss_val, global_step)
            # visualizer.add_scalar("l1_loss", l1_loss_val, global_step)
            # visualizer.add_scalar("total_loss", total_loss_val, global_step)

            # if global_step % args.eval_per_steps == 0:
            #     evaluate(args, category, model, visualizer, global_step)

            if global_step % args.log_per_steps == 0:
                if global_step < args.de_st_steps:
                    print(
                        f"Training {obj} at global step {global_step}"
                    )
                else:
                    print(
                        f"Training {obj} at global step {global_step}"
                    )

            if global_step >= args.steps:
                flag = False
                break

    torch.save(
        model.state_dict(), os.path.join(args.checkpoint_path, run_name + ".pckl")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--mvtec_path", type=str, default="/apdcephfs/private_laurelgui/data/mvtec/")
    parser.add_argument("--dtd_path", type=str, default="./datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model/")
    parser.add_argument("--run_name_head", type=str, default="DeSTSeg_MVTec")
    parser.add_argument("--log_path", type=str, default="./logs/")

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr_de_st", type=float, default=0.4)
    parser.add_argument("--lr_res", type=float, default=0.1)
    parser.add_argument("--lr_seghead", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--de_st_steps", type=int, default=1000
    )  # steps of training the denoising student model
    parser.add_argument("--eval_per_steps", type=int, default=1000)
    parser.add_argument("--log_per_steps", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=4)  # for focal loss
    parser.add_argument("--T", type=int, default=100)  # for image-level inference

    parser.add_argument(
        "--custom_training_category", action="store_true", default=False
    )
    parser.add_argument("--no_rotation_category", nargs="*", type=str, default=list())
    parser.add_argument(
        "--slight_rotation_category", nargs="*", type=str, default=list()
    )
    parser.add_argument("--rotation_category", nargs="*", type=str, default=list())
    parser.add_argument("--obj_id", type=int, default=-1, required=True)

    parser.add_argument("--anomaly_source_path", action='store', required=True, type=str)
    parser.add_argument("--mode", type=int, required=True) 
    parser.add_argument("--p_cutoff", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed) # 为CPU设置随机种子
    os.environ["PYTHONSEED"] = str(args.seed)
    torch.cuda.manual_seed(args.seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.

    if args.custom_training_category:
        no_rotation_category = args.no_rotation_category
        slight_rotation_category = args.slight_rotation_category
        rotation_category = args.rotation_category
        # check
        for category in (
            no_rotation_category + slight_rotation_category + rotation_category
        ):
            assert category in ALL_CATEGORY
    else:
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

    # with torch.cuda.device(args.gpu_id):
    #     for obj in no_rotation_category:
    #         print(obj)
    #         train(args, obj)

    #     for obj in slight_rotation_category:
    #         print(obj)
    #         train(args, obj, rotate_90=False, random_rotate=5)

    #     for obj in rotation_category:
    #         print(obj)
    #         train(args, obj, rotate_90=True, random_rotate=5)

    obj_batch = [['capsule'],   # 0
                ['bottle'],    # 1
                ['carpet'],    # 2
                ['leather'],   # 3
                ['pill'],      # 4
                ['transistor'],# 5
                ['tile'],      # 6
                ['cable'],     # 7
                ['zipper'],    # 8
                ['hazelnut'],# 9
                ['metal_nut'], # 10
                ['screw'],     # 11
                ['grid'],      # 12
                ['wood'],       # 13
                ['toothbrush'], 
                ]
    
    with torch.cuda.device(args.gpu_id):
        obj = obj_batch[args.obj_id][0]
        if obj in no_rotation_category:
            train(args, obj)
        elif obj in slight_rotation_category:
            train(args, obj, rotate_90=False, random_rotate=5)
        else:
            train(args, obj, rotate_90=True, random_rotate=5)