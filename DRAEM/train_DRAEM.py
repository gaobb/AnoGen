import torch
from data_loader import MVTecDRAEMTrainDataset,MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM, DiceLoss
import os
import numpy as np
import random
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import pyplot as plt

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

@torch.no_grad()
def update_ema(ema_model, model, alpha=0.99):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
 

@torch.no_grad()
def test(obj_name, dataset, dataloader, model, model_seg):
    if True:
        img_dim = 256

        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []


        for i_batch, sample_batched in enumerate(dataloader):

            gray_batch = sample_batched["image"].cuda()

            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            gray_rec = model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)


            out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()

            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)

            anomaly_score_prediction.append(image_score)

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)

        return auroc, ap, auroc_pixel, ap_pixel


def train_on_device(obj_names, args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    for obj_name in obj_names:
        run_name = 'DRAEM_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+obj_name+'_'

        visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.cuda()
        model.apply(weights_init)


        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.cuda()
        model_seg.apply(weights_init)


        optimizer = torch.optim.Adam([
                                      {"params": model.parameters(), "lr": args.lr},
                                      {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[400,600],gamma=0.1, last_epoch=-1)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_seg = DiceLoss() if args.loss_seg == 'dice' else FocalLoss()
        # loss_focal = FocalLoss()
        # loss_dice = dice_loss()

        dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, resize_shape=[256, 256], obj_name = obj_name, anomaly_source_path_DM=args.anomaly_source_path_DM)


        dataloader = DataLoader(dataset, batch_size=args.bs,
                                shuffle=True, num_workers=4)
        
    

        n_iter = 0
        

        for epoch in tqdm(range(args.epochs)):
            model.train()
            model_seg.train()
            for i_batch, sample_batched in enumerate(dataloader):

                gray_batch = sample_batched["image"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()
                anomaly_mask = sample_batched["anomaly_mask"].cuda()
                dtd = sample_batched["dtd"].cuda()
                # dtd = torch.BoolTensor(dtd)

                gray_rec = model(aug_gray_batch)
                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1) #[16,2,256,256]
                # out_mask_sm = out_mask.sigmoid()

                l2_loss = loss_l2(gray_rec,gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)

                # 前景概率
                max_p = out_mask_sm[:,1,:,:].detach()
                # 得到阈值
                # 前景置信度大于阈值的mask
                if args.warmup < 0:
                    args.p_cutoff = min(float(epoch)/(700*5), 0.2)
                    loss_mask = (max_p > args.p_cutoff)
                    
                else:
                    if epoch >= args.warmup:
                        loss_mask = (max_p > args.p_cutoff)
                        loss_mask_one = (max_p > 0.0)
                        loss_mask = loss_mask.unsqueeze(1)
                        loss_mask_one = loss_mask_one.unsqueeze(1)
                    else:
                        loss_mask = (max_p > 0.0)
                # 和真实mask运算得到最终mask
                reverse_anomaly_mask = 1.- anomaly_mask
                loss_mask = torch.logical_or(loss_mask, reverse_anomaly_mask).float()
                loss_mask_one = torch.logical_or(loss_mask_one, reverse_anomaly_mask).float()

                
                segment_loss = loss_seg(out_mask_sm, anomaly_mask) * loss_mask
                segment_loss = segment_loss.mean()  
                # print(segment_loss)
                
                loss = l2_loss + ssim_loss + segment_loss
                
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                if args.visualize and n_iter % 200 == 0:
                    visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                    visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                    visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')
                if args.visualize and n_iter % 400 == 0:
                    t_mask = out_mask_sm[:, 1:, :, :]
                    visualizer.visualize_image_batch(aug_gray_batch, n_iter, image_name='batch_augmented')
                    visualizer.visualize_image_batch(gray_batch, n_iter, image_name='batch_recon_target')
                    visualizer.visualize_image_batch(gray_rec, n_iter, image_name='batch_recon_out')
                    visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name='mask_target')
                    visualizer.visualize_image_batch(t_mask, n_iter, image_name='mask_out')


                n_iter +=1

            # if epoch % 1 == 0:
            #     print("epoch: ", epoch)
            # print(loss)

            scheduler.step()
            if (epoch+1)%100==0 and epoch >= 698:
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, str(epoch)+"_"+run_name+".pckl"))
                torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, str(epoch)+"_"+run_name+"_seg.pckl"))


if __name__=="__main__":
    import argparse
    # seed = 42

    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path_DM', action='store',type=str,required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--p_cutoff', action='store', type=float, default=0.5)
    parser.add_argument('--loss_seg', action='store', type=str, default='focal')
    parser.add_argument('--warmup', action='store', type=int, default=0)
    parser.add_argument('--seed', action='store', type=int, default=0)


    
    args = parser.parse_args()


    torch.manual_seed(args.seed) # 为CPU设置随机种子
    os.environ["PYTHONSEED"] = str(args.seed)
    torch.cuda.manual_seed(args.seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    obj_batch = [['capsule'],   # 0
                 ['bottle'],    # 1
                 ['carpet'],    # 2
                 ['leather'],   # 3
                 ['pill'],      # 4
                 ['transistor'],# 5
                 ['tile'],      # 6
                 ['cable'],     # 7
                 ['zipper'],    # 8
                 ['toothbrush'],# 9
                 ['metal_nut'], # 10
                 ['screw'],     # 11
                 ['grid'],      # 12
                 ['wood'],       # 13
                 ['hazelnut'], 
                 ]

    if int(args.obj_id) == -1:
        obj_list = ['capsule',
                     'bottle',
                     'carpet',
                     'leather',
                     'pill',
                     'transistor',
                     'tile',
                     'cable',
                     'zipper',
                     'toothbrush',
                     'metal_nut',
                     'hazelnut',
                     'screw',
                     'grid',
                     'wood'
                     ]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)

