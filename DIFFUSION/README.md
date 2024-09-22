# Few-Shot Anomaly-Driven Generation

---

## 预先准备工作

docker环境
```
mirrors.tencent.com/python3.8_torch1.13/textual:latest
```

diffusion model下载
```sh
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

训练的embedding向量
```
[mvtec] ./checkpoints/textual_models/AD-diifusion-v1
[visa]
```

生成的图像路径
```
[mvtec] ./checkpoints/data/anomaly_mvtec
[visa] ./checkpoints/data/anomaly_visa
```

图像生成所用的全部的mask(可选)
```
[mvtec] ./checkpoints/diffusion_results/mask_mvtec
[visa] 
```
---

## 图像生成
我们使用缺陷对应的向量引导扩散模型生成图像。为了增加可控性，我们在正常图像上的指定位置(box区域)生成缺陷。\
我们已经提供了训练好的向量，在logs/中，您可以直接使用这些向量生成图像. \
示例：给定单张正常图像和bounding box，生成异常图像 （```imgaes/demo_images/```提供了示例图像）
```python
python scripts/txt2img.py \
            --ddim_eta 0.0 \
            --n_samples 1 \
            --n_iter 2 \
            --scale 10.0 \
            --ddim_steps 50 \
            --embedding_path "logs/bottle_broken_large/embeddings.pt" \
            --ckpt_path "models/ldm/text2img-large/model.ckpt" \
            --prompt "*" \
            --mask_prompt "images/demo_images/mask.png" \
            --image_prompt "images/demo_images/bottle.png" \
            --outdir "outputs"
```

参数说明
```
--embedding_path 向量路径 /*更改此向量，可以生成不同的缺陷*/
--ckpt_path DM模型参数
--mask_prompt box区域的mask
--image_prompt 正常图像
--outdir 生成图像存储位置
```

也可以一次性生成多张异常图像.\
指定正常图像文件夹（例如：mvtec的训练集），以及mask的集合，遍历每一张正常图像，从mask集合中随机采样两个mask作为指定的缺陷生成区域。我们提供了.sh脚本供参考。
```sh
sh generate.sh
```
完整图像的路径
```
./checkpoints/data/anomaly_mvtec
```

---

## 学习向量
给定支持集```images/train_images```，即可训练每一种缺陷对应的向量
```sh
sh train.sh
```
参数说明
```
name 向量的路径+名称
path 支持集的路径+名称
word 当前的缺陷类型（可忽略）
--base 模型配置文件（建议默认）
-t --actual_resume 预训练的diffusion model模型位置
-n 向量路径
--gpus 暂不支持多卡
--data_root 支持集路径
--init_word 向量初始化词汇
```
---

## 完整的过程 && 额外的辅助脚本
您可以从零开始，在自己的数据集上生成异常图像，这将包含以下步骤：
- 创建支持集
- 定义box的大致区域、形状范围，并生成mask
- 训练向量
- 图像生成
- 按指定文件夹格式整理生成图像

注意：完整的步骤包含对数据集的处理，我们提供的额外脚本仅供参考(MVTec数据集格式)
1. 创建支持集。从每一种缺陷的真实异常图像中随机采样，构造支持集
```sh
sh make_trainset.sh
```
&emsp; 参数说明
```
--dataset_path 数据集路径
--k_shot 支持集中，每一类的真实图像的数量(论文中，我们设置的是3)
--trainset_path 创建后的支持集的路径
--save_dict 采样的序号列表存储位置
--seed 
```

2. 生成mask集合 
- 方式一：预定于位置模板(非必需). 当前缺陷位于目标的特殊部位（如瓶子的边缘），通过定义模板，确保box区域一定落在指定部位，而不是背景。我们提供了MVTec中每一类缺陷的模板```images/mvtec_template/```，您可以根据自己的需求对其进行更改。
- 方式二：根据目标生成mask. 对正常图像进行前景分割（如图像中的钉子），使缺陷生成区域在目标上，而不是背景。
```
注意：两种不同的mask生成逻辑，在后续的脚本文件处理中不同，可以参考提供文件中“bottle”类和"screw"类的区别
```
- 预定于box长宽的上下界（非必需）。一些缺陷（如划痕）具有特定的长宽比例和大小范围，通过简单定义box的长宽的上下界，可以使生成的缺陷更逼真。我们提供了MVTec中每一类缺陷的长宽上下界```utils_scripts/mvtec_range.json```，仅供参考，您可以根据自己的情况修改。\
json文件示例
```json
"bottle": { 
    "broken_large": [[0.20, 0.25], [0.25, 0.8], "True"],
    "broken_small": [[0.15, 0.25], [0.15, 0.25], "False"],
    "contamination": [[0.2, 0.4], [0.2, 0.4], "True"]
}
//对于缺陷类型bottle-contamination，我们定义长的上下界为[0.2,0.4]，宽的上下界为[0.2,0.4]，True代表此mask可以旋转
```
根据模板和上下界，执行以下脚本文件，即可生成缺陷的box mask. 方式一：
```
python utils_scripts/creatMask_v1.py
```
方式二：
```
python utils_scripts/creatMask_v2.py
```

3. 学习向量

4. 图像生成


5. 异常数据集制作。为了使生成的图像可以更加灵活地应用于下游任务，可以将生成图像和对应的图像制作为固定格式的数据集，我们提供了参考脚本
```
python utils_scripts/makeDataset.py
```
生成的数据将按照如下格式整理为新的数据集
```
--{category}
    --images
        {id-image}-{id-mask}-{id-sample}-{defect}.png
    --masks
        {id-mask}-{id-sample}.png
```
例
```
--bottle
    --images
        087-091-0001-broken_large.png
        057-015-0000-contamination.png
        ......
    --masks
        091-broken_large.png
        051-015-contamination.png
        ......
```

