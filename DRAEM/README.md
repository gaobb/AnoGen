# Few-Shot Anomaly-Driven Generation

---

## 预先准备工作

docker环境
```
mirrors.tencent.com/anomaly_python3.8/anomalib_v4:anomalib_v4
```

DTD数据集下载
```sh
sh scripts/download_dataset.sh
```

模型地址
- MVTec
```
./checkpoints/draem_results/checkpoints_draem_dm_lr_0.2_tau_0.1_prob_0.5
```
- Visa
```
./checkpoints/draem_results/checkpoints_visa
```
---

## 训练模型
```
sh ./train.sh
```
参数说明
```sh
--obj_id 训练的类别序号，为-1时，训练所有类别
--data_path MVTec路径
--anomaly_source_path DTD数据路径
--anomaly_source_path_DM 生成的异常数据路径
--checkpoint_path 模型保存路径
--p_cutoff 阈值t
```

## 验证模型
```
sh ./test.sh
```
参数说明
```sh
--checkpoint_path 模型路径
--obj_id 要验证的类别序号，为-1时，验证所有类别
```