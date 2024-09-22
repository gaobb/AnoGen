# Few-Shot Anomaly-Driven Generation

---

## 预先准备工作 (同DRAEM)

docker环境
```
mirrors.tencent.com/anomaly_python3.8/anomalib_v4:anomalib_v4
```

DTD数据集下载
```sh
sh scripts/download_dataset.sh
```

模型地址
```
./checkpoints/destseg_results/saved_models_data_dm_dtd_p_0_1
```
---

## 训练模型
```
sh train.sh
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
sh test.sh
```
参数说明
```sh
--checkpoint_path 模型路径
--obj_id 要验证的类别序号，为-1时，验证所有类别
```