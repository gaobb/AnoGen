{ 
for ((i=0;i<1;i++))
do
    python train.py \
        --gpu_id 0 \
        --num_workers 8 \
        --obj_id 8 \
        --anomaly_source_path ../datasets/mvtec_anomaly/ \
        --checkpoint_path ../saved_destseg_models_test_4 \
        --mode 2 \
        --p_cutoff 0.0
done
}