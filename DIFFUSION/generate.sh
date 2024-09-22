mvtec_root_path="../datasets/mvtec"
mvtec_mask_root_path="mvtec_masks"

# 1. bottle
categories=("broken_large" "broken_small" "contamination")
name="bottle"
img_root_path="${mvtec_root_path}/${name}/train/good"
for ((i=0;i<3;i++))
do
    embbeding_path="logs/${name}_${categories[$i]}/embeddings.pt"
    mask_root_path="${mvtec_mask_root_path}/${name}"
    find $img_root_path -type f | while read file; 
    do
        file=$(basename $file)
        img_path="${img_root_path}/${file}"
        mask_path="${mask_root_path}/${categories[i]}"
        for ((j=0;j<2;j++))
        do
            files=$(find "$mask_path" -type f | shuf)
            mask_file=$(echo "$files" | head -n 1)
            mask_file=$(basename $mask_file)
            mask_path_finally="${mask_path}/${mask_file}"
            out_path="outputs/${name}/${categories[$i]}/${file:0:3}-${mask_file:0:3}"
            python scripts/txt2img.py \
                --ddim_eta 0.0 \
                --n_samples 1 \
                --n_iter 2 \
                --scale 10.0 \
                --ddim_steps 50 \
                --embedding_path $embbeding_path \
                --ckpt_path models/ldm/text2img-large/model.ckpt \
                --prompt "*" \
                --mask_prompt $mask_path_finally \
                --image_prompt $img_path \
                --outdir $out_path
        done
    done
done

# 2. cable
categories=("bent_wire" "cable_swap" "combined" "cut_inner_insulation" "cut_outer_insulation" "missing_cable" "missing_wire" "poke_insulation")
name="cable"
img_root_path="${mvtec_root_path}/${name}/train/good"
for ((i=0;i<8;i++))
do
    embbeding_path="logs/${name}_${categories[$i]}/embeddings.pt"
    mask_root_path="${mvtec_mask_root_path}/${name}"
    find $img_root_path -type f | while read file; 
    do
        file=$(basename $file)
        img_path="${img_root_path}/${file}"
        mask_path="${mask_root_path}/${categories[i]}"
        for ((j=0;j<2;j++))
        do
            files=$(find "$mask_path" -type f | shuf)
            mask_file=$(echo "$files" | head -n 1)
            mask_file=$(basename $mask_file)
            mask_path_finally="${mask_path}/${mask_file}"
            out_path="outputs/${name}/${categories[$i]}/${file:0:3}-${mask_file:0:3}"
            python scripts/txt2img.py \
                --ddim_eta 0.0 \
                --n_samples 1 \
                --n_iter 2 \
                --scale 10.0 \
                --ddim_steps 50 \
                --embedding_path $embbeding_path \
                --ckpt_path models/ldm/text2img-large/model.ckpt \
                --prompt "*" \
                --mask_prompt $mask_path_finally \
                --image_prompt $img_path \
                --outdir $out_path
        done
    done
done

# 3. capsule
categories=("crack" "faulty_imprint" "poke" "scratch" "squeeze")
name="capsule"
img_root_path="${mvtec_root_path}/${name}/train/good"
for ((i=0;i<5;i++))
do
    embbeding_path="logs/${name}_${categories[$i]}/embeddings.pt"
    mask_root_path="${mvtec_mask_root_path}/${name}"
    find $img_root_path -type f | while read file; 
    do
        file=$(basename $file)
        img_path="${img_root_path}/${file}"
        mask_path="${mask_root_path}/${categories[i]}"
        for ((j=0;j<2;j++))
        do
            files=$(find "$mask_path" -type f | shuf)
            mask_file=$(echo "$files" | head -n 1)
            mask_file=$(basename $mask_file)
            mask_path_finally="${mask_path}/${mask_file}"
            out_path="outputs/${name}/${categories[$i]}/${file:0:3}-${mask_file:0:3}"
            python scripts/txt2img.py \
                --ddim_eta 0.0 \
                --n_samples 1 \
                --n_iter 2 \
                --scale 10.0 \
                --ddim_steps 50 \
                --embedding_path $embbeding_path \
                --ckpt_path models/ldm/text2img-large/model.ckpt \
                --prompt "*" \
                --mask_prompt $mask_path_finally \
                --image_prompt $img_path \
                --outdir $out_path
        done
    done
done

# 4. carpet
categories=("color" "cut" "hole" "metal_contamination" "thread")
name="carpet"
img_root_path="${mvtec_root_path}/${name}/train/good"
for ((i=0;i<5;i++))
do
    embbeding_path="logs/${name}_${categories[$i]}/embeddings.pt"
    mask_root_path="${mvtec_mask_root_path}/${name}"
    find $img_root_path -type f | while read file; 
    do
        file=$(basename $file)
        img_path="${img_root_path}/${file}"
        mask_path="${mask_root_path}/${categories[i]}"
        for ((j=0;j<2;j++))
        do
            files=$(find "$mask_path" -type f | shuf)
            mask_file=$(echo "$files" | head -n 1)
            mask_file=$(basename $mask_file)
            mask_path_finally="${mask_path}/${mask_file}"
            out_path="outputs/${name}/${categories[$i]}/${file:0:3}-${mask_file:0:3}"
            python scripts/txt2img.py \
                --ddim_eta 0.0 \
                --n_samples 1 \
                --n_iter 2 \
                --scale 10.0 \
                --ddim_steps 50 \
                --embedding_path $embbeding_path \
                --ckpt_path models/ldm/text2img-large/model.ckpt \
                --prompt "*" \
                --mask_prompt $mask_path_finally \
                --image_prompt $img_path \
                --outdir $out_path
        done
    done
done

# 5. grid
categories=("bent" "broken" "glue" "metal_contamination" "thread")
name="grid"
img_root_path="${mvtec_root_path}/${name}/train/good"
for ((i=0;i<5;i++))
do
    embbeding_path="logs/${name}_${categories[$i]}/embeddings.pt"
    mask_root_path="${mvtec_mask_root_path}/${name}"
    find $img_root_path -type f | while read file; 
    do
        file=$(basename $file)
        img_path="${img_root_path}/${file}"
        mask_path="${mask_root_path}/${categories[i]}"
        for ((j=0;j<2;j++))
        do
            files=$(find "$mask_path" -type f | shuf)
            mask_file=$(echo "$files" | head -n 1)
            mask_file=$(basename $mask_file)
            mask_path_finally="${mask_path}/${mask_file}"
            out_path="outputs/${name}/${categories[$i]}/${file:0:3}-${mask_file:0:3}"
            python scripts/txt2img.py \
                --ddim_eta 0.0 \
                --n_samples 1 \
                --n_iter 2 \
                --scale 10.0 \
                --ddim_steps 50 \
                --embedding_path $embbeding_path \
                --ckpt_path models/ldm/text2img-large/model.ckpt \
                --prompt "*" \
                --mask_prompt $mask_path_finally \
                --image_prompt $img_path \
                --outdir $out_path
        done
    done
done

# 6. hazelnut
categories=("crack" "cut" "hole" "print")
name="hazelnut"
img_root_path="${mvtec_root_path}/${name}/train/good"
for ((i=0;i<4;i++))
do
    embbeding_path="logs/${name}_${categories[$i]}/embeddings.pt"
    mask_root_path="${mvtec_mask_root_path}/${name}"
    find $img_root_path -type f | while read file; 
    do
        file=$(basename $file)
        img_path="${img_root_path}/${file}"
        mask_path="${mask_root_path}/${categories[i]}"
        for ((j=0;j<2;j++))
        do
            files=$(find "$mask_path" -type f | shuf)
            mask_file=$(echo "$files" | head -n 1)
            mask_file=$(basename $mask_file)
            mask_path_finally="${mask_path}/${mask_file}"
            out_path="outputs/${name}/${categories[$i]}/${file:0:3}-${mask_file:0:3}"
            python scripts/txt2img.py \
                --ddim_eta 0.0 \
                --n_samples 1 \
                --n_iter 2 \
                --scale 10.0 \
                --ddim_steps 50 \
                --embedding_path $embbeding_path \
                --ckpt_path models/ldm/text2img-large/model.ckpt \
                --prompt "*" \
                --mask_prompt $mask_path_finally \
                --image_prompt $img_path \
                --outdir $out_path
        done
    done
done

# 7. leather
categories=("color" "cut" "fold" "glue" "poke")
name="leather"
img_root_path="${mvtec_root_path}/${name}/train/good"
for ((i=0;i<5;i++))
do
    embbeding_path="logs/${name}_${categories[$i]}/embeddings.pt"
    mask_root_path="${mvtec_mask_root_path}/${name}"
    find $img_root_path -type f | while read file; 
    do
        file=$(basename $file)
        img_path="${img_root_path}/${file}"
        mask_path="${mask_root_path}/${categories[i]}"
        for ((j=0;j<2;j++))
        do
            files=$(find "$mask_path" -type f | shuf)
            mask_file=$(echo "$files" | head -n 1)
            mask_file=$(basename $mask_file)
            mask_path_finally="${mask_path}/${mask_file}"
            out_path="outputs/${name}/${categories[$i]}/${file:0:3}-${mask_file:0:3}"
            python scripts/txt2img.py \
                --ddim_eta 0.0 \
                --n_samples 1 \
                --n_iter 2 \
                --scale 10.0 \
                --ddim_steps 50 \
                --embedding_path $embbeding_path \
                --ckpt_path models/ldm/text2img-large/model.ckpt \
                --prompt "*" \
                --mask_prompt $mask_path_finally \
                --image_prompt $img_path \
                --outdir $out_path
        done
    done
done

# 8. metal_nut
categories=("bent" "color" "flip" "scratch")
name="metal_nut"
img_root_path="${mvtec_root_path}/${name}/train/good"
for ((i=0;i<4;i++))
do
    embbeding_path="logs/${name}_${categories[$i]}/embeddings.pt"
    mask_root_path="${mvtec_mask_root_path}/${name}"
    find $img_root_path -type f | while read file; 
    do
        file=$(basename $file)
        img_path="${img_root_path}/${file}"
        mask_path="${mask_root_path}/${categories[i]}"
        for ((j=0;j<2;j++))
        do
            files=$(find "$mask_path" -type f | shuf)
            mask_file=$(echo "$files" | head -n 1)
            mask_file=$(basename $mask_file)
            mask_path_finally="${mask_path}/${mask_file}"
            out_path="outputs/${name}/${categories[$i]}/${file:0:3}-${mask_file:0:3}"
            python scripts/txt2img.py \
                --ddim_eta 0.0 \
                --n_samples 1 \
                --n_iter 2 \
                --scale 10.0 \
                --ddim_steps 50 \
                --embedding_path $embbeding_path \
                --ckpt_path models/ldm/text2img-large/model.ckpt \
                --prompt "*" \
                --mask_prompt $mask_path_finally \
                --image_prompt $img_path \
                --outdir $out_path
        done
    done
done

# 9. pill
categories=("color" "combined" "contamination" "crack" "faulty_imprint" "pill_type" "scratch")
name="pill"
img_root_path="${mvtec_root_path}/${name}/train/good"
for ((i=0;i<7;i++))
do
    embbeding_path="logs/${name}_${categories[$i]}/embeddings.pt"
    mask_root_path="${mvtec_mask_root_path}/${name}"
    find $img_root_path -type f | while read file; 
    do
        file=$(basename $file)
        img_path="${img_root_path}/${file}"
        mask_path="${mask_root_path}/${categories[i]}"
        for ((j=0;j<2;j++))
        do
            files=$(find "$mask_path" -type f | shuf)
            mask_file=$(echo "$files" | head -n 1)
            mask_file=$(basename $mask_file)
            mask_path_finally="${mask_path}/${mask_file}"
            out_path="outputs/${name}/${categories[$i]}/${file:0:3}-${mask_file:0:3}"
            python scripts/txt2img.py \
                --ddim_eta 0.0 \
                --n_samples 1 \
                --n_iter 2 \
                --scale 10.0 \
                --ddim_steps 50 \
                --embedding_path $embbeding_path \
                --ckpt_path models/ldm/text2img-large/model.ckpt \
                --prompt "*" \
                --mask_prompt $mask_path_finally \
                --image_prompt $img_path \
                --outdir $out_path
        done
    done
done

# 11. tile
categories=("crack" "glue_strip" "gray_stroke" "oil" "rough")
name="tile"
img_root_path="${mvtec_root_path}/${name}/train/good"
for ((i=0;i<5;i++))
do
    embbeding_path="logs/${name}_${categories[$i]}/embeddings.pt"
    mask_root_path="${mvtec_mask_root_path}/${name}"
    find $img_root_path -type f | while read file; 
    do
        file=$(basename $file)
        img_path="${img_root_path}/${file}"
        mask_path="${mask_root_path}/${categories[i]}"
        for ((j=0;j<2;j++))
        do
            files=$(find "$mask_path" -type f | shuf)
            mask_file=$(echo "$files" | head -n 1)
            mask_file=$(basename $mask_file)
            mask_path_finally="${mask_path}/${mask_file}"
            out_path="outputs/${name}/${categories[$i]}/${file:0:3}-${mask_file:0:3}"
            python scripts/txt2img.py \
                --ddim_eta 0.0 \
                --n_samples 1 \
                --n_iter 2 \
                --scale 10.0 \
                --ddim_steps 50 \
                --embedding_path $embbeding_path \
                --ckpt_path models/ldm/text2img-large/model.ckpt \
                --prompt "*" \
                --mask_prompt $mask_path_finally \
                --image_prompt $img_path \
                --outdir $out_path
        done
    done
done

# 12. toothbrush
categories=("defective")
name="toothbrush"
img_root_path="${mvtec_root_path}/${name}/train/good"
for ((i=0;i<1;i++))
do
    embbeding_path="logs/${name}_${categories[$i]}/embeddings.pt"
    mask_root_path="${mvtec_mask_root_path}/${name}"
    find $img_root_path -type f | while read file; 
    do
        file=$(basename $file)
        img_path="${img_root_path}/${file}"
        mask_path="${mask_root_path}/${categories[i]}"
        for ((j=0;j<2;j++))
        do
            files=$(find "$mask_path" -type f | shuf)
            mask_file=$(echo "$files" | head -n 1)
            mask_file=$(basename $mask_file)
            mask_path_finally="${mask_path}/${mask_file}"
            out_path="outputs/${name}/${categories[$i]}/${file:0:3}-${mask_file:0:3}"
            python scripts/txt2img.py \
                --ddim_eta 0.0 \
                --n_samples 1 \
                --n_iter 2 \
                --scale 10.0 \
                --ddim_steps 50 \
                --embedding_path $embbeding_path \
                --ckpt_path models/ldm/text2img-large/model.ckpt \
                --prompt "*" \
                --mask_prompt $mask_path_finally \
                --image_prompt $img_path \
                --outdir $out_path
        done
    done
done

# 13. transistor
categories=("bent_lead" "cut_lead" "damaged_case" "misplaced")
name="transistor"
img_root_path="${mvtec_root_path}/${name}/train/good"
for ((i=0;i<4;i++))
do
    embbeding_path="logs/${name}_${categories[$i]}/embeddings.pt"
    mask_root_path="${mvtec_mask_root_path}/${name}"
    find $img_root_path -type f | while read file; 
    do
        file=$(basename $file)
        img_path="${img_root_path}/${file}"
        mask_path="${mask_root_path}/${categories[i]}"
        for ((j=0;j<2;j++))
        do
            files=$(find "$mask_path" -type f | shuf)
            mask_file=$(echo "$files" | head -n 1)
            mask_file=$(basename $mask_file)
            mask_path_finally="${mask_path}/${mask_file}"
            out_path="outputs/${name}/${categories[$i]}/${file:0:3}-${mask_file:0:3}"
            python scripts/txt2img.py \
                --ddim_eta 0.0 \
                --n_samples 1 \
                --n_iter 2 \
                --scale 10.0 \
                --ddim_steps 50 \
                --embedding_path $embbeding_path \
                --ckpt_path models/ldm/text2img-large/model.ckpt \
                --prompt "*" \
                --mask_prompt $mask_path_finally \
                --image_prompt $img_path \
                --outdir $out_path
        done
    done
done

# 14. wood
categories=("color" "combined" "hole" "liquid" "scratch")
name="wood"
img_root_path="${mvtec_root_path}/${name}/train/good"
for ((i=0;i<5;i++))
do
    embbeding_path="logs/${name}_${categories[$i]}/embeddings.pt"
    mask_root_path="${mvtec_mask_root_path}/${name}"
    find $img_root_path -type f | while read file; 
    do
        file=$(basename $file)
        img_path="${img_root_path}/${file}"
        mask_path="${mask_root_path}/${categories[i]}"
        for ((j=0;j<2;j++))
        do
            files=$(find "$mask_path" -type f | shuf)
            mask_file=$(echo "$files" | head -n 1)
            mask_file=$(basename $mask_file)
            mask_path_finally="${mask_path}/${mask_file}"
            out_path="outputs/${name}/${categories[$i]}/${file:0:3}-${mask_file:0:3}"
            python scripts/txt2img.py \
                --ddim_eta 0.0 \
                --n_samples 1 \
                --n_iter 2 \
                --scale 10.0 \
                --ddim_steps 50 \
                --embedding_path $embbeding_path \
                --ckpt_path models/ldm/text2img-large/model.ckpt \
                --prompt "*" \
                --mask_prompt $mask_path_finally \
                --image_prompt $img_path \
                --outdir $out_path
        done
    done
done

# 15. zipper
categories=("broken_teeth" "combined" "fabric_border" "fabric_interior" "rough" "split_teeth" "squeezed_teeth")
name="zipper"
img_root_path="${mvtec_root_path}/${name}/train/good"
for ((i=0;i<8;i++))
do
    embbeding_path="logs/${name}_${categories[$i]}/embeddings.pt"
    mask_root_path="${mvtec_mask_root_path}/${name}"
    find $img_root_path -type f | while read file; 
    do
        file=$(basename $file)
        img_path="${img_root_path}/${file}"
        mask_path="${mask_root_path}/${categories[i]}"
        for ((j=0;j<2;j++))
        do
            files=$(find "$mask_path" -type f | shuf)
            mask_file=$(echo "$files" | head -n 1)
            mask_file=$(basename $mask_file)
            mask_path_finally="${mask_path}/${mask_file}"
            out_path="outputs/${name}/${categories[$i]}/${file:0:3}-${mask_file:0:3}"
            python scripts/txt2img.py \
                --ddim_eta 0.0 \
                --n_samples 1 \
                --n_iter 2 \
                --scale 10.0 \
                --ddim_steps 50 \
                --embedding_path $embbeding_path \
                --ckpt_path models/ldm/text2img-large/model.ckpt \
                --prompt "*" \
                --mask_prompt $mask_path_finally \
                --image_prompt $img_path \
                --outdir $out_path
        done
    done
done

# 10.screw
categories=("manipulated_front" "scratch_head" "scratch_neck" "thread_side" "thread_top")
name="screw"
img_root_path="${mvtec_root_path}/${name}/train/good"
for ((i=0;i<5;i++))
do
    embbeding_path="logs/${name}_${categories[$i]}/embeddings.pt"
    mask_root_path="${mvtec_mask_root_path}/${name}"
    find $img_root_path -type f | while read file; 
    do
        file=$(basename $file)
        img_path="${img_root_path}/${file}"
        mask_path="${mask_root_path}/${file:0:3}"
        ls $mask_path
        find $mask_path -type f | while read mask_file;
        do
            mask_file=$(basename $mask_file)
            mask_path_finally="${mask_path}/${mask_file}"
            out_path="outputs/${name}/${categories[$i]}/${file:0:3}-${mask_file:0:3}"
            python scripts/txt2img.py \
                --ddim_eta 0.0 \
                --n_samples 1 \
                --n_iter 2 \
                --scale 10.0 \
                --ddim_steps 50 \
                --embedding_path $embbeding_path \
                --ckpt_path models/ldm/text2img-large/model.ckpt \
                --prompt "*" \
                --mask_prompt $mask_path_finally \
                --image_prompt $img_path \
                --outdir $out_path
        done
    done
done