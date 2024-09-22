# 1. bottle
categories=("broken_large" "broken_small" "contamination")
words=("broken" "broken" "broken")
for ((i=0;i<3;i++))
do
    name="v2_bottle_${categories[$i]}"
    path="mvtec_train_data/bottle/${categories[$i]}"
    word=${words[$i]}
    python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 2. cable
categories=("bent_wire" "cable_swap" "combined" "cut_inner_insulation" "cut_outer_insulation" "missing_cable" "missing_wire" "poke_insulation")
words=("bent" "swap" "combined" "cut" "cut" "missing" "missing" "poke")
for ((i=0;i<8;i++))
do
    name="v2_cable_${categories[$i]}"
    path="mvtec_train_data/cable/${categories[$i]}"
    word=${words[$i]}
    python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 3. capsule
categories=("crack" "faulty_imprint" "poke" "scratch" "squeeze")
words=("crack" "imprint" "poke" "scratch" "squeeze")
for ((i=0;i<5;i++))
do
    name="v2_capsule_${categories[$i]}"
    path="mvtec_train_data/capsule/${categories[$i]}"
    word=${words[$i]}
    python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 4. carpet
categories=("color" "cut" "hole" "metal_contamination" "thread")
words=("color" "cut" "hole" "contamination" "thread")
for ((i=0;i<5;i++)) 
do
    name="v2_carpet_${categories[$i]}"
    path="mvtec_train_data/carpet/${categories[$i]}"
    word=${words[$i]}
    python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 5. grid
categories=("bent" "broken" "glue" "metal_contamination" "thread")
words=("bent" "broken" "glue" "contamination" "thread")
for ((i=0;i<5;i++)) 
do
    name="v2_grid_${categories[$i]}"
    path="mvtec_train_data/grid/${categories[$i]}"
    word=${words[$i]}
    python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 6. hazelnut
categories=("crack" "cut" "hole" "print")
words=("crack" "cut" "hole" "print")
for ((i=0;i<4;i++)) 
do
    name="v2_hazelnut_${categories[$i]}"
    path="mvtec_train_data/hazelnut/${categories[$i]}"
    word=${words[$i]}
    python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 7. leather
categories=("color" "cut" "fold" "glue" "poke")
words=("color" "cut" "fold" "glue" "poke")
for ((i=0;i<5;i++)) 
do
    name="v2_leather_${categories[$i]}"
    path="mvtec_train_data/leather/${categories[$i]}"
    word=${words[$i]}
    python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 8. metal_nut
categories=("bent" "color" "flip" "scratch")
words=("bent" "color" "flip" "scratch")
for ((i=0;i<4;i++)) 
do
    name="v2_metal_nut_${categories[$i]}"
    path="mvtec_train_data/metal_nut/${categories[$i]}"
    word=${words[$i]}
    python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 9. pill
categories=("color" "combined" "contamination" "crack" "faulty_imprint" "pill_type" "scratch")
words=("color" "combined" "contamination" "crack" "imprint" "pill" "scratch")
for ((i=0;i<7;i++)) 
do
    name="v2_pill_${categories[$i]}"
    path="mvtec_train_data/pill/${categories[$i]}"
    word=${words[$i]}
    python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 10. screw
categories=("manipulated_front" "scratch_head" "scratch_neck" "thread_side" "thread_top")
words=("manipulated_front" "scratch" "scratch" "thread" "thread")
for ((i=0;i<5;i++)) 
do
    name="v2_screw_${categories[$i]}"
    path="mvtec_train_data/screw/${categories[$i]}"
    word=${words[$i]}
    python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 11. tile
categories=("crack" "glue_strip" "gray_stroke" "oil" "rough")
words=("crack" "strip" "stroke" "oil" "rough")
for ((i=0;i<5;i++)) 
do
    name="v2_tile_${categories[$i]}"
    path="mvtec_train_data/tile/${categories[$i]}"
    word=${words[$i]}
    python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 12. toothbrush
categories=("defective")
words=("defective")
for ((i=0;i<1;i++)) 
do
    name="v2_toothbrush_${categories[$i]}"
    path="mvtec_train_data/toothbrush/${categories[$i]}"
    word=${words[$i]}
    python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 13. transistor
categories=("bent_lead" "cut_lead" "damaged_case" "misplaced")
words=("bent" "cut" "damaged" "placed")
for ((i=0;i<4;i++)) 
do
    name="v2_transistor_${categories[$i]}"
    path="mvtec_train_data/transistor/${categories[$i]}"
    word=${words[$i]}
    python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 14. wood
categories=("color" "combined" "hole" "liquid" "scratch")
words=("color" "combined" "hole" "liquid" "scratch")
for ((i=0;i<5;i++)) 
do
    name="v2_wood_${categories[$i]}"
    path="mvtec_train_data/wood/${categories[$i]}"
    word=${words[$i]}
    python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done

# 15. zipper
categories=("broken_teeth" "combined" "fabric_border" "fabric_interior" "rough" "split_teeth" "squeezed_teeth")
words=("broken" "combined" "fabric" "fabric" "rough" "split" "squeezed")
for ((i=0;i<7;i++)) 
do
    name="v2_zipper_${categories[$i]}"
    path="mvtec_train_data/zipper/${categories[$i]}"
    word=${words[$i]}
    python main.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t --actual_resume models/ldm/text2img-large/model.ckpt \
    -n $name \
    --gpus 0, \
    --data_root $path \
    --init_word "defect"
done