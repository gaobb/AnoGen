import os
from PIL import Image
from shutil import copyfile

if __name__ == '__main__':
    mvtec_info = {
    'bottle': ['broken_large', 'broken_small', 'contamination'],
    'cable': ['bent_wire', 'cable_swap', 'combined', 'cut_inner_insulation', 'cut_outer_insulation', 'missing_cable','missing_wire','poke_insulation'],
    'capsule': ['crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze'],
    'carpet': ['color', 'cut', 'hole', 'metal_contamination', 'thread'],
    'grid': ['bent', 'broken', 'glue', 'metal_contamination', 'thread'],
    'hazelnut': ['crack', 'cut', 'hole', 'print'],
    'leather': ['color', 'cut', 'fold', 'glue', 'poke'],
    'metal_nut': ['bent', 'color', 'scratch'],
    'pill': ['color', 'combined', 'contamination', 'crack', 'faulty_imprint', 'pill_type', 'scratch'],
    'tile': ['crack', 'glue_strip', 'gray_stroke', 'oil', 'rough'],
    'toothbrush': ['defective'],
    'transistor': ['bent_lead', 'cut_lead', 'damaged_case', 'misplaced'],
    'wood': ['color', 'combined', 'hole', 'liquid', 'scratch'],
    'zipper': ['broken_teeth', 'combined', 'fabric_border', 'fabric_interior', 'split_teeth', 'rough', 'squeezed_teeth'],
    }

    #生成图像的路径
    Generate_path = 'outputs4/' 
    #mask的路径
    Mask_path = 'mvtec_masks'
    #新数据集的路径
    Save_root_path = '/apdcephfs/private_laurelgui/data/mvtec_anomaly_4'

    #处理除了screw以外的类别
    for key in mvtec_info.keys():
        for defect in mvtec_info[key]:
            if True:
                print("making {}-{}: ".format(key, defect))
                root_path = os.path.join(Generate_path, key, defect)
                all_files = os.listdir(root_path) # [000-017,001-022,...]


                save_root_image_path = os.path.join(Save_root_path, key, 'images')
                save_root_mask_path = os.path.join(Save_root_path, key, 'masks')

                if not os.path.exists(save_root_image_path):
                    os.makedirs(save_root_image_path)
                if not os.path.exists(save_root_mask_path):
                    os.makedirs(save_root_mask_path)
                
                all_files = sorted(all_files)

                for one_file in all_files:
                    img1_name = one_file + '-0000-' + defect + '.png'
                    img2_name = one_file + '-0001-' + defect + '.png'

                    copyfile(os.path.join(root_path, one_file, 'samples/0000.jpg'), os.path.join(save_root_image_path, img1_name))
                    copyfile(os.path.join(root_path, one_file, 'samples/0001.jpg'), os.path.join(save_root_image_path, img2_name))

                        
                mask_files = os.listdir(os.path.join(Mask_path, key, defect))
                for one_file in mask_files:
                    # mask = Image.open(os.path.join(Mask_path, key, defect, one_file))
                    mask_name = one_file[:3] + "-" + defect + ".png"

                    copyfile(os.path.join(Mask_path, key, defect, one_file), os.path.join(save_root_mask_path, mask_name))

    # 处理screw
    defects = ['manipulated_front', 'scratch_head', 'scratch_neck', 'thread_side', 'thread_top']
    image_root_path = os.path.join(Generate_path, 'screw')

    save_root_image_path = os.path.join(Save_root_path, 'screw', 'images')
    save_root_mask_path = os.path.join(Save_root_path, 'screw', 'images')
    if not os.path.exists(save_root_image_path):
            os.makedirs(save_root_image_path)
    if not os.path.exists(save_root_mask_path):
            os.makedirs(save_root_mask_path)

    for defect in defects:
        print(defect)
        root_path = os.path.join(image_root_path, defect)
        all_files = os.listdir(root_path)

        for one_file in all_files:
            img1_name = one_file + '-0000-' + defect + '.png'
            img2_name = one_file + '-0001-' + defect + '.png'

            copyfile(os.path.join(root_path, one_file, 'samples/0000.jpg'), os.path.join(save_root_image_path, img1_name))
            copyfile(os.path.join(root_path, one_file, 'samples/0001.jpg'), os.path.join(save_root_image_path, img2_name))

    mask_root_path = os.path.join(Mask_path, 'screw')
    all_files = os.listdir(mask_root_path)
    
    for one_file in all_files:
        mask1_name = one_file + '-000.png'
        mask2_name = one_file + '-001.png'

        copyfile(os.path.join(mask_root_path, one_file, '000.png'), os.path.join(save_root_mask_path, mask1_name))
        copyfile(os.path.join(mask_root_path, one_file, '001.png'), os.path.join(save_root_mask_path, mask2_name))


