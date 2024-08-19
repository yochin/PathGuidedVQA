# task_name="SimpleDescription_LFR_wOD"

# python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ../val100 --prompt-id 9998 --output ../output_T91_${task_name} | tee ../log_T91_${task_name}.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 9998 --output ../output_T1_${task_name} 2>&1 | tee ../log_T1_${task_name}.txt



# task_name="SimpleDescription_LFR"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 9999 --output ../output_T1_${task_name} 2>&1 | tee ../log_T1_${task_name}.txt
# python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ../val100 --prompt-id 9999 --output ../output_T91_${task_name} | tee ../log_T91_${task_name}.txt



# task_name="HBO_B91118_QActFirst_ExpDst_ASTARBOLD_ONLYDESC3"
# # db_dir="../val100_astarbold_r5"
# db_dir="../val100_astarbold_adaptR5_adaptTH1"

# python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ${db_dir} --prompt-id 1540302 --output ../output_T5_${task_name} | tee ../log_T5_${task_name}.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ${db_dir} --prompt-id 1540302 --output ../output_T1_${task_name} 2>&1 | tee ../log_T1_${task_name}.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ${db_dir} --prompt-id 1540303 --output ../output_T2_${task_name} 2>&1 | tee ../log_T2_${task_name}.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ${db_dir} --prompt-id 1540301 --output ../output_T9_${task_name} 2>&1 | tee ../log_T9_${task_name}.txt


# task_name="HBO_B91118_MASKED_NoPathR_Boundary"
# db_dir="../val100_astarbold_adaptR5_adaptTH1"

# python generate_data_by_yochin_masked.py --llava-model-dir dummy --model-name chatgpt --db-dir ${db_dir} --prompt-id 4510 --output ../output_T5_${task_name} | tee ../log_T5_${task_name}.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin_masked.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ${db_dir} --prompt-id 4510 --output ../output_T1_${task_name} | tee ../log_T1_${task_name}.txt


task_name="HBO_B91118_MASKED_NoPathR_Boundary_More2k_Wide"
db_dir="../valMore2k"
path_img="${db_dir}/original_images"
path_depths="${db_dir}/depth_anything_v2"
path_pd="${db_dir}/det_anno_pred"
path_gt="${db_dir}/det_anno_gt"
path_path="${db_dir}/paths"
path_gp="${db_dir}/anno"

# python utils/list_select_copy.py

# python generate_masked_data_args.py --llava-model-dir dummy --model-name chatgpt --db-dir ${db_dir} --images ${path_img} --det-anno-pred ${path_pd} --det-anno-gt ${path_gt} --gp-info ${path_gp} --paths ${path_path} --depths ${path_depths} --prompt-id 4510 --output ../output_T5_${task_name} | tee ../log_T5_${task_name}.txt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_masked_data_args.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ${db_dir} --images ${path_img} --det-anno-pred ${path_pd} --det-anno-gt ${path_gt} --gp-info ${path_gp} --paths ${path_path} --depths ${path_depths} --prompt-id 4510 --output ../output_T1_${task_name} | tee ../log_T1_${task_name}.txt
