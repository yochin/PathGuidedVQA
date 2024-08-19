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



# 

# python generate_masked_data.py --path_to_conf ${path_to_conf} | tee ../log_T5_${task_name}.txt

# path_to_conf="./config.yaml"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_masked_data.py --config ${path_to_conf}

path_to_conf_1="./config.yaml"
# python utils/find_select_copy.py --config ${path_to_conf_1}
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_masked_data.py --config ${path_to_conf_1}
