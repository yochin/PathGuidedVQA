# # step1. get the long answer sentence
# task_name="HBO"

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 91118 --output ../output_${task_name}_t11 | tee ../log_${task_name}_t11.txt
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 91148 --output ../output_${task_name}_t12 | tee ../log_${task_name}_t12.txt

# # step2. get the short word answer from long answer sentences using GPT4
# for tt in 11 12
# do
#     # # just remove special tokens and merge into one long sentences.
#     # python prepare_eval_clipscore.py --gt-dir ../output_${task_name}_t"$tt"/qa --output-dir ../output_${task_name}_t"$tt"/eval
#     # # extract short answer from long sentences using GPT4
#     # python prepare_eval.py --gt-dir ../output_${task_name}_t"$tt"/qa --output-dir ../output_${task_name}_t"$tt"/eval --prompt-id 1118 | tee ../log_${task_name}_t"$tt"_step2.txt    
#     # get acc and prec, recall, f1
#     python eval_accuracy.py --pred-dir ../output_${task_name}_t"$tt" --gt-dir ../val100/anno
# done


# task_name="HBO_B91118_QActFirst_ExpDst_ASTARBOLD_R5"

# # python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ../val100_astarbold_r5 --prompt-id 540302 --output ../output_T5_${task_name} | tee ../log_T5_${task_name}.txt
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_astarbold_r5 --prompt-id 540302 --output ../output_T1_${task_name} 2>&1 | tee ../log_T1_${task_name}.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_astarbold_r5 --prompt-id 540303 --output ../output_T2_${task_name} 2>&1 | tee ../log_T2_${task_name}.txt
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_astarbold_r5 --prompt-id 540301 --output ../output_T9_${task_name} 2>&1 | tee ../log_T9_${task_name}.txt

# # step2. get the short word answer from long answer sentences using GPT4
# for tt in 2
# do
#     mv ../log_T"$tt"_${task_name}.txt ../output_T"$tt"_${task_name}/log_T"$tt"_${task_name}.txt

#     # just remove special tokens and merge into one long sentences.
#     python prepare_eval_clipscore.py --gt-dir ../output_T"$tt"_${task_name}/qa --output-dir ../output_T"$tt"_${task_name}/eval
#     # extract short answer from long sentences using GPT4
#     python prepare_eval.py --gt-dir ../output_T"$tt"_${task_name}/qa --output-dir ../output_T"$tt"_${task_name}/eval --prompt-id 1118 2>&1 | tee ../log_T"$tt"_${task_name}_step2.txt
#     mv ../log_T"$tt"_${task_name}_step2.txt ../output_T"$tt"_${task_name}/log_T"$tt"_${task_name}_step2.txt

#     python eval_accuracy.py --pred-dir ../output_T"$tt"_${task_name} --gt-dir ../val100/anno
# done



# task_name="HBO_B91118_QActFirst_ExpDst_ASTARBOLD_R5_LCR"

# # python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ../val100_astarbold_r5_lcr --prompt-id 540302 --output ../output_T5_${task_name} | tee ../log_T5_${task_name}.txt
# python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ../val100_astarbold_r5_lcr --prompt-id 540303 --output ../output_T52_${task_name} | tee ../log_T52_${task_name}.txt
# # python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ../val100_astarbold_r5_lcr --prompt-id 540301 --output ../output_T59_${task_name} | tee ../log_T59_${task_name}.txt
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_astarbold_r5_lcr --prompt-id 540302 --output ../output_T1_${task_name} 2>&1 | tee ../log_T1_${task_name}.txt
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_astarbold_r5_lcr --prompt-id 540303 --output ../output_T2_${task_name} 2>&1 | tee ../log_T2_${task_name}.txt
# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_astarbold_r5_lcr --prompt-id 540301 --output ../output_T9_${task_name} 2>&1 | tee ../log_T9_${task_name}.txt

# # step2. get the short word answer from long answer sentences using GPT4
# for tt in 52
# do
#     mv ../log_T"$tt"_${task_name}.txt ../output_T"$tt"_${task_name}/log_T"$tt"_${task_name}.txt

#     # just remove special tokens and merge into one long sentences.
#     python prepare_eval_clipscore.py --gt-dir ../output_T"$tt"_${task_name}/qa --output-dir ../output_T"$tt"_${task_name}/eval
#     # extract short answer from long sentences using GPT4
#     python prepare_eval.py --gt-dir ../output_T"$tt"_${task_name}/qa --output-dir ../output_T"$tt"_${task_name}/eval --prompt-id 1118 2>&1 | tee ../log_T"$tt"_${task_name}_step2.txt
#     mv ../log_T"$tt"_${task_name}_step2.txt ../output_T"$tt"_${task_name}/log_T"$tt"_${task_name}_step2.txt

#     python eval_accuracy.py --pred-dir ../output_T"$tt"_${task_name} --gt-dir ../val100/anno
# done




# task_name="HBO_B91118"

# python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ../val100 --prompt-id 41138 --output ../output_T52_${task_name} 2>&1 | tee ../log_T52_${task_name}.txt

# # step2. get the short word answer from long answer sentences using GPT4
# for tt in 52
# do
#     mv ../log_T"$tt"_${task_name}.txt ../output_T"$tt"_${task_name}/log_T"$tt"_${task_name}.txt

#     # just remove special tokens and merge into one long sentences.
#     python prepare_eval_clipscore.py --gt-dir ../output_T"$tt"_${task_name}/qa --output-dir ../output_T"$tt"_${task_name}/eval
#     # extract short answer from long sentences using GPT4
#     python prepare_eval.py --gt-dir ../output_T"$tt"_${task_name}/qa --output-dir ../output_T"$tt"_${task_name}/eval --prompt-id 1118 2>&1 | tee ../log_T"$tt"_${task_name}_step2.txt
#     mv ../log_T"$tt"_${task_name}_step2.txt ../output_T"$tt"_${task_name}/log_T"$tt"_${task_name}_step2.txt

#     python eval_accuracy.py --pred-dir ../output_T"$tt"_${task_name} --gt-dir ../val100/anno
# done


# task_name="HBO_B91118_QActFirst_ExpDst_ASTARBOLD_R5_LCR_NoDesDIR"

# python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ../val100_astarbold_r5_lcr --prompt-id 540312 --output ../output_T5_${task_name} | tee ../log_T5_${task_name}.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_astarbold_r5_lcr --prompt-id 540312 --output ../output_T1_${task_name} 2>&1 | tee ../log_T1_${task_name}.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_astarbold_r5_lcr --prompt-id 540313 --output ../output_T2_${task_name} 2>&1 | tee ../log_T2_${task_name}.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_astarbold_r5_lcr --prompt-id 540311 --output ../output_T9_${task_name} 2>&1 | tee ../log_T9_${task_name}.txt

# # step2. get the short word answer from long answer sentences using GPT4
# for tt in 5 9 2 1
# do
#     mv ../log_T"$tt"_${task_name}.txt ../output_T"$tt"_${task_name}/log_T"$tt"_${task_name}.txt

#     # just remove special tokens and merge into one long sentences.
#     python prepare_eval_clipscore.py --gt-dir ../output_T"$tt"_${task_name}/qa --output-dir ../output_T"$tt"_${task_name}/eval
#     # extract short answer from long sentences using GPT4
#     python prepare_eval.py --gt-dir ../output_T"$tt"_${task_name}/qa --output-dir ../output_T"$tt"_${task_name}/eval --prompt-id 1118 2>&1 | tee ../log_T"$tt"_${task_name}_step2.txt
#     mv ../log_T"$tt"_${task_name}_step2.txt ../output_T"$tt"_${task_name}/log_T"$tt"_${task_name}_step2.txt

#     python eval_accuracy.py --pred-dir ../output_T"$tt"_${task_name} --gt-dir ../val100/anno
# done




# # task_name="HBO_B91118_QActFirst_ExpDst"

# # # # # # # python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ../val100 --prompt-id 41819 --output ../output_${task_name}_t15 | tee ../log_${task_name}_t15.txt
# # # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 41178 --output ../output_T9_${task_name} 2>&1 | tee ../log_T9_${task_name}.txt

# # # step2. get the short word answer from long answer sentences using GPT4
# # for tt in 9
# # do
# #     mv ../log_T"$tt"_${task_name}.txt ../output_T"$tt"_${task_name}/log_T"$tt"_${task_name}.txt

# #     # just remove special tokens and merge into one long sentences.
# #     python prepare_eval_clipscore.py --gt-dir ../output_T"$tt"_${task_name}/qa --output-dir ../output_T"$tt"_${task_name}/eval
# #     # extract short answer from long sentences using GPT4
# #     python prepare_eval.py --gt-dir ../output_T"$tt"_${task_name}/qa --output-dir ../output_T"$tt"_${task_name}/eval --prompt-id 1118 2>&1 | tee ../log_T"$tt"_${task_name}_step2.txt
# #     mv ../log_T"$tt"_${task_name}_step2.txt ../output_T"$tt"_${task_name}/log_T"$tt"_${task_name}_step2.txt

# #     python eval_accuracy.py --pred-dir ../output_T"$tt"_${task_name} --gt-dir ../val100/anno
# # done