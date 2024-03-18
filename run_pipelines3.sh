# step1. get the long answer sentence
# python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ../val100 --prompt-id 11119 --output ../output_t15perm | tee ../log_t15perm.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 11119 --output ../output_t11perm | tee ../log_t11perm.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 11118 --output ../output_t12perm | tee ../log_t12perm.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --ex-dir ../examples_perm --prompt-id 11118 --output ../output_t14perm | tee ../log_t14perm.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --ex-dir ../examples_perm --prompt-id 11119 --output ../output_t13perm | tee ../log_t13perm.txt

# python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ../val100 --prompt-id 11159 --output ../output_t515perm | tee ../log_t515perm.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 11159 --output ../output_t511perm | tee ../log_t511perm.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 11158 --output ../output_t512perm | tee ../log_t512perm.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --ex-dir ../examples_perm_noA --prompt-id 11158 --output ../output_t514perm | tee ../log_t514perm.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --ex-dir ../examples_perm_noA --prompt-id 11159 --output ../output_t513perm | tee ../log_t513perm.txt

# python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ../val100 --prompt-id 1159 --output ../output_t515 | tee ../log_t515.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 1159 --output ../output_t511 | tee ../log_t511.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 1158 --output ../output_t512 | tee ../log_t512.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --ex-dir ../examples_noA --prompt-id 1158 --output ../output_t514 | tee ../log_t514.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --ex-dir ../examples_noA --prompt-id 1159 --output ../output_t513 | tee ../log_t513.txt


# # step2. get the short word answer from long answer sentences using GPT4
# for tt in 11 12 13 14
# do
#     # just remove special tokens and merge into one long sentences.
#     python prepare_eval_clipscore.py --gt-dir ../output_t"$tt"perm/qa --output-dir ../output_t"$tt"perm/eval
#     # extract short answer from long sentences using GPT4
#     python prepare_eval.py --gt-dir ../output_t"$tt"perm/qa --output-dir ../output_t"$tt"perm/eval --prompt-id 11118 | tee ../log_t"$tt"perm_step2.txt
# done
# for tt in 515 511 512 514 513
# do
#     # just remove special tokens and merge into one long sentences.
#     python prepare_eval_clipscore.py --gt-dir ../output_t"$tt"perm/qa --output-dir ../output_t"$tt"perm/eval
#     # extract short answer from long sentences using GPT4
#     python prepare_eval.py --gt-dir ../output_t"$tt"perm/qa --output-dir ../output_t"$tt"perm/eval --prompt-id 11158 | tee ../log_t"$tt"perm_step2.txt
# done
# for tt in 515 511 512 514 513
# do
#     # just remove special tokens and merge into one long sentences.
#     python prepare_eval_clipscore.py --gt-dir ../output_t"$tt"/qa --output-dir ../output_t"$tt"/eval
#     # extract short answer from long sentences using GPT4
#     python prepare_eval.py --gt-dir ../output_t"$tt"/qa --output-dir ../output_t"$tt"/eval --prompt-id 1158 | tee ../log_t"$tt"_step2.txt
# done

# # # step3. evaluation
# # get clipscore
# # run eval/clipscore/run_eval_clipscore.sh
# # # get acc and prec, recall, f1
# for tt in 11 12 13 14
# do
#     python eval_accuracy.py --pred-dir ../output_t"$tt"perm --gt-dir ../val100/anno
# done
# # for tt in 515 511 512 514 513
# # do
# #     python eval_accuracy.py --pred-dir ../output_t"$tt"perm --gt-dir ../val100/anno
# # done
# # for tt in 515 511 512 514 513
# # do
# #     python eval_accuracy.py --pred-dir ../output_t"$tt" --gt-dir ../val100/anno
# # done
