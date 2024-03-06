# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_yochin --prompt-id 119 --output ../output_t1 | tee ../log_t1_JPG.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_yochin --prompt-id 118 --output ../output_t2 | tee ../log_t2_JPG.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_yochin --ex-dir ../examples --prompt-id 120 --output ../output_t3 | tee ../log_t3_JPG.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_yochin --ex-dir ../examples --prompt-id 130 --output ../output_t4 | tee ../log_t4_JPG.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_hbo --prompt-id 119 --output ../output_t1 | tee ../log_t1_hbo_JPG.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_hbo --prompt-id 118 --output ../output_t2 | tee ../log_t2_hbo_JPG.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_hbo --ex-dir ../examples --prompt-id 120 --output ../output_t3 | tee ../log_t3_hbo_JPG.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_hbo --ex-dir ../examples --prompt-id 130 --output ../output_t4 | tee ../log_t4_hbo_JPG.txt


# # get the long answer sentence
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 1119 --output ../output_t11 | tee ../log_t11.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 1118 --output ../output_t12 | tee ../log_t12.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --ex-dir ../examples --prompt-id 1119 --output ../output_t13 | tee ../log_t13.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --ex-dir ../examples --prompt-id 1118 --output ../output_t14 | tee ../log_t14.txt
# python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ../val100 --prompt-id 1119 --output ../output_t15 | tee ../log_t15.txt


# get the short word answer
for tt in 11 12 13 14 15
do
    # python prepare_eval_clipscore --gt-dir ../output_t"$tt"/qa --output-dir ../output_t"$tt"/eval
    python prepare_eval --gt-dir ../output_t"$tt"/qa --output-dir ../output_t"$tt"/eval
done