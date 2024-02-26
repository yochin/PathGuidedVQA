CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_yochin --prompt-id 119 --output ../output_t1 | tee ../log_t1_JPG.txt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_yochin --prompt-id 118 --output ../output_t2 | tee ../log_t2_JPG.txt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_yochin --ex-dir ../examples --prompt-id 120 --output ../output_t3 | tee ../log_t3_JPG.txt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_yochin --ex-dir ../examples --prompt-id 130 --output ../output_t4 | tee ../log_t4_JPG.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_hbo --prompt-id 119 --output ../output_t1 | tee ../log_t1_hbo_JPG.txt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_hbo --prompt-id 118 --output ../output_t2 | tee ../log_t2_hbo_JPG.txt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_hbo --ex-dir ../examples --prompt-id 120 --output ../output_t3 | tee ../log_t3_hbo_JPG.txt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100_hbo --ex-dir ../examples --prompt-id 130 --output ../output_t4 | tee ../log_t4_hbo_JPG.txt