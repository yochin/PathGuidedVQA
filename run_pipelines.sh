# # step1. get the long answer sentence
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 1119 --output ../output_t11 | tee ../log_t11.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 1118 --output ../output_t12 | tee ../log_t12.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --ex-dir ../examples --prompt-id 1119 --output ../output_t13 | tee ../log_t13.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --ex-dir ../examples --prompt-id 1118 --output ../output_t14 | tee ../log_t14.txt
# python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ../val100 --prompt-id 1119 --output ../output_t15 | tee ../log_t15.txt

# # step2. get the short word answer from long answer sentences using GPT4
# for tt in 11 12 13 14 15
# do
#     # just remove special tokens and merge into one long sentences.
#     python prepare_eval_clipscore.py --gt-dir ../output_t"$tt"/qa --output-dir ../output_t"$tt"/eval
#     # extract short answer from long sentences using GPT4
#     python prepare_eval.py --gt-dir ../output_t"$tt"/qa --output-dir ../output_t"$tt"/eval
# done

# step3. evaluation
# # get clipscore
# # run eval/clipscore/run_eval_clipscore.sh
# get acc and prec, recall, f1
for tt in 11 12 13 14 15
do
    python eval_accuracy.py --pred-dir ../output_t"$tt" --gt-dir ../val100/anno
done
