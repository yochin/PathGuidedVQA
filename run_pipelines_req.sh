# step1. get the long answer sentence
task_name="ReQuest_DesPres4_LCR"

python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir ../val100 --prompt-id 1819 --output ../output_${task_name}_t15 | tee ../log_${task_name}_t15.txt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 1819 --output ../output_${task_name}_t11 | tee ../log_${task_name}_t11.txt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_data_by_yochin.py --llava-model-dir ../llm_models/llava/llava-v1.6-34b --model-name llava16_cli --db-dir ../val100 --prompt-id 1859 --output ../output_${task_name}_t16 | tee ../log_${task_name}_t16.txt

# step2. get the short word answer from long answer sentences using GPT4
for tt in 11 15
do
    # just remove special tokens and merge into one long sentences.
    python prepare_eval_clipscore.py --gt-dir ../output_${task_name}_t"$tt"/qa --output-dir ../output_${task_name}_t"$tt"/eval
    # extract short answer from long sentences using GPT4
    # python prepare_eval.py --gt-dir ../output_${task_name}_t"$tt"/qa --output-dir ../output_${task_name}_t"$tt"/eval --prompt-id 2118 | tee ../log_${task_name}_t"$tt"_step2_2118.txt
    python prepare_eval.py --gt-dir ../output_${task_name}_t"$tt"/qa --output-dir ../output_${task_name}_t"$tt"/eval --prompt-id 1118 | tee ../log_${task_name}_t"$tt"_step2.txt
done
for tt in 16
do
    # just remove special tokens and merge into one long sentences.
    python prepare_eval_clipscore.py --gt-dir ../output_${task_name}_t"$tt"/qa --output-dir ../output_${task_name}_t"$tt"/eval
    # extract short answer from long sentences using GPT4
    # python prepare_eval.py --gt-dir ../output_${task_name}_t"$tt"/qa --output-dir ../output_${task_name}_t"$tt"/eval --prompt-id 2158 | tee ../log_${task_name}_t"$tt"_step2_2158.txt
    python prepare_eval.py --gt-dir ../output_${task_name}_t"$tt"/qa --output-dir ../output_${task_name}_t"$tt"/eval --prompt-id 1158 | tee ../log_${task_name}_t"$tt"_step2.txt
done

# step3. evaluation
# get clipscore
# run eval/clipscore/run_eval_clipscore.sh
# get acc and prec, recall, f1
for tt in 11 15 16
do
    python eval_accuracy.py --pred-dir ../output_${task_name}_t"$tt" --gt-dir ../val100/anno
done
