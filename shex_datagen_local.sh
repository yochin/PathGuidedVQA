# # step1. get the long answer sentence
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_data_by_yochin.py --llava-model-dir /mnt/data_disk2/models/vlms/llava/llava-v1.6-34b --model-name llava16_cli --db-dir /mnt/data_disk/dbs/gd_datagen/val100 --prompt-id 1119 --output res/datagen/output_t11 
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_data_by_yochin.py --llava-model-dir /mnt/data_disk2/models/vlms/llava/llava-v1.6-34b --model-name llava16_cli --db-dir /mnt/data_disk/dbs/gd_datagen/val100 --prompt-id 1118 --output res/datagen/output_t12  
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_data_by_yochin.py --llava-model-dir /mnt/data_disk2/models/vlms/llava/llava-v1.6-34b --model-name llava16_cli --db-dir /mnt/data_disk/dbs/gd_datagen/val100 --ex-dir examples --prompt-id 1119 --output res/datagen/output_t13  
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_data_by_yochin.py --llava-model-dir /mnt/data_disk2/models/vlms/llava/llava-v1.6-34b --model-name llava16_cli --db-dir /mnt/data_disk/dbs/gd_datagen/val100 --prompt-id 1118 --ex-dir examples --output res/datagen/output_t14  
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_data_by_yochin.py --llava-model-dir dummy --model-name chatgpt --db-dir /mnt/data_disk/dbs/gd_datagen/val100 --prompt-id 1119 --output res/datagen/output_t15

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_data_by_yochin.py --llava-model-dir /mnt/data_disk2/models/vlms/llava/llava-v1.6-34b --model-name llava16_cli --db-dir /mnt/data_disk/dbs/gd_datagen/val100 --prompt-id 91118 --output res/datagen/output_t918
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_data_by_yochin.py --llava-model-dir /mnt/data_disk2/models/vlms/llava/llava-v1.6-34b --model-name llava16_cli --db-dir /mnt/data_disk/dbs/gd_datagen/val100 --prompt-id 91138 --output res/datagen/output_t938
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_data_by_yochin.py --llava-model-dir /mnt/data_disk2/models/vlms/llava/llava-v1.6-34b --model-name llava16_cli --db-dir /mnt/data_disk/dbs/gd_datagen/val100 --prompt-id 91128 --output res/datagen/output_t928
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_data_by_yochin.py --llava-model-dir /mnt/data_disk2/models/vlms/llava/llava-v1.6-34b --model-name llava16_cli --db-dir /mnt/data_disk/dbs/gd_datagen/val100 --prompt-id 91158 --output res/datagen/output_t958


# # step2. get the short word answer from long answer sentences using GPT4
#for tt in 11 12 13 14 15
for tt in 918
do
    # just remove special tokens and merge into one long sentences.
    python prepare_eval_clipscore.py --gt-dir res/datagen/output_t"$tt"/qa --output-dir res/datagen/output_t"$tt"/eval
    # extract short answer from long sentences using GPT4
    python prepare_eval.py --gt-dir res/datagen/output_t"$tt"/qa --output-dir res/datagen/output_t"$tt"/eval --prompt-id 1118
done

# step3. evaluation
# get clipscore
# run ./eval/clipscore/run_eval_clipscore_by_hbo.sh
# get acc and prec, recall, f1
#for tt in 11 12 13 14 15
for tt in 918
do
    python eval_accuracy.py --pred-dir res/datagen/output_t"$tt" --gt-dir /mnt/data_disk/dbs/gd_datagen/val100/anno
done
