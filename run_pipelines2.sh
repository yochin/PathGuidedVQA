# step2. get the short word answer from long answer sentences using GPT4
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
# for tt in 511
# do
#     # just remove special tokens and merge into one long sentences.
#     python prepare_eval_clipscore.py --gt-dir ../output_t"$tt"/qa --output-dir ../output_t"$tt"/eval
#     # extract short answer from long sentences using GPT4
#     python prepare_eval.py --gt-dir ../output_t"$tt"/qa --output-dir ../output_t"$tt"/eval --prompt-id 1158 | tee ../log_t"$tt"_step2.txt
# done

# # step3. evaluation
# get clipscore
# run eval/clipscore/run_eval_clipscore.sh
# # get acc and prec, recall, f1
# for tt in 11 12 13 14 15
# do
#     python eval_accuracy.py --pred-dir ../output_t"$tt"perm --gt-dir ../val100/anno
# done
# for tt in 514
# do
#     python eval_accuracy.py --pred-dir ../output_t"$tt"perm --gt-dir ../val100/anno
# done
for tt in 51515
do
    python eval_accuracy.py --pred-dir ../output_t"$tt" --gt-dir ../val100/anno
done
