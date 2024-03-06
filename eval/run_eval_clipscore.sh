for tt in 11 12 13 14 15
do
    python ./clipscore/clipscore.py ../../output_t"$tt"/eval/pred_one_sentence.json ../../val100/images/ --save_per_instance ../../output_t"$tt"/eval/clipscore_per_inst.json | tee ../../output_t"$tt"/eval/clipscore_result.txt
done