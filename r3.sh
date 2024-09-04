# # generate p3 on test1k and re-eval
# path_to_conf="./configs/test1k/config_p3_test1k.yaml"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_masked_data_recycle_xmls.py --config ${path_to_conf}    # generate training data
# python eval/eval_xmls.py --config ${path_to_conf}
# # python eval/eval_xmls_llm.py --config ${path_to_conf}
# python eval/eval_xmls_llm2.py --config ${path_to_conf}

# generate p3 on tr20k
path_to_conf="./configs/tr20k/config_p3_tr20k.yaml"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_masked_data_recycle_xmls.py --config ${path_to_conf}    # generate training data
