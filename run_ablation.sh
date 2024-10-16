# P4: Test1k on pipeline using masked destination
path_to_conf="./configs/test1k/config_p4_test1k.yaml"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_masked_data.py --config ${path_to_conf}    # generate training data
python eval/eval_xmls.py --config ${path_to_conf}
# python eval/eval_xmls_bertS_extWord.py --config ${path_to_conf}
python eval/eval_xmls_llmjudge.py --config ${path_to_conf}

# P5: Test1k on baseline using LLM for decision
path_to_conf="./configs/test1k/config_p5_test1k.yaml"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_masked_data.py --config ${path_to_conf}    # generate training data
python eval/eval_xmls.py --config ${path_to_conf}
# python eval/eval_xmls_bertS_extWord.py --config ${path_to_conf}
python eval/eval_xmls_llmjudge.py --config ${path_to_conf}

# P3: Test1k on pipeline using masked images only for left, right, and path
path_to_conf="./configs/test1k/config_p3_test1k.yaml"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_masked_data_recycle_xmls.py --config ${path_to_conf}    # generate training data
python eval/eval_xmls.py --config ${path_to_conf}
# python eval/eval_xmls_llm.py --config ${path_to_conf}
python eval/eval_xmls_llm2.py --config ${path_to_conf}

# P6: Test1k on pipeline using detection information
path_to_conf="./configs/test1k/config_p6_test1k.yaml"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_masked_data.py --config ${path_to_conf}    # generate training data
python eval/eval_xmls.py --config ${path_to_conf}
# python eval/eval_xmls_bertS_extWord.py --config ${path_to_conf}
python eval/eval_xmls_llmjudge.py --config ${path_to_conf}

# P7: Test1k on baseline of multi-turn query
path_to_conf="./configs/test1k/config_p7_test1k.yaml"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_masked_data.py --config ${path_to_conf}    # generate training data
python eval/eval_xmls.py --config ${path_to_conf}
# python eval/eval_xmls_bertS_extWord.py --config ${path_to_conf}
python eval/eval_xmls_llmjudge.py --config ${path_to_conf}

# P8: Test1k on baseline of single-turn query
path_to_conf="./configs/test1k/config_p8_test1k.yaml"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_masked_data_single_turn.py --config ${path_to_conf}    # generate training data
python eval/eval_xmls.py --config ${path_to_conf}
# python eval/eval_xmls_bertS_extWord.py --config ${path_to_conf}
python eval/eval_xmls_llmjudge.py --config ${path_to_conf}
