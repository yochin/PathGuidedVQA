# # generate tr data
# path_to_conf="./config_dup.yaml"
# # # python utils/find_select_copy.py --config ${path_to_conf}   # copy images from server to local
# # CUDA_VISIBLE_DEVICES=0,1,4,5 python generate_masked_data.py --config ${path_to_conf}    # generate training data
# python utils/xml_to_json.py --config ${path_to_conf}    # xml to json
# python utils/search_and_move.py --config ${path_to_conf}    # move images

# Test1k on pipeline
path_to_conf="./configs/test1k/config_p4_test1k.yaml"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_masked_data.py --config ${path_to_conf}    # generate training data
python eval/eval_xmls.py --config ${path_to_conf}
# python eval/eval_xmls_llm.py --config ${path_to_conf}
python eval/eval_xmls_llm2.py --config ${path_to_conf}

# # Test1k on baseline
path_to_conf="./configs/test1k/config_p5_test1k.yaml"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_masked_data.py --config ${path_to_conf}    # generate training data
python eval/eval_xmls.py --config ${path_to_conf}
# python eval/eval_xmls_llm.py --config ${path_to_conf}
python eval/eval_xmls_llm2.py --config ${path_to_conf}

# # Test1k on baseline
path_to_conf="./configs/test1k/config_p7_test1k.yaml"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_masked_data.py --config ${path_to_conf}    # generate training data
python eval/eval_xmls.py --config ${path_to_conf}
# python eval/eval_xmls_llm.py --config ${path_to_conf}
python eval/eval_xmls_llm2.py --config ${path_to_conf}

# Test1k on pipeline
path_to_conf="./configs/test1k/config_p6_test1k.yaml"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_masked_data.py --config ${path_to_conf}    # generate training data
python eval/eval_xmls.py --config ${path_to_conf}
# python eval/eval_xmls_llm.py --config ${path_to_conf}
python eval/eval_xmls_llm2.py --config ${path_to_conf}
