# Generating a training dataset
# generate tr data
# path_to_conf="./config.yaml"
# # python utils/find_select_copy.py --config ${path_to_conf}   # copy images from server to local
# CUDA_VISIBLE_DEVICES=2,3,6,7 python generate_masked_data.py --config ${path_to_conf}    # generate training data
# python utils/xml_to_json.py --config ${path_to_conf}    # xml to json
# python utils/search_and_move.py --config ${path_to_conf}    # move images

# generate tr data
# path_to_conf="./config_dup.yaml"
# # python utils/find_select_copy.py --config ${path_to_conf}   # copy images from server to local
# CUDA_VISIBLE_DEVICES=0,1,4,5 python generate_masked_data.py --config ${path_to_conf}    # generate training data
# python utils/xml_to_json.py --config ${path_to_conf}    # xml to json
# python utils/search_and_move.py --config ${path_to_conf}    # move images

# generate p3 on tr20k
path_to_conf="./configs/tr20k/config_p3_tr20k.yaml"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python generate_masked_data_recycle_xmls.py --config ${path_to_conf}    # generate training data
python utils/xml_to_json.py --config ${path_to_conf}    # xml to json
# python utils/search_and_move.py --config ${path_to_conf}    # move images
