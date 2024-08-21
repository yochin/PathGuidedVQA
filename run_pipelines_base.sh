# Test1k on pipeline - OK
path_to_conf="./config_test1k.yaml"
# python utils/find_select_copy.py --config ${path_to_conf_1}   # copy images from server to local
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_masked_data.py --config ${path_to_conf}    # generate training data

# Test1k on baseline
path_to_conf="./config_baseline_test1k.yaml"
# python utils/find_select_copy.py --config ${path_to_conf_1}   # copy images from server to local
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_masked_data.py --config ${path_to_conf}    # generate training data

# generate tr data
path_to_conf="./config.yaml"
# python utils/find_select_copy.py --config ${path_to_conf_1}   # copy images from server to local
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_masked_data.py --config ${path_to_conf}    # generate training data
