path_to_conf="./config.yaml"
# python utils/find_select_copy.py --config ${path_to_conf_1}   # copy images from server to local
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python generate_masked_data.py --config ${path_to_conf}    # generate training data
