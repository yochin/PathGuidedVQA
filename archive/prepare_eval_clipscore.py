import os
import json
import argparse
import openai
from llava.serve.cli import init_llava16_model_cli, run_llava16_model_cli
from transformers import set_seed

from tools import read_gt_pair_from_xml
import pdb

# from predicted result xml files,
# extract 3 answers 
# 1. long sentence for clipscore
# 2. 4 decisions
# 3. obstacles

def parse_args():
     parser = argparse.ArgumentParser(
        description='Generate json file for the llava training framework')

     parser.add_argument(
         '--gt-dir', metavar='DIRECTORY for xml files', 
         help='directory which contains images and object properties')
     
     parser.add_argument(
         '--output-dir', metavar='DERECTORY for several answer files', 
         help='derectory for several output files')

     return parser.parse_args()
     

def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    path_output_clipscore = os.path.join(args.output_dir, 'pred_one_sentence.json')

    files = os.listdir(args.gt_dir)
    xml_files = [f for f in files if f.endswith(('.xml'))]
    xml_files = sorted(xml_files)

    list_removal_tokens = ['<|startoftext|>', '<|im_end|>', '[!@#$NEXT!@#$]']

    res_clipscore = {}
    for xml_file in xml_files:
        xml_path = os.path.join(args.gt_dir, xml_file)
        img_filename, answer = read_gt_pair_from_xml(xml_path)

        img_filename_only = os.path.splitext(img_filename)[0]

        # clipscore
        for rem in list_removal_tokens:
            answer = answer.replace(rem, '')

        res_clipscore[img_filename_only] = answer

    with open(path_output_clipscore, 'w', encoding='utf-8') as json_file:
        json.dump(res_clipscore, json_file, indent="\t", ensure_ascii=False)


    return
   

if __name__ == '__main__':
    main()
