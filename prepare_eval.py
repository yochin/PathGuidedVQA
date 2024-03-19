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
     
     parser.add_argument(
         '--prompt-id', default=18, type=int)

     return parser.parse_args()
     

def main():
    args = parse_args()

    use_llava = False    # generate json
    use_gpt = True     # generate folder and txt file, sometimes lost the connection

    if use_llava:
        llava_model_path = '../llm_models/llava/llava-v1.6-34b'
        llava_model_base_path = None
        llava_tokenizer, llava_model, llava_image_processor, llava_context_len, llava_model_name, llava_input_conv_mode = init_llava16_model_cli(model_path=llava_model_path, model_base=llava_model_base_path, input_conv_mode=None)
    if use_gpt:
        openai.api_key = "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"

    # set the output directory
    path_output_action_llava = os.path.join(args.output_dir, 'pred_action_llava')
    path_output_obs = os.path.join(args.output_dir, 'pred_obs')

    if not os.path.exists(path_output_action_llava):
        os.makedirs(path_output_action_llava)
    
    if not os.path.exists(path_output_obs):
        os.makedirs(path_output_obs)
    

    # read the list of answer files
    files = os.listdir(args.gt_dir)
    xml_files = [f for f in files if f.endswith(('.xml'))]
    xml_files = sorted(xml_files)

    list_removal_tokens = ['<|startoftext|>', '<|im_end|>', '[!@#$NEXT!@#$]']

    for xml_file in xml_files:
        xml_path = os.path.join(args.gt_dir, xml_file)
        img_filename, answer = read_gt_pair_from_xml(xml_path)

        img_filename_only = os.path.splitext(img_filename)[0]

        # # debug, temp
        # answer = answer.split('[!@#$NEXT!@#$]')[0]
        # print('split answer and use the first one')

        # remove special tokens
        for rem in list_removal_tokens:
            answer = answer.replace(rem, '')

        # actions - word matching
        l_answer = answer.lower()

        # actions, obstacles - using llm
        list_prompt = []
        
        if args.prompt_id == 1118:
            list_prompt.append(f'{answer}\n Based on the description, what action is recommended? Choose from the following options. A) Go straight, B) Go left 45, C) Go right 45, D) Stop. Say only the answer.')
        elif args.prompt_id == 11118:
            list_prompt.append(f'{answer}\n Based on the description, what action is recommended? Choose from the following options. A) Go right 45, B) Stop, C) Go straight, D) Go left 45. Say only the answer.')
        elif args.prompt_id == 1158:
            list_prompt.append(f'{answer}\n Based on the description, what action is recommended? Choose from the following options. Go straight, Go left 45, Go right 45, Stop. Say only the answer.')
        elif args.prompt_id == 11158:
            list_prompt.append(f'{answer}\n Based on the description, what action is recommended? Choose from the following options. Go right 45, Stop, Go straight, Go left 45. Say only the answer.')
        elif args.prompt_id == 2118:
            list_prompt.append(f'{answer}\n Based on the description, what action is now recommended to get the destination? Choose from the following options. A) Go straight, B) Move forward slightly to the left, C) Move forward slightly to the right, D) Wait. Say only the answer.')
        elif args.prompt_id == 2158:
            list_prompt.append(f'{answer}\n Based on the description, what action is now recommended to get the destination? Choose from the following options. Go straight, Move forward slightly to the left, Move forward slightly to the right, Wait. Say only the answer.')
        elif args.prompt_id == 3118:
            list_prompt.append(f'{answer}\n Based on the description, what action is recommended at the first step and at the second step? Choose from the following options. A) Go straight, B) Go left 45, C) Go right 45, D) Stop. Say only the answer.')
        elif args.prompt_id == 3158:
            list_prompt.append(f'{answer}\n Based on the description, what action is recommended at the first step and at the second step? Choose from the following options. Go straight, Go left 45, Go right 45, Stop. Say only the answer.')

        list_prompt.append(f'{answer}\n Based on the description, what obstacles are on the path? List one by one. Say only the answer. Use a comma as a separator. If there is no obstacles, say "no obstacles".')
            
        if use_llava:
            set_seed(42)
            input_temperature = 0.6
            input_top_p = 0.9
            list_answer = []
            for prompt in list_prompt:
                answer = run_llava16_model_cli(llava_tokenizer, llava_model, llava_image_processor, llava_context_len, llava_model_name, 
                                                image_files=[], list_queries=[prompt], input_conv_mode=llava_input_conv_mode,
                                                input_temperature=input_temperature, input_top_p=input_top_p, input_num_beams=1,
                                                input_max_new_tokens=512, input_debug=True, use_ex_image=False)
                list_answer.append(answer)

        if use_gpt:
            list_answer = []
            for prompt in list_prompt:
                response = openai.chat.completions.create(
                    # model="gpt-4",
                    model='gpt-3.5-turbo',
                    messages=[
                        {
                            "role": "user", 
                            "content": prompt,
                        }
                    ],
                    max_tokens=1024,
                )
                answer = response.choices[0].message.content
                list_answer.append(answer)

        print('list_prompt: ', list_prompt)
        print('list_answer: ', list_answer)
        print('\n')

        # obstacles
        with open(os.path.join(path_output_obs, img_filename_only + '.txt'), 'w') as fid:
            fid.write(list_answer[1])

        # actions - llava
        with open(os.path.join(path_output_action_llava, img_filename_only + '.txt'), 'w') as fid:
            fid.write(list_answer[0])

    return
   

if __name__ == '__main__':
    main()
