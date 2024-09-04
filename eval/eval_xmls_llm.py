from PIL import Image
import requests
import copy
import torch
import argparse
import glob
import xml.etree.ElementTree as ET
from nltk.translate.meteor_score import meteor_score
from bert_score import score
from rouge_score import rouge_scorer
import os, sys
import numpy as np
from datetime import datetime
import json
import re
import pdb
import argparse
import yaml
import spacy
from itertools import product

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from gpt_wrapper import gpt_wrapper
from llm_wrapper import llm_wrapper


# spaCy의 중형 영어 모델 로드
nlp = spacy.load('en_core_web_md')
# gpt_model = gpt_wrapper('gpt-4o-mini-2024-07-18', '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')
gpt_model = gpt_wrapper('gpt-4o-2024-08-06', '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')
# gpt_model = llm_wrapper('meta-llama/Meta-Llama-3.1-8B-Instruct')

def parse_args():
    parser = argparse.ArgumentParser(description='Copy image files')   

    parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()

    return args

def eval_text_llm_judge(gt_text, infer_text):
    llm_system = 'Please act as an impartial judge and evaluate the quality of the description provided by an ' \
                 'AI assistant to the user. Your evaluation should consider correctness and ' \
                 'helpfulness. You will be given a reference description and the assistant\'s description. You ' \
                 'evaluation should focus on the assistant\'s description. Begin your ' \
                 'evaluation by comparing the assistant\'s description with the reference description. Identify and ' \
                 'correct any mistakes. Be as objective as possible. After providing your explanation, you ' \
                 'must rate the response on a scale of 1 to 10 by strictly following this format: ' \
                 '"[[rating]]", for example: "Rating: [[5]]". \n'
    
    llm_prompt = '<|The Start of Reference Description|>\n' \
                 f'{gt_text}\n' \
                 '<|The End of Reference Description|>\n\n' \
                 '<|The Start of Assistant A\'s Description|>\n' \
                 f'{infer_text}\n' \
                 '<|The End of Assistant A\'s Description|>\n' \

    gt_key_words_llm = gpt_model.generate_llm_response(llm_system, llm_prompt, seed=17, temperature=0.2)

    pdb.set_trace()


def eval_text(gt_text, infer_text):

    gt_tokens = gt_text.split()
    infer_tokens = infer_text.split()
    m_score = meteor_score([gt_tokens], infer_tokens)
    # print(f"METEOR Score: {m_score}")

    P, R, F1 = score([infer_text], [gt_text], lang="en", model_type="bert-base-uncased")
    # print(f"BERTScore Precision: {P.mean().item()}")
    # print(f"BERTScore Recall: {R.mean().item()}")
    # print(f"BERTScore F1: {F1.mean().item()}")
    b_scores = {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }


    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r_scores = r_scorer.score(gt_text, infer_text)
    # print("ROUGE-1: ", r_scores['rouge1'])
    # print("ROUGE-2: ", r_scores['rouge2'])
    # print("ROUGE-L: ", r_scores['rougeL'])

    return m_score, b_scores, r_scores

def replace_words_to_words(sentence, from_list, to_list):
    for from_word, to_word in zip(from_list, to_list):
        sentence = sentence.replace(from_word, to_word)

    return sentence

def extract_nouns(doc):
    """
    문장에서 핵심 명사(존재하는 물체의 이름)을 모두 추출하는 함수
    """
    nouns = [token for token in doc if token.pos_ == 'NOUN' and not token.is_stop]
    return nouns

def calculate_max_similarity(nouns1, nouns2):
    """
    두 문장에서 추출한 명사 쌍들 사이의 최대 유사도를 계산하는 함수
    """
    max_similarity = 0.0
    
    # 가능한 모든 명사 쌍을 비교
    for noun1, noun2 in product(nouns1, nouns2):
        similarity = noun1.similarity(noun2)
        if similarity > max_similarity:
            max_similarity = similarity
    
    return max_similarity

def evaluate_xmls(pred_xml_list, gt_dir):
    

    error_file_list = []    # error_file_list

    # gt tag setup
    eval_gt_degree = 'simple'  # or 'complex'
    eval_gt_tags = ['dest', 'left', 'right', 'path']
    for i, tag in enumerate(eval_gt_tags):
        eval_gt_tags[i] = './/' + tag + '/' + eval_gt_degree
    eval_gt_tags.append('.//recommend')

    # pred tag setup
    eval_infer_tags = ['dest_desc', 'left_desc', 'right_desc', 'path_desc', 'recommend']
    llm_targets = ['destination', 'left', 'right', 'path', 'recommend']

    tag_scores = {}
    avg_scores = {
        'm_scores': {tag: 0.0 for tag in eval_gt_tags},
        'b_scores': {tag: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0} for tag in eval_gt_tags},
        'r_scores': {tag: {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0} for tag in eval_gt_tags},
    }
    eval_counts = {tag: 0 for tag in eval_gt_tags}

    for pred_xml_idx, pred_xml_path in enumerate(pred_xml_list):
        xml_filename = os.path.basename(pred_xml_path)
        tag_scores[xml_filename] = {
            'm_scores': {tag: [] for tag in eval_gt_tags},
            'b_scores': {tag: {'precision': [], 'recall': [], 'f1': []} for tag in eval_gt_tags},
            'r_scores': {tag: {'rouge1': [], 'rouge2': [], 'rougeL': []} for tag in eval_gt_tags},
            'gt': {tag: {'raw': [], 'llm': []} for tag in eval_gt_tags},
            'output': {tag: {'raw': [], 'llm': []} for tag in eval_gt_tags}
        }

        gt_xml_path = os.path.join(gt_dir, xml_filename)


        # 파싱 가능한 XML로 변경된 내용을 출력
        output_root = ET.parse(pred_xml_path).getroot()

        gt_tree = ET.parse(gt_xml_path)
        gt_root = gt_tree.getroot()

        print(f"Intermediate {xml_filename} results after {pred_xml_idx + 1}/{len(pred_xml_list)} files:")
        for i, gt_tag in enumerate(eval_gt_tags):
            pred_tag = eval_infer_tags[i]
            print(f"Tag: {gt_tag}, {pred_tag}")

            output_desc = output_root.find(pred_tag)
            if output_desc == None:
                error_file_list.append(f'{xml_filename}, {pred_tag} in pred')
                continue
            else:
                output_desc = output_desc.text.strip()

            try:
                gt_dest_desc = gt_root.find(gt_tag).text.strip()
                gt_dest_desc = replace_words_to_words(gt_dest_desc, ['India', '"', "'"], ['sidewalk', '', ""])
            except Exception as e:
                error_file_list.append(f'{xml_filename}, {gt_tag} in gt')
                continue

            # parsing word and get the result
            print(f"gt_dest_desc 1: {gt_dest_desc}")
            print(f"output_desc 2: {output_desc}")

            llm_system = ''

            print(llm_targets[i])
            if llm_targets[i] == 'recommend':
                # f'According to the following sentence, can the user walk? Should the user stop? Choose between "walk" and "stop": {gt_dest_desc}'
                llm_prompt_gt =     f'Description: {gt_dest_desc}\nAccording to the following sentence, can the user walk? Should the user stop? Choose between "walk" and "stop".  Say only a word. '
                llm_prompt_output = f'Description: {output_desc}\nAccording to the following sentence, can the user walk? Should the user stop? Choose between "walk" and "stop".  Say only a word. '
            elif llm_targets[i] == 'destination':
                # f'According to the following sentence, what is on the {llm_targets[i]}? If it is obscured or covered, what is it? Answer with a word: {gt_dest_desc}'
                llm_prompt_gt =     f'Description: {gt_dest_desc}\nAccording to the description, where is the destination? Say only words. If it is obscured or covered, say the objects instead of the destination. '
                llm_prompt_output = f'Description: {output_desc}\nAccording to the description, where is the destination? Say only words. If it is obscured or covered, say the objects instead of the destination. '
            else:
                # f'According to the following sentence, what is on the {llm_targets[i]}? Answer with a word: {gt_dest_desc}'
                llm_prompt_gt =     f'Description: {gt_dest_desc}\nAccording to the description, what objects are on the {llm_targets[i]} other than the floor? Say only words. '
                llm_prompt_output = f'Description: {output_desc}\nAccording to the description, what objects are on the {llm_targets[i]} other than the floor? Say only words. '

            gt_key_words_llm = gpt_model.generate_llm_response(llm_system, llm_prompt_gt, seed=17, temperature=0.2)
            output_key_words_llm = gpt_model.generate_llm_response(llm_system, llm_prompt_output, seed=17, temperature=0.2)

            gt_key_words_llm = replace_words_to_words(gt_key_words_llm, ['.', '"', "'"], ['', '', ""])
            gt_key_words_llm = gt_key_words_llm.lower()

            output_key_words_llm = replace_words_to_words(output_key_words_llm, ['.', '"', "'"], ['', '', ""])
            output_key_words_llm = output_key_words_llm.lower()

            print(f"gt_key_words_llm 1: {gt_key_words_llm}")
            print(f"output_key_words_llm 2: {output_key_words_llm}")

            m_score, b_scores, r_scores = eval_text(gt_key_words_llm, output_key_words_llm)
            # llm_score = eval_text_llm_judge(gt_key_words_llm, output_key_words_llm)

            eval_counts[gt_tag] += 1

            tag_scores[xml_filename]['gt'][gt_tag]['raw'].append(gt_dest_desc)
            tag_scores[xml_filename]['gt'][gt_tag]['llm'].append(gt_key_words_llm)
            tag_scores[xml_filename]['output'][gt_tag]['raw'].append(output_desc)
            tag_scores[xml_filename]['output'][gt_tag]['llm'].append(output_key_words_llm)

            tag_scores[xml_filename]['m_scores'][gt_tag].append(m_score)
            tag_scores[xml_filename]['b_scores'][gt_tag]['precision'].append(b_scores['precision'])
            tag_scores[xml_filename]['b_scores'][gt_tag]['recall'].append(b_scores['recall'])
            tag_scores[xml_filename]['b_scores'][gt_tag]['f1'].append(b_scores['f1'])
            tag_scores[xml_filename]['r_scores'][gt_tag]['rouge1'].append(r_scores['rouge1'].fmeasure)
            tag_scores[xml_filename]['r_scores'][gt_tag]['rouge2'].append(r_scores['rouge2'].fmeasure)
            tag_scores[xml_filename]['r_scores'][gt_tag]['rougeL'].append(r_scores['rougeL'].fmeasure)

            # Update running averages            
            avg_scores['m_scores'][gt_tag] += (m_score - avg_scores['m_scores'][gt_tag]) / eval_counts[gt_tag]
            avg_scores['b_scores'][gt_tag]['precision'] += (b_scores['precision'] - avg_scores['b_scores'][gt_tag]['precision']) / eval_counts[gt_tag]
            avg_scores['b_scores'][gt_tag]['recall'] += (b_scores['recall'] - avg_scores['b_scores'][gt_tag]['recall']) / eval_counts[gt_tag]
            avg_scores['b_scores'][gt_tag]['f1'] += (b_scores['f1'] - avg_scores['b_scores'][gt_tag]['f1']) / eval_counts[gt_tag]
            avg_scores['r_scores'][gt_tag]['rouge1'] += (r_scores['rouge1'].fmeasure - avg_scores['r_scores'][gt_tag]['rouge1']) / eval_counts[gt_tag]
            avg_scores['r_scores'][gt_tag]['rouge2'] += (r_scores['rouge2'].fmeasure - avg_scores['r_scores'][gt_tag]['rouge2']) / eval_counts[gt_tag]
            avg_scores['r_scores'][gt_tag]['rougeL'] += (r_scores['rougeL'].fmeasure - avg_scores['r_scores'][gt_tag]['rougeL']) / eval_counts[gt_tag]

            print(f"  GT  : {gt_dest_desc}")
            print(f"  Pred: {output_desc}")
            print(f"  METEOR Score: {m_score} (Avg: {avg_scores['m_scores'][gt_tag]})")
            print(f"  BERTScore Precision: {b_scores['precision']} (Avg: {avg_scores['b_scores'][gt_tag]['precision']})")
            print(f"  BERTScore Recall: {b_scores['recall']} (Avg: {avg_scores['b_scores'][gt_tag]['recall']})")
            print(f"  BERTScore F1: {b_scores['f1']} (Avg: {avg_scores['b_scores'][gt_tag]['f1']})")
            print(f"  ROUGE-1: {r_scores['rouge1'].fmeasure} (Avg: {avg_scores['r_scores'][gt_tag]['rouge1']})")
            print(f"  ROUGE-2: {r_scores['rouge2'].fmeasure} (Avg: {avg_scores['r_scores'][gt_tag]['rouge2']})")
            print(f"  ROUGE-L: {r_scores['rougeL'].fmeasure} (Avg: {avg_scores['r_scores'][gt_tag]['rougeL']})")

        print("\n")

    return tag_scores, avg_scores, error_file_list, eval_counts


def evaluate_folders(pred_db_dir, gt_db_dir, output_dir):
    # data setting
    pred_xml_list = glob.glob(f'{pred_db_dir}/**/*xml', recursive=True)
    pred_xml_list = sorted(pred_xml_list)

    tag_scores, avg_scores, error_file_list, eval_counts = evaluate_xmls(pred_xml_list, gt_db_dir)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the output file name
    # model_ckpt_name = os.path.basename(args.model_ckpt_path)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Write the result to the text file
    output_file_name = f"avg_scores_{current_time}.txt"
    output_file_path = os.path.join(output_dir, output_file_name)
    with open(output_file_path, 'w') as f:
        f.write(json.dumps({"avg_scores": avg_scores,
                            "eval_counts": eval_counts
                            }, indent=4))

    output_file_name = f"tag_scores_{current_time}.txt"
    output_file_path = os.path.join(output_dir, output_file_name)
    with open(output_file_path, 'w') as f:
        f.write(json.dumps({"tag_scores": tag_scores}, indent=4))

    error_file_path = os.path.join(output_dir, 'error_file_list.txt')
    with open(error_file_path, 'w') as fid:
        for item in error_file_list:
            fid.write(item + '\n')
    

    print(f"Average scores and tag scores saved to {output_file_path}")

def main():
    args = parse_args()

    with open(args.config) as fid:
        conf = yaml.load(fid, Loader=yaml.FullLoader)

    pred_db_dir = os.path.join(conf['output_dir'], conf['task_name'], 'qa')
    eval_db_dir = os.path.join(conf['db']['base_dir'], conf['db']['gt'])
    output_dir = os.path.join(conf['output_dir'], conf['task_name'], 'eval_llm')

    evaluate_folders(pred_db_dir, eval_db_dir, output_dir)


if __name__ == "__main__":
    main()

    # gt_desc = 'There is a wire fence on the right.'
    # pred_desc = 'There is a chain-link fence on the right side of the path.'
    # m_score, b_scores, r_scores = eval_text(gt_desc, pred_desc)

    # pdb.set_trace()

    # task_name = 'Pipeline4_Test40'

    # pred_db_dir = f'/home/yochin/Desktop/PathGuidedVQA_Base/output/2024.08.20/{task_name}/qa'
    # eval_db_dir = f'/home/yochin/Desktop/PathGuidedVQA_Base/test1k_0826/organized/desc_gt_en'
    # output_dir = f'/home/yochin/Desktop/PathGuidedVQA_Base/output/2024.08.20/{task_name}/qa/eval_further'

    # evaluate_folders(pred_db_dir, eval_db_dir, output_dir)

