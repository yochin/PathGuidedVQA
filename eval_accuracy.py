import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import json
import os
import pdb
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from gensim.models import KeyedVectors
from scipy import spatial
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()   

    parser.add_argument(
         '--pred-dir', metavar='DIRECTORY', required=True,
         help='directory which contains predicted results')
     
    parser.add_argument(
         '--gt-dir', metavar='DIRECTORY', required=True,
         help='directory which contains ground-truth')
     
    return parser.parse_args()
     

# 유사도 계산 함수 정의
def get_word_vector(word, model):
    try:
        return model[word]
    except KeyError:
        return np.zeros(model.vector_size)
    
def calculate_similarity_between_phrases(phrase1, phrase2, model):
    words1 = phrase1.split()
    words2 = phrase2.split()

    vector1 = np.mean([get_word_vector(word, model) for word in words1], axis=0)
    vector2 = np.mean([get_word_vector(word, model) for word in words2], axis=0)

    similarity = 1 - spatial.distance.cosine(vector1, vector2)
    return similarity

# Precision과 Recall 계산 함수 수정
def calculate_confusion_matrix_with_similarity(predicted, ground_truth, model, similarity_threshold=0.7):
    TP = 0
    all_matches = []
    
    for pred in predicted:
        for gt in ground_truth:
            # 유사도 계산을 위해 수정된 부분
            similarity = calculate_similarity_between_phrases(pred, gt, model)
            if similarity >= similarity_threshold:
                TP += 1
                all_matches.append((pred, gt))
                break  # 첫 번째 동의어/유사 단어를 찾으면 중지
    
    # 예측된 항목 중 실제 항목과 일치하거나 유사한 항목을 제외
    FP = len(predicted) - len(all_matches)
    # 실제 항목 중 예측에 실패한 항목 계산
    FN = len(ground_truth) - len(all_matches)

    return TP, FP, FN

# import spacy

# # spaCy 모델 로드
# nlp = spacy.load("en_core_web_sm")


# # 명사 및 복합 명사 추출 함수
# def extract_nouns_spacy(sentences):
#     extracted_nouns = []
#     for sentence in sentences:
#         doc = nlp(sentence)
#         for chunk in doc.noun_chunks:
#             extracted_nouns.append(chunk.text)
#     return extracted_nouns


def extract_nouns_nltk(word):
    """주어진 단어에서 명사를 추출"""
    nouns = []
    for token, pos in pos_tag(word_tokenize(word)):
        if pos.startswith('NN'):  # 명사의 POS 태그는 'NN'으로 시작합니다.
            nouns.append(token)
    return nouns

def check_synonym(word1, word2):
    """두 단어가 동의어인지 체크"""
    word1_synsets = wn.synsets(word1)
    word2_synsets = wn.synsets(word2)
    for synset1 in word1_synsets:
        for synset2 in word2_synsets:
            if synset1 == synset2:
                return True  # 동의어를 찾았습니다.
    return False  # 동의어가 없습니다.

def calculate_confusion_matrix_with_synonyms(predicted, ground_truth):
    TP = 0
    for pred in predicted:
        # 예측된 각각의 항목에 대해 실제 값 리스트 내 동의어가 있는지 체크
        if pred in ground_truth or any(check_synonym(pred, gt) for gt in ground_truth):
            TP += 1
            
    FP = len(predicted) - TP
    FN = len([gt for gt in ground_truth if not any(gt == pred or check_synonym(gt, pred) for pred in predicted)])

    return TP, FP, FN

# In gt, not find -> false negative
# In gt, find -> true positive
# Not in gt, find -> false positive
# Not in gt, not find -> true negative


# def calculate_confusion_matrix(predicted, ground_truth):
#     # True Positive (TP): 예측과 실제 값이 모두 참인 경우
#     TP = len(set(predicted) & set(ground_truth))
#     # False Positive (FP): 예측은 참이지만 실제 값은 거짓인 경우
#     FP = len(set(predicted) - set(ground_truth))
#     # False Negative (FN): 실제 값은 참이지만 예측은 거짓인 경우
#     FN = len(set(ground_truth) - set(predicted))

#     return TP, FP, FN


def calculate_precision_recall(TP, FP, FN):
    # Precision 계산
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    # Recall 계산
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    # F1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1



def read_gt(path_gt_file):
    with open(path_gt_file, 'r') as fid:
        lines = fid.read().splitlines()

        # - GP (x_pixel, y_pixel) (좌상단이 0,0)
        # - go -45, go 0, go 45, stop (-45가 10시방향, 45가 2시방향), 편의상 straight를 0으로 바꾸었습니다.
        # - 1) 경로상 존재하는 장애물의 list (name은 aihub, gd에서 활용)
        # - 2) 경로상 존재한는 장애물의 list (aihub/gd X)
        # - 3) 경로상 존재하는 활용물의 list (name은 aihub, gd에서 활용)
        # - 4) 경로상 존재하는 활용물의 list (aihub/gd X)
        cx, cy = lines[0].split(' ')

        act = lines[1]

        if len(lines) > 2:
            obs_in_list = lines[2].split(',')
            obs_in_list = [item.strip() for item in obs_in_list]
        else:
            obs_in_list = []

        if len(lines) > 3:
            obs_out_list = lines[3].split(',')
            obs_out_list = [item.strip() for item in obs_out_list]
        else:
            obs_out_list = []
        
        if len(lines) > 4:
            util_in_list = lines[4].split(',')
            util_in_list = [item.strip() for item in util_in_list]
        else:
            util_in_list = []

        if len(lines) > 5:
            util_out_list = lines[5].split(',')
            util_out_list = [item.strip() for item in util_out_list]
        else:
            util_out_list = []

    res = {
        'gp': [cx, cy],
        'act': act,
        'obs_in_list': obs_in_list,
        'obs_out_list': obs_out_list,
        'util_in_list': util_in_list,
        'util_out_list': util_out_list
    }

    return res
     

def main():
    args = parse_args()

    path_to_pred_base = args.pred_dir
    path_to_gt = args.gt_dir

    path_to_pred_action = os.path.join(path_to_pred_base, 'eval/pred_action_llava')
    path_to_pred_obs = os.path.join(path_to_pred_base, 'eval/pred_obs')

    use_w2v = True
    remove_dup_obstacle = True

    list_action_gt = []
    list_action_pred = []

    list_tp = []
    list_tn = []
    list_fp = []
    list_fn = []

    list_filename = []

    if use_w2v:
        # 모델 로딩 예시 (실제 경로를 지정해야 함)
        w2v_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

    # read the list of answer files
    files = os.listdir(path_to_gt)
    anno_files = [f for f in files if f.endswith(('.txt'))]
    anno_files = sorted(anno_files)

    for filename in anno_files:
        path_gt_file = os.path.join(path_to_gt, filename)
        dict_gt = read_gt(path_gt_file)

        # about action
        gt_act_index = ['go 0', 'go -45', 'go 45', 'stop'].index(dict_gt['act'].strip())

        with open(os.path.join(path_to_pred_action, filename), 'r') as fid:
            lines_action = fid.read().splitlines()
        str_pred = lines_action[0].lower() 

        # predicted as: A) Go straight, B) Go left 45, C) Go right 45, D) Stop.
        # labeled gt as: 'go 0', 'go -45', 'go 45', 'stop'

        pred_act_index = -1
        for ith, item in enumerate(['straight', 'left', 'right', 'stop']):
            if item in str_pred:
                pred_act_index = ith
        
        if pred_act_index == -1:
            for ith, item in enumerate(['a)', 'b)', 'c)', 'd)']):
                if item in str_pred:
                    pred_act_index = ith

        if pred_act_index == -1:
            print(dict_gt['act'])
            print(lines_action[0])
            pdb.set_trace()
            pred_act_index = 4

        list_action_gt.append(gt_act_index)
        list_action_pred.append(pred_act_index)

        # about obstacles
        gt_total_obs = dict_gt['obs_out_list'] + dict_gt['obs_in_list']

        with open(os.path.join(path_to_pred_obs, filename), 'r') as fid:
            lines_obs = fid.read().splitlines()
        assert len(lines_obs) == 1
        list_obs_pred = lines_obs[0].split(',')
        list_obs_pred = [item.lower().strip() for item in list_obs_pred]

        # no obstacles -> ""
        list_obs_pred = [item.replace("no obstacles", "") for item in list_obs_pred]

        # remove "" items
        list_removal_word = ['']
        for rm_word in list_removal_word:
            while rm_word in list_obs_pred:  
                list_obs_pred.remove(rm_word)

            while rm_word in gt_total_obs:  
                gt_total_obs.remove(rm_word)

        # extract nouns
        list_obs_nouns_gt = []
        for item in gt_total_obs:
            extracted_noun = extract_nouns_nltk(item)

            if len(extracted_noun) > 0:
                list_obs_nouns_gt.append(extracted_noun[-1])
        
        list_obs_nouns_pred = []
        for item in list_obs_pred:
            extracted_noun = extract_nouns_nltk(item)

            if len(extracted_noun) > 0:
                list_obs_nouns_pred.append(extracted_noun[-1])

        list_removal_word = ['[', ']']
        for rm_word in list_removal_word:
            while rm_word in list_obs_nouns_pred:  
                list_obs_nouns_pred.remove(rm_word)

            while rm_word in list_obs_nouns_gt:  
                list_obs_nouns_gt.remove(rm_word)

        if remove_dup_obstacle:
            list_obs_nouns_pred = list(set(list_obs_nouns_pred))
            list_obs_nouns_gt = list(set(list_obs_nouns_gt))

        if use_w2v:
            TP, FP, FN = calculate_confusion_matrix_with_similarity(predicted=list_obs_nouns_pred, ground_truth=list_obs_nouns_gt, model=w2v_model)
        else:
            TP, FP, FN = calculate_confusion_matrix_with_synonyms(predicted=list_obs_nouns_pred, ground_truth=list_obs_nouns_gt)
        list_tp.append(TP)
        list_fp.append(FP)
        list_fn.append(FN)

        print('\n')
        print(filename)
        print(list_obs_nouns_gt)
        print(list_obs_nouns_pred)
        print(TP, FP, FN)

        list_filename.append(filename)
    
    precision, recall, f1 = calculate_precision_recall(sum(list_tp), sum(list_fp), sum(list_fn))
    print(f1, precision, recall)

    # 일치하는 요소의 개수 계산
    correct_predictions = sum(t == p for t, p in zip(list_action_gt, list_action_pred))
    # 정확도 계산
    accuracy = correct_predictions / len(list_action_gt)
    print('accuracy: ', accuracy)


if __name__ == '__main__':
    main()