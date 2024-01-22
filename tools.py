import os
import random
import shutil
import xml.etree.ElementTree as ET

# Read text, return list
# Coded by ChatGPT 4
def read_text(file_path):
    lines = []

    # 파일을 열고 각 줄을 읽어 리스트에 추가합니다
    with open(file_path, 'r') as file:
        for line in file:
            # strip() 함수를 사용하여 줄바꿈 문자를 제거합니다
            lines.append(line.strip())

    return lines


def dict_to_xml(input_dict, root_tag):
    root = ET.Element(root_tag)
    for key, value in input_dict.items():
        child = ET.SubElement(root, key)
        child.text = str(value)
    return root


def save_xml(xml_element, filename):
    tree = ET.ElementTree(xml_element)
    tree.write(filename, encoding='utf-8', xml_declaration=True)



def select_and_save_files(src_file, dest_folder, list_file, num_files):
    # 특정 파일 목록 읽기
    files = read_text(src_file)
    
    # N개의 파일을 랜덤하게 선택
    selected_files = random.sample(files, min(num_files, len(files)))

    # 선택된 파일 목록을 파일로 저장
    with open(list_file, 'w') as f:
        for file in selected_files:
            f.write(file + '\n')


def example():
    # 사용 예시
    path_to_list = '/home/yochin/Desktop/GLIP/odinw/naeultech_231109_at_server/organized/ImageSets/list_train.txt'  # 원본 list 파일 경로
    dest_folder = '/home/yochin/Desktop/PathGuidedVQA/sample100'  # 대상 폴더 경로
    list_file = '/home/yochin/Desktop/PathGuidedVQA/sample100/list_selected_files.txt'  # 선택된 파일 목록을 저장할 파일
    num_files = 100  # 선택할 파일의 개수

    select_and_save_files(path_to_list, dest_folder, list_file, num_files)

    # 저장된 파일 목록을 읽어와 지정된 폴더로 복사
    with open(list_file, 'r') as f:
        for line in f:
            file = line.strip()

            # image
            shutil.copy(os.path.join('/home/yochin/Desktop/GLIP/OUTPUT_gd/toomuch_labels/resized_image', file + '.jpg'), 
                        os.path.join(dest_folder, 'images', file + '.jpg'))

            # anno_aihub
            shutil.copy(os.path.join('/home/yochin/Desktop/GLIP/OUTPUT_gd/aihub_labels/pred_pascal', file + '.xml'), 
                        os.path.join(dest_folder, 'anno_aihub', file + '.xml'))

            # anno_gt
            shutil.copy(os.path.join('/home/yochin/Desktop/GLIP/OUTPUT_gd/toomuch_labels/pred_pascal', file + '.xml'), 
                        os.path.join(dest_folder, 'anno_toomuch', file + '.xml'))

            # anno_toomuch
            shutil.copy(os.path.join('/home/yochin/Desktop/GLIP/odinw/naeultech_231109_at_server/organized/Annotations', file + '.xml'), 
                        os.path.join(dest_folder, 'anno_gt', file + '.xml'))
