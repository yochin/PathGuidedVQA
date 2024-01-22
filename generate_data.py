from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pdb
import os
import random

from setGP import read_anno, get_gp, split_images, remove_outer_bbox
from tools import read_text
from vlm_description import describe_all_bboxes_with_chatgpt


def dict_to_xml(input_dict, root_tag):
    root = ET.Element(root_tag)
    for key, value in input_dict.items():
        child = ET.SubElement(root, key)
        child.text = str(value)
    return root

def save_xml(xml_element, filename):
    tree = ET.ElementTree(xml_element)
    tree.write(filename, encoding='utf-8', xml_declaration=True)
 

# Assisted by ChatGPT 4
def main():
    

    # 이미지가 저장된 폴더 경로
    image_path = '/mnt/exp_disk/dbs/gd_datagen/sample100/images'
    anno_path1 = '/mnt/exp_disk/dbs/gd_datagen/sample100/anno_aihub'
    anno_path2 = '/mnt/exp_disk/dbs/gd_datagen/sample100/anno_toomuch'
    anno_path_gt = '/mnt/exp_disk/dbs/gd_datagen/anno_gt'
    label_path_gt = '/mnt/exp_disk/dbs/gd_datagen/sample100/default_labels.txt'
    label_path_removal = '/mnt/exp_disk/dbs/gd_datagen/removal_labels.txt'
    llava_model_path = '/mnt/data_disk/models/vlms/llava/llava-v1.5-7b'

    choose_one_random_gp = True     # select one random gp when many gps are detected

    assert choose_one_random_gp     # treat the output filename related to several gps

    # related to output
    output_path = 'output'
    output_path_qa = os.path.join(output_path, 'qa_unified')
    output_path_debug = os.path.join(output_path, 'debug')

    if not os.path.exists(output_path_qa):
        os.makedirs(output_path_qa)

    if not os.path.exists(output_path_debug):
        os.makedirs(output_path_debug)


    # 폴더 내의 모든 파일 목록을 가져옴
    files = os.listdir(image_path)

    # 이미지 파일들만 필터링
    image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 0. Definition
    # list_goal_names = ['stairs', 'door', 'elevator']
    list_goal_names = read_text(label_path_gt)
    print('list_goal_names: ', list_goal_names)

    # 각 이미지 파일에 대해
    for img_file in image_files:
        # 이미지 파일의 전체 경로
        img_path = os.path.join(image_path, img_file)
        print(f'processing {img_path}...\n')
        # XML 파일의 전체 경로 (파일 이름은 같지만 확장자만 xml로 변경)
        xml_path1 = os.path.join(anno_path1, os.path.splitext(img_file)[0] + '.xml')
        xml_path2 = os.path.join(anno_path2, os.path.splitext(img_file)[0] + '.xml')
        xml_path_gt = os.path.join(anno_path_gt, os.path.splitext(img_file)[0] + '.xml')

        img_file_wo_ext = os.path.splitext(img_file)[0]        

        # 이미지를 열고
        img = Image.open(img_path)
        # draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype('arial.ttf', size=40)

        # # XML 파일을 파싱하여 Bounding Box 정보를 가져옴
        # bboxes1 = read_anno(xml_path1) # list of [label_name, [x_min, y_min, x_max, y_max], score]
        # bboxes2 = read_anno(xml_path2)
        # bboxes = bboxes1
        # bboxes.extend(bboxes2)

        # for label_name, bbox, score in bboxes:
        #     # Bounding Box를 이미지에 그림
        #     draw.rectangle(bbox, outline='yellow', width=2)
        #     draw.text(bbox[:2], label_name, fill='white', font=font)

        # # 이미지 및 Bounding Box 표시
        # plt.imshow(img)
        # plt.axis('off')
        # plt.title(img_file)
        # plt.show()

        whole_width, whole_height = img.size

        # 1. Split input images into cropped images along with the goal path (yochin)
        # 1.1. read annotation and convert into bboxes with label info.
        # XML 파일을 파싱하여 Bounding Box 정보를 가져옴 (if rescaling to 0 ~ 1)
        bboxes1 = read_anno(xml_path1, rescaling=True, filtering_score=0.7) # list of [label_name, [x_min, y_min, x_max, y_max], score]
        bboxes2 = read_anno(xml_path2, rescaling=True, filtering_score=0.8)
        bboxes = bboxes1
        bboxes.extend(bboxes2)

        bboxes_gt = read_anno(xml_path_gt, rescaling=True)
        bboxes.extend(bboxes_gt)

        # removal specific classes
        list_labels_removal = read_text(label_path_removal)
        bboxes = [item for item in bboxes if item[0] not in list_labels_removal]

        # 1.2. set goal position
        list_labels_gps = get_gp(bboxes, list_goal_names)  # list of [label_name, [cx, cy]]
        if choose_one_random_gp:
            list_labels_gps = [random.choice(list_labels_gps)]

        # 1.3. split images into sub-images
        for goal_label_cxcy in list_labels_gps:
            print('the goal info:', goal_label_cxcy)
            goal_label, goal_cxcy = goal_label_cxcy
        
            #description = describe_all_bboxes_with_chatgpt(img_path, bboxes, goal_label_cxcy)
            description = describe_all_bboxes_with_llava(llava_model_path, image_path, bboxes, goal_label_cxcy)

            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype('arial.ttf', size=40)
            radius = 20

            for label_name, bbox, score in bboxes:
                # Bounding Box를 이미지에 그림
                draw_bbox = [bbox[0] * whole_width, bbox[1] * whole_height, bbox[2] * whole_width, bbox[3] * whole_height]
                draw.rectangle(draw_bbox, outline='blue', width=2)
                draw.text(draw_bbox[:2], label_name, fill='white', font=font)

            goal_point_draw = [goal_cxcy[0] * whole_width, goal_cxcy[1] * whole_height]
            goal_point_lu_draw = (int(goal_point_draw[0]-radius), int(goal_point_draw[1]-radius))
            goal_point_rd_draw = (int(goal_point_draw[0]+radius), int(goal_point_draw[1]+radius))

            draw.ellipse([goal_point_lu_draw, goal_point_rd_draw], fill='red')

            ## 이미지 및 Bounding Box 표시
            plt.imshow(img)
            plt.axis('off')
            plt.title(img_file + f', goal_label:{goal_label}')
            plt.show()

            final_answer = description

            output_dict = {
                'image_filename': img_file,
                'goal_position_xy': goal_cxcy,
                'goal_object_label': goal_label,
                'answer': final_answer
            }
            xml_all_info = dict_to_xml(output_dict, 'Annotation')
            save_xml(xml_all_info, os.path.join(output_path_qa, img_file_wo_ext + '.xml'))


    return

if __name__ == '__main__':
    main()