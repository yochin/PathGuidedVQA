import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pdb
import os
import random

from setGP import read_anno, get_gp, split_images, remove_outer_bbox, clamp, reorigin_bbox_point
from tools import read_text, dict_to_xml, save_xml
from vlm_description_for_multi_images import LargeMultimodalModels
import argparse

def parse_args():

     parser = argparse.ArgumentParser(
        description='Generate QA (Questions and Answers) for Path Guided VQA')   

     parser.add_argument(
         '--db-dir', metavar='DIRECTORY', required=True,
         help='directory which contains images and object properties')

     # lora 사용시에는 llava-model-base-dir이 있어야함
     # ex) llava-v1.5-7b-lora 사용시 base model은 llava-v1.5-7b, model은 llava-v1.5-7b-lora
     parser.add_argument(
         '--llava-model-base-dir', default = None, metavar='DIRECTORY', 
         help='directory for LLaVa checkpoints ')

     parser.add_argument(
         '--llava-model-dir', metavar='DIRECTORY', required=True,
         help='directory for LLaVa checkpoints ')
     
     parser.add_argument(
         '--output', metavar='DIRECTORY', required=True,
         help='directory for output ')

     return parser.parse_args()


def draw_long_text(draw, position=(0,0), text='blank', fill='white', font=None, max_width=100):
    """
    이미지에 텍스트를 주어진 폭 안에서 자동으로 줄바꿈하여 그립니다.

    Args:
    draw (ImageDraw): PIL ImageDraw 객체
    text (str): 그릴 텍스트
    position (tuple): 텍스트 시작 위치 (x, y)
    font (ImageFont): 사용할 폰트
    max_width (int): 최대 폭
    """
    words = text.split()
    lines = []
    current_line = words[0]

    for word in words[1:]:
        # 현재 줄에 단어를 추가했을 때의 길이
        test_line = current_line + ' ' + word
        # 폰트와 텍스트 크기를 고려하여 텍스트 크기 측정
        # width, _ = draw.textsize(test_line, font=font)
        # width, _ = font.getsize(test_line)
        # width = font.getlength(test_line)
        tx1, ty1, tx2, ty2 = font.getbbox(test_line)
        width = tx2 - tx1
        if width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)

    y = position[1]
    for line in lines:
        draw.text((position[0], y), line, font=font, fill=fill)
        # y += font.getlength(line)
        tx1, ty1, tx2, ty2 = font.getbbox(line)
        y += (ty2 - ty1)


# Assisted by ChatGPT 4
def main():
    args = parse_args()

    # 이미지가 저장된 폴더 경로
    image_path = os.path.join(args.db_dir, 'images')
    anno_path1 = os.path.join(args.db_dir, 'anno_aihub')
    anno_path2 = os.path.join(args.db_dir, 'anno_toomuch')
    anno_path_gt = os.path.join(args.db_dir, 'anno_gt')
    label_path_gt = os.path.join(args.db_dir, 'default_labels.txt')
    label_path_removal = os.path.join(args.db_dir, 'removal_labels.txt')
    llava_model_base_path = args.llava_model_base_dir
    llava_model_path = args.llava_model_dir
    output_path = args.output

    # option: crop image into sub-images
    num_cropped_image = 1

    # option: treat point as bbox
    return_as_bbox = True

    # option: detection info
    use_det_info = True

    # 

    if return_as_bbox:
        assert num_cropped_image == 1

    lvm = LargeMultimodalModels('llava', llava_model_base_path=llava_model_base_path, llava_model_path=llava_model_path)


    choose_one_random_gp = True     # select one random gp when many gps are detected

    assert choose_one_random_gp     # treat the output filename related to several gps

    # related to output
    output_path_subimage = os.path.join(output_path, 'sub_images')
    output_path_qa = os.path.join(output_path, 'qa')
    output_path_debug = os.path.join(output_path, 'debug')

    if not os.path.exists(output_path_subimage):
        os.makedirs(output_path_subimage)

    if not os.path.exists(output_path_qa):
        os.makedirs(output_path_qa)

    if not os.path.exists(output_path_debug):
        os.makedirs(output_path_debug)
            
    # 폴더 내의 모든 파일 목록을 가져옴
    files = os.listdir(image_path)

    # 이미지 파일들만 필터링
    image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
    sorted(image_files)

    # 0. Definition
    # list_goal_names = ['stairs', 'door', 'elevator']
    list_goal_names = read_text(label_path_gt)
    print('list_goal_names: ', list_goal_names)

    # 각 이미지 파일에 대해
    for img_file in image_files:
        # 이미지 파일의 전체 경로
        img_path = os.path.join(image_path, img_file)
        print('==============================\n')
        print(f'processing {img_path}...')
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
        list_labels_gps = get_gp(bboxes, list_goal_names, return_as_bbox=return_as_bbox)  # list of [label_name, [cx, cy]]

        if choose_one_random_gp:
            list_labels_gps = [random.choice(list_labels_gps)]
                
        list_queries = []
        list_descriptions = []

        # 1.3. split images into sub-images
        for i_gp, goal_label_cxcy in enumerate(list_labels_gps):
            print('the goal info:', goal_label_cxcy)

            goal_label, goal_cxcy = goal_label_cxcy

            if num_cropped_image > 1:
                list_subimage_boxes_on_path, list_subimage_centerpoints_on_path, list_cropped_images = split_images(goal_cxcy, 1.0, 1.0, pil_image=img, sub_image_ratio=0.5, num_divisions=(num_cropped_image-2))

                for i_sub, (subimage_boxes, subimage_centerpoint, pil_sub_image) in enumerate(zip(list_subimage_boxes_on_path, list_subimage_centerpoints_on_path, list_cropped_images)):
                    # 1.4. remove outer bbox
                    thresh_intersect_over_bbox = 0.5
                    inner_bboxes_original = remove_outer_bbox(bboxes, subimage_boxes, thresh_intersect_over_bbox)
                    inner_bboxes_reorigin, goal_cxcy_reorigin = reorigin_bbox_point(inner_bboxes_original, goal_label_cxcy[1], subimage_boxes)
                    
                    # 2. generate answers 1 and 2 using LLM
                    # 결과 문장 생성
                    goal_label_cxcy_clamp_reorigin = [goal_label_cxcy[0], goal_cxcy_reorigin]
                    
                    sub_img_path_temp = os.path.join(output_path_subimage, f'{img_file_wo_ext}_{i_gp}_{i_sub}.jpg')
                    pil_sub_image.save(sub_img_path_temp)

                    # description = f'hellow world, this is {img_file}, {i_gp}, {i_sub}'
                    # description = describe_all_bboxes_with_chatgpt(img_path, inner_bboxes_reorigin, goal_label_cxcy_clamp)
                    # description = describe_all_bboxes_with_llava(llava_model_path, img_path, bboxes, goal_label_cxcy_clamp)
                    i_sub_query, description = lvm.describe_images_with_boxes([sub_img_path_temp], inner_bboxes_reorigin, goal_label_cxcy_clamp_reorigin, 
                                                                order=(i_sub+1), num_total=num_cropped_image, 
                                                                merge=False, previous_descriptions=[])
                    list_queries.append(i_sub_query)
                    list_descriptions.append(description)


                    # 4.1. draw all sub images
                    sub_img = Image.open(sub_img_path_temp)
                    draw = ImageDraw.Draw(sub_img)
                    font = ImageFont.truetype('arial.ttf', size=15)
                    radius = 20

                    # for mid_box, mid_point in zip(list_subimage_boxes_on_path, list_subimage_centerpoints_on_path):
                    mid_box = inner_bboxes_reorigin
                    mid_point = goal_cxcy_reorigin
                    sub_width, sub_height = sub_img.size
                    mid_point_draw = [mid_point[0] * sub_width, mid_point[1] * sub_height]

                    mid_point_lu_draw = (int(mid_point_draw[0]-radius), int(mid_point_draw[1]-radius))
                    mid_point_rd_draw = (int(mid_point_draw[0]+radius), int(mid_point_draw[1]+radius))
                    draw.ellipse([mid_point_lu_draw, mid_point_rd_draw], fill='blue')

                    # draw detection results
                    for label_name, bbox, score in inner_bboxes_reorigin:
                        # Bounding Box를 이미지에 그림
                        draw_bbox = [bbox[0] * sub_width, bbox[1] * sub_height, bbox[2] * sub_width, bbox[3] * sub_height]
                        draw.rectangle(draw_bbox, outline='yellow', width=2)
                        draw.text(draw_bbox[:2], label_name, fill='white', font=font)

                    # # 이미지 및 Bounding Box 표시
                    # plt.imshow(sub_img)
                    # plt.axis('off')
                    # plt.title(img_file + f', goal_label:{goal_label}')
                    # plt.show()
                        
                    # draw.text((10, sub_height-60), f'q:{i_sub_query}', fill='red', font=font)
                    draw.text((10, sub_height-30), f'a:{description}', fill='blue', font=font)

                    path_to_debug = os.path.join(output_path_debug, f'{img_file_wo_ext}_{i_gp}_sub_{i_sub}.jpg')
                    sub_img.save(path_to_debug)


                    # 4.2. draw all whole original image
                    # draw start, mid, and goal points and boxes
                    img = Image.open(img_path)
                    draw = ImageDraw.Draw(img)
                    font = ImageFont.truetype('arial.ttf', size=20)
                    radius = 20
                    
                    # for mid_box, mid_point in zip(list_subimage_boxes_on_path, list_subimage_centerpoints_on_path):
                    mid_box = subimage_boxes
                    mid_point = subimage_centerpoint
                    mid_point_draw = [mid_point[0] * whole_width, mid_point[1] * whole_height]
                    mid_box_draw = [mid_box[0] * whole_width, mid_box[1] * whole_height, mid_box[2] * whole_width, mid_box[3] * whole_height]

                    mid_point_lu_draw = (int(mid_point_draw[0]-radius), int(mid_point_draw[1]-radius))
                    mid_point_rd_draw = (int(mid_point_draw[0]+radius), int(mid_point_draw[1]+radius))
                    draw.ellipse([mid_point_lu_draw, mid_point_rd_draw], fill='red')
                    draw.rectangle(mid_box_draw, outline='red', width=4)

                    # draw detection results
                    for label_name, bbox, score in bboxes:
                        # Bounding Box를 이미지에 그림
                        draw_bbox = [bbox[0] * whole_width, bbox[1] * whole_height, bbox[2] * whole_width, bbox[3] * whole_height]
                        draw.rectangle(draw_bbox, outline='yellow', width=2)
                        draw.text(draw_bbox[:2], label_name, fill='white', font=font)

                    # draw detection results
                    for label_name, bbox, score in inner_bboxes_original:
                        # Bounding Box를 이미지에 그림
                        draw_bbox = [bbox[0] * whole_width, bbox[1] * whole_height, bbox[2] * whole_width, bbox[3] * whole_height]
                        draw.rectangle(draw_bbox, outline='blue', width=2)
                        draw.text(draw_bbox[:2], label_name, fill='white', font=font)

                    # # 이미지 및 Bounding Box 표시
                    # plt.imshow(img)
                    # plt.axis('off')
                    # plt.title(img_file + f', goal_label:{goal_label}')
                    # plt.show()

                    path_to_debug = os.path.join(output_path_debug, f'{img_file_wo_ext}_{i_gp}_whole_{i_sub}.jpg')
                    img.save(path_to_debug)

                    print(f'{i_sub}: {description}')


                # 3. merge answers into the final answer
                # final_answer = list_descriptions[0] + list_descriptions[1] + list_descriptions[2]
                # final_description = f'hellow world, this is {img_file}, {i_gp}, {i_sub}'
                # final_description = describe_all_bboxes_with_chatgpt(img_path, inner_bboxes_reorigin, goal_label_cxcy_clamp)
                # final_description = describe_all_bboxes_with_llava(llava_model_path, img_path, bboxes, goal_label_cxcy_clamp)
                final_query, final_description = lvm.describe_images_with_boxes([img_path], bboxes, goal_label_cxcy, order=num_cropped_image, num_total=num_cropped_image, 
                                                                merge=True, previous_descriptions=list_descriptions)
                
            else:
                if use_det_info is False:
                    bboxes = []

                final_query, final_description = lvm.describe_whole_images_with_boxes([img_path], bboxes, goal_label_cxcy, step_by_step=True)

            output_dict = {
                'image_filename': img_file,
                'goal_position_xy': goal_cxcy,
                'goal_object_label': goal_label,
                'final_query': final_query,
                'answer': final_description
            }

            output_dict['bboxes'] = bboxes

            # additional information
            if num_cropped_image > 1:
                output_dict_sub = {}
                for i_sub in range(len(list_descriptions)):
                    output_dict_sub[f'subimage-{i_sub}'] = list_subimage_boxes_on_path[i_sub]
                    output_dict_sub[f'query-{i_sub}'] = list_queries[i_sub]
                    output_dict_sub[f'answer-{i_sub}'] = list_descriptions[i_sub]
                output_dict.update(output_dict_sub)

            xml_all_info = dict_to_xml(output_dict, 'Annotation')
            save_xml(xml_all_info, os.path.join(output_path_qa, img_file_wo_ext + '.xml'))


            # 4.3. draw all whole original image
            # draw start, mid, and goal points and boxes
            img = Image.open(img_path)
            img_note = Image.new('RGB', (whole_width, int(whole_height*1.5)), color='white')
            img_note.paste(img, (0, 0))

            draw = ImageDraw.Draw(img_note)
            font = ImageFont.truetype('arial.ttf', size=20)
            radius = 20
            
            # for mid_box, mid_point in zip(list_subimage_boxes_on_path, list_subimage_centerpoints_on_path):
            mid_point_draw = [goal_cxcy[0] * whole_width, goal_cxcy[1] * whole_height]
            mid_point_lu_draw = (int(mid_point_draw[0]-radius), int(mid_point_draw[1]-radius))
            mid_point_rd_draw = (int(mid_point_draw[0]+radius), int(mid_point_draw[1]+radius))
            draw.ellipse([mid_point_lu_draw, mid_point_rd_draw], fill='red')

            # draw detection results
            for label_name, bbox, score in bboxes:
                # Bounding Box를 이미지에 그림
                draw_bbox = [bbox[0] * whole_width, bbox[1] * whole_height, bbox[2] * whole_width, bbox[3] * whole_height]
                draw.rectangle(draw_bbox, outline='yellow', width=2)
                draw.text(draw_bbox[:2], label_name, fill='white', font=font)
                

            # draw.text((10, whole_height-60), f'q:{i_sub_query}', fill='red', font=font)
            # draw.text((10, whole_height-30), f'a:{final_description}', fill='blue', font=font)
            draw_long_text(draw, position=(10, whole_height+10), text=f'a:{final_description}', fill='blue', font=font, max_width=whole_width-20)
            

            path_to_debug = os.path.join(output_path_debug, f'{img_file_wo_ext}_{i_gp}_whole_final.jpg')
            img_note.save(path_to_debug)
    return

if __name__ == '__main__':
    main()