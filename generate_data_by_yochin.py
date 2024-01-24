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

     parser.add_argument(
         '--llava-model-dir', metavar='DIRECTORY', required=True,
         help='directory for LLaVa checkpoints ')
     
     parser.add_argument(
         '--output', metavar='DIRECTORY', required=True,
         help='directory for output ')

     return parser.parse_args()


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
    llava_model_path = args.llava_model_dir
    output_path = args.output

    lvm = LargeMultimodalModels('llava', llava_model_path)


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

    # 0. Definition
    # list_goal_names = ['stairs', 'door', 'elevator']
    list_goal_names = read_text(label_path_gt)
    print('list_goal_names: ', list_goal_names)

    # 각 이미지 파일에 대해
    for img_file in image_files:
        # 이미지 파일의 전체 경로
        img_path = os.path.join(image_path, img_file)
        print(f'\nprocessing {img_path}...')
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
        bboxes2 = read_anno(xml_path2, rescaling=True, filtering_score=0.7)
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
                
        list_descriptions = []

        # 1.3. split images into sub-images
        num_cropped_image = 3
        for i_gp, goal_label_cxcy in enumerate(list_labels_gps):
            print('the goal info:', goal_label_cxcy)
            goal_label, goal_cxcy = goal_label_cxcy
            list_subimage_boxes_on_path, list_subimage_centerpoints_on_path, list_cropped_images = split_images(goal_cxcy, 1.0, 1.0, pil_image=img, sub_image_ratio=0.5, num_divisions=(num_cropped_image-2))
        
            for i_sub, (subimage_boxes, subimage_centerpoint, pil_sub_image) in enumerate(zip(list_subimage_boxes_on_path, list_subimage_centerpoints_on_path, list_cropped_images)):
                # 1.4. remove outer bbox
                thresh_intersect_over_bbox = 0.5
                inner_bboxes_original = remove_outer_bbox(bboxes, subimage_boxes, thresh_intersect_over_bbox)
                inner_bboxes_reorigin, goal_cxcy_reorigin = reorigin_bbox_point(inner_bboxes_original, goal_label_cxcy[1], subimage_boxes)
                
                # 2. generate answers 1 and 2 using LLM
                # 결과 문장 생성
                goal_label_cxcy_clamp = [goal_label_cxcy[0], goal_cxcy_reorigin]
                
                sub_img_path_temp = os.path.join(output_path_subimage, f'{img_file_wo_ext}_{i_gp}_{i_sub}.jpg')
                pil_sub_image.save(sub_img_path_temp)

                # description = f'hellow world, this is {img_file}, {i_gp}, {i_sub}'
                # description = describe_all_bboxes_with_chatgpt(img_path, inner_bboxes_reorigin, goal_label_cxcy_clamp)
                # description = describe_all_bboxes_with_llava(llava_model_path, img_path, bboxes, goal_label_cxcy_clamp)
                description = lvm.describe_images_with_boxes(img_path, bboxes, goal_label_cxcy_clamp, order=(i_sub+1), num_total=num_cropped_image, 
                                                             merge=False, previous_descriptions=[])
                list_descriptions.append(description)


                # 4.1. draw all in sub image
                sub_img = Image.open(sub_img_path_temp)
                draw = ImageDraw.Draw(sub_img)
                font = ImageFont.truetype('arial.ttf', size=40)
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

                path_to_debug = os.path.join(output_path_debug, f'{img_file_wo_ext}_{i_gp}_sub_{i_sub}.jpg')
                sub_img.save(path_to_debug)


                # 4.2. draw all in original image
                # draw start, mid, and goal points and boxes
                img = Image.open(img_path)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype('arial.ttf', size=40)
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
            final_description = lvm.describe_images_with_boxes(img_path, bboxes, goal_label_cxcy_clamp, order=num_cropped_image, num_total=num_cropped_image, 
                                                             merge=True, previous_descriptions=list_descriptions)

            output_dict = {
                'image_filename': img_file,
                'goal_position_xy': goal_cxcy,
                'goal_object_label': goal_label,
                'answer1': list_descriptions[0],
                'answer2': list_descriptions[1],
                'answer3': list_descriptions[2],
                'answer': final_description
            }
            xml_all_info = dict_to_xml(output_dict, 'Annotation')
            save_xml(xml_all_info, os.path.join(output_path_qa, img_file_wo_ext + '.xml'))
    
    return

if __name__ == '__main__':
    main()