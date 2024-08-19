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
         '--model-name', default = None, required=True,
         help='model name')

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
     
     parser.add_argument(
         '--ex-dir', metavar='DIRECTORY', default=None,
         help='directory which contains examples')
     
     parser.add_argument(
         '--prompt-id', default=18, type=int)

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


def load_examples(path_to_base):
    path_to_images = os.path.join(path_to_base, 'images')
    path_to_prompt = os.path.join(path_to_base, 'prompt')


    files = os.listdir(path_to_images)

    # 이미지 파일들만 필터링
    list_images = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
    list_images.sort()  # return itself

    list_prompt_ex = []
    for filename in list_images:
        filename_txt = os.path.splitext(filename)[0] + '.txt'
        answer_prompt = read_text(os.path.join(path_to_prompt, filename_txt))

        long_prompt = []

        for i_th in range(0, len(answer_prompt), 2):
            long_prompt.append(answer_prompt[i_th])

        list_prompt_ex.append(long_prompt)

            # f'{role_user}{answer_prompt[0]}\n<image-placeholder>\n{role_asst}{answer_prompt[2]}\n' 
            # f'{role_user}{answer_prompt[4]}\n{role_asst}{answer_prompt[6]}\n'
            # f'{role_user}{answer_prompt[8]}\n{role_asst}{answer_prompt[10]}\n'
            # f'{role_user}{answer_prompt[12]}\n{role_asst}{answer_prompt[14]}')

    list_images = [os.path.join(path_to_images, f) for f in list_images]

    return list_images, list_prompt_ex


import json
def get_points_array(image_path, depth_width, depth_height):
    json_path = image_path.replace('.jpg', '.json')

    with open(json_path, 'r') as fid:
        data = json.load(fid)

    if 'No' in data['result']:
        ret = False
        path_arr_xy = [[0, 0]]
    else:
        ret = True
        path_arr_yx = data['path']
        path_arr_xy = [[item[1]/depth_width, item[0]/depth_height] for item in path_arr_yx]

    return ret, path_arr_xy


def generate_points(image_path):
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    # 이미지의 가로 중간 지점
    x_center = image_width // 2
    
    # 세로는 위에서 2/3 지점
    y_start = (2 * image_height) // 3
    
    # 포인트 배열 생성: 시작점부터 이미지 하단까지
    points_xy = [[x_center/image_width, y/image_height] for y in range(y_start, image_height)]
    
    return points_xy


import cv2
import numpy as np
def create_thick_line_mask(image, points, thickness):
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Convert points list to numpy array
    points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    
    # Draw thick line on the mask
    cv2.polylines(mask, [points], isClosed=False, color=(255, 255, 255), thickness=thickness)
    
    return mask

def create_circular_mask(image, points, radius):
    # Initialize mask with zeros (black)
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Draw white filled circles on the mask at each point
    for point in points:
        cv2.circle(mask, center=tuple(point), radius=radius, color=(255, 255, 255), thickness=-1)
    
    return mask


def gen_mask_v1(cv_img, path_array_xy):
    img_h, img_w, img_c = cv_img.shape

    mask_left = np.zeros_like(cv_img, dtype=np.uint8)
    mask_right = np.zeros_like(cv_img, dtype=np.uint8)
    
    # Convert points list to numpy array
    points = np.array(path_array_xy, dtype=np.int32)
    
    # Split points into two parts: above and below the line
    for i in range(1, len(points)):
        pts = np.array([points[i-1], points[i], [0, img_h], [0, 0]])
        cv2.fillPoly(mask_left, [pts], (255, 255, 255))
        
        pts = np.array([points[i-1], points[i], [img_w, 0], [img_w, img_h]])
        cv2.fillPoly(mask_right, [pts], (255, 255, 255))

    return mask_left, mask_right
    

def gen_mask_v2(image, points):
    mask_left = np.zeros_like(image, dtype=np.uint8)
    mask_right = np.zeros_like(image, dtype=np.uint8)
    
    # Convert points list to numpy array
    points = np.array(points, dtype=np.int32)
    
    # Create a thick line mask
    line_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.polylines(line_mask, [points], isClosed=False, color=(255, 255, 255), thickness=10)
    
    # Create left and right masks by flood filling from the borders
    flood_filled_left = line_mask.copy()
    flood_filled_right = line_mask.copy()
    
    # Flood fill from the left and right of the image to create two separate regions
    cv2.floodFill(flood_filled_left, None, (0, 0), (255, 255, 255))
    cv2.floodFill(flood_filled_right, None, (image.shape[1] - 1, 0), (255, 255, 255))
    
    # Left side mask is where flood_filled_left is white and line_mask is not white
    mask_left[(flood_filled_left == 255) & (line_mask == 0)] = (255, 255, 255)
    
    # Right side mask is where flood_filled_right is white and line_mask is not white
    mask_right[(flood_filled_right == 255) & (line_mask == 0)] = (255, 255, 255)
    
    return mask_left, mask_right


def generate_mask(img_path, path_array_xy_norm, line_thickness=50):
    cv_img = cv2.imread(img_path)

    img_h, img_w, img_c = cv_img.shape
    path_array_xy = [[x*img_w, y*img_h] for x, y in path_array_xy_norm]
    points = np.array(path_array_xy, dtype=np.int32)

    mask_left, mask_right = gen_mask_v1(cv_img, path_array_xy)

    mask_path = create_thick_line_mask(cv_img, points, thickness=line_thickness)

    return {'L': mask_left, 
            'R': mask_right, 
            'P': mask_path}

def create_trapezoid_mask(img_path, target_point, r):
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    x, y = target_point

    # Define the trapezoid points
    top_left = (0, 0)
    top_right = (width, 0)
    bottom_left = (max(x - r, 0), y)
    bottom_right = (min(x + r, width), y)
    
    # Create a mask
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Define the trapezoid as a polygon
    trapezoid = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    
    # Fill the trapezoid area in the mask
    cv2.fillPoly(mask, [trapezoid], (255, 255, 255))
        
    return mask
    

def save_debug_image(img_path, dict_masks):
    debug_img_path = img_path.replace('images', 'debug_images')

    debug_img_folder, file_with_ext = os.path.split(debug_img_path)
    filename, ext = os.path.splitext(file_with_ext)

    cv_img = cv2.imread(img_path)

    if not os.path.exists(debug_img_folder):
        os.makedirs(debug_img_folder)

    for key, mask in dict_masks.items():
        path_to_save = os.path.join(debug_img_folder, f'{filename}_{key}{ext}')
        debug_image = cv2.bitwise_and(cv_img, mask)
        cv2.imwrite(path_to_save, debug_image)


def create_depth_mask(depth_image, target_point, radius, buffer_dist):
    x, y = target_point
    height, width = depth_image.shape
    
    # Ensure the radius is within the image bounds
    x_min = max(x - radius, 0)
    x_max = min(x + radius, width)
    y_min = max(y - radius, 0)
    y_max = min(y + radius, height)
    
    # Extract the region of interest
    roi = depth_image[y_min:y_max, x_min:x_max]
    
    # Calculate the average depth in the region of interest
    avg_depth = np.max(roi)
    thresh_depth = avg_depth + buffer_dist
    
    # Create a mask where depth is greater than the average depth
    mask = np.zeros_like(depth_image, dtype=np.uint8)
    mask[depth_image < thresh_depth] = 255
    
    return mask


# Assisted by ChatGPT 4
def main():
    args = parse_args()

    # 이미지가 저장된 폴더 경로
    image_path = os.path.join(args.db_dir, 'images')

    anno_path1 = os.path.join(args.db_dir, 'det_anno_pred')
    # anno_path2 = os.path.join(args.db_dir, 'det_anno_toomuch')
    anno_path2 = None
    anno_path_gt = os.path.join(args.db_dir, 'det_anno_gt')

    label_path_removal = os.path.join(args.db_dir, 'removal_labels.txt')    # if you want to remove some classes in the det_anno.
    
    label_path_gp = os.path.join(args.db_dir, 'gp_labels.txt')
    anno_path_gp = os.path.join(args.db_dir, 'anno')    # annotation has GP in txt file

    use_gp = 'anno'
    if not os.path.exists(label_path_gp):
        assert os.path.exists(anno_path_gp)
        use_gp = 'anno'

    if not os.path.exists(anno_path_gp):
        assert os.path.exists(label_path_gp)
        use_gp = 'det'
    

    llava_model_base_path = args.llava_model_base_dir
    llava_model_path = args.llava_model_dir
    output_path = args.output

    # option: treat point as bbox
    return_as_bbox = True

    # option: detection info
    use_det_info = True

    # option: give examples
    if args.ex_dir is not None:
        example_path = os.path.join(args.ex_dir)
        print(f'@main - example_path is given: {example_path}')

        list_ex_images, list_ex_prompt = load_examples(example_path)
    else:
        list_ex_images = [] 
        list_ex_prompt = []

    print('@main - prompt_id: ', args.prompt_id)
    lvm = LargeMultimodalModels(args.model_name, llava_model_base_path=llava_model_base_path, llava_model_path=llava_model_path)

    choose_one_random_gp = True     # select one random gp when many gps are detected

    assert choose_one_random_gp     # treat the output filename related to several gps

    # related to output
    output_path_qa = os.path.join(output_path, 'qa')
    output_path_debug = os.path.join(output_path, 'debug')

    if not os.path.exists(output_path_qa):
        os.makedirs(output_path_qa)

    if not os.path.exists(output_path_debug):
        os.makedirs(output_path_debug)
            
    # 폴더 내의 모든 파일 목록을 가져옴
    files = os.listdir(image_path)

    # 이미지 파일들만 필터링
    image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
    image_files.sort()  # return itself

    # 0. Definition    
    if use_gp == 'det':
        list_goal_names = read_text(label_path_gp)
        print('@main - list_goal_names: ', list_goal_names)
    
    # 각 이미지 파일에 대해
    for img_file in image_files:
        # 이미지 파일의 전체 경로
        img_path = os.path.join(image_path, img_file)
        img_file_wo_ext = os.path.splitext(img_file)[0]
        print('==============================\n')
        print(f'@main - processing {img_path}...')
        # XML 파일의 전체 경로 (파일 이름은 같지만 확장자만 xml로 변경)
        xml_path1 = os.path.join(anno_path1, img_file_wo_ext + '.xml')
        if anno_path2 is not None:
            xml_path2 = os.path.join(anno_path2, img_file_wo_ext + '.xml')
        xml_path_gt = os.path.join(anno_path_gt, img_file_wo_ext + '.xml')
     
        # 이미지를 열고
        img = Image.open(img_path)
        whole_width, whole_height = img.size

        # 1. read annotation and convert into bboxes with label info.
        # XML 파일을 파싱하여 Bounding Box 정보를 가져옴 (if rescaling to 0 ~ 1)
        bboxes1 = read_anno(xml_path1, rescaling=True, filtering_score=0.7) # list of [label_name, [x_min, y_min, x_max, y_max], score]
        bboxes = bboxes1

        if anno_path2 is not None:
            bboxes2 = read_anno(xml_path2, rescaling=True, filtering_score=0.8)
            bboxes.extend(bboxes2)

        bboxes_gt = read_anno(xml_path_gt, rescaling=True)
        bboxes.extend(bboxes_gt)

        # removal specific classes
        list_labels_removal = read_text(label_path_removal)
        bboxes = [item for item in bboxes if item[0] not in list_labels_removal]

        # 2. set goal position
        if use_gp == 'det':
            list_labels_gps = get_gp(bboxes, list_goal_names, return_as_bbox=return_as_bbox)  # list of [label_name, [cx, cy]]

            if choose_one_random_gp:
                list_labels_gps = [random.choice(list_labels_gps)]
        else:
            with open(os.path.join(anno_path_gp, img_file_wo_ext + '.txt')) as fid:
                lines = fid.read().splitlines()
            cx, cy = lines[0].split(' ')
            cx = float(cx) / whole_width
            cy = float(cy) / whole_height
            list_labels_gps = [['point', [cx, cy]]]
                
        list_queries = []
        list_descriptions = []

        # 3. for each gp in one image
        for i_gp, goal_label_cxcy in enumerate(list_labels_gps):
            print('@main - the goal info:', goal_label_cxcy)

            goal_label, goal_cxcy = goal_label_cxcy

            if use_det_info is False:
                bboxes = []

            
            if len(list_ex_images) > 0:
                list_img_path = [item for item in list_ex_images]
                list_img_path.append(img_path)

                list_example_prompt = list_ex_prompt
            else:
                list_img_path = [img_path]
                list_example_prompt = []

            final_query, final_description = lvm.describe_whole_images_with_boxes(list_img_path, bboxes, goal_label_cxcy, 
                                                                                    step_by_step=True, 
                                                                                    list_example_prompt=list_example_prompt,
                                                                                    prompt_id=args.prompt_id)

            output_dict = {
                'image_filename': img_file,
                'goal_position_xy': goal_cxcy,
                'goal_object_label': goal_label,
                'final_query': final_query,
                'answer': final_description
            }

            output_dict['bboxes'] = bboxes

            xml_all_info = dict_to_xml(output_dict, 'Annotation')
            save_xml(xml_all_info, os.path.join(output_path_qa, img_file_wo_ext + '.xml'))


            # 4.3. draw all whole original image
            # draw start, mid, and goal points and boxes
            img = Image.open(img_path)
            # img_note = Image.new('RGB', (whole_width, int(whole_height*1.5)), color='white')    # note at bottom
            img_note = Image.new('RGB', (int(whole_width*2), int(whole_height)), color='white')    # note at right
            img_note.paste(img, (0, 0))

            draw = ImageDraw.Draw(img_note)
            font = ImageFont.truetype('arial.ttf', size=20)
            radius = 20
            
            # for mid_box, mid_point in zip(list_subimage_boxes_on_path, list_subimage_centerpoints_on_path):
            mid_point_draw = [goal_cxcy[0] * whole_width, goal_cxcy[1] * whole_height]
            mid_point_lu_draw = (int(mid_point_draw[0]-radius), int(mid_point_draw[1]-radius))
            mid_point_rd_draw = (int(mid_point_draw[0]+radius), int(mid_point_draw[1]+radius))
            draw.ellipse([mid_point_lu_draw, mid_point_rd_draw], fill='orange')

            # draw detection results
            for label_name, bbox, score in bboxes:
                # Bounding Box를 이미지에 그림
                draw_bbox = [bbox[0] * whole_width, bbox[1] * whole_height, bbox[2] * whole_width, bbox[3] * whole_height]
                draw.rectangle(draw_bbox, outline='yellow', width=2)
                draw.text(draw_bbox[:2], label_name, fill='white', font=font)
                
            # draw.text((10, whole_height-60), f'q:{i_sub_query}', fill='red', font=font)
            # draw.text((10, whole_height-30), f'a:{final_description}', fill='blue', font=font)
            # draw_long_text(draw, position=(10, whole_height+10), text=f'a:{final_description}', fill='blue', font=font, max_width=whole_width-20) # note at bottom
            draw_long_text(draw, position=(whole_width+10, 10), text=f'a:{final_description}', fill='blue', font=font, max_width=whole_width-20)   # note at right
            draw_long_text(draw, position=(whole_width+10, int(whole_height/2) + 10), text=f'q:{final_query}', fill='green', font=font, max_width=whole_width-20)   # note at right
            
            path_to_debug = os.path.join(output_path_debug, f'{img_file_wo_ext}_{i_gp}_whole_final.jpg')
            img_note.save(path_to_debug)
    return

if __name__ == '__main__':
    main()