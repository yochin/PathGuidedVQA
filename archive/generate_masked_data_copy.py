import os
import sys
import argparse
import yaml
import logging
import pdb

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import random

from setGP import read_anno, get_gp, split_images, remove_outer_bbox, clamp, reorigin_bbox_point
from tools import read_text, dict_to_xml, save_xml, save_json
from vlm_description_for_multi_images import LargeMultimodalModels
from depth_anything_wrapper import depth_anything
from ultralytics import YOLO
import torch
from generate_image_data import depth_lpp, draw_path_on_scaled_image, save_and_visualize_depth_map
from prompt_library import get_prompt
from llm_wrapper import llm_wrapper
from gpt_wrapper import gpt_wrapper
import pickle
import random
import time
from utils.xml_to_json import convert_xml_files
import cv2
import numpy as np
from masking_utils import generate_mask, get_intrinsic_ratio, create_circle_mask, create_depth_mask
from masking_utils import mask_dest_depth_lr, mask_depth_image_using_path, save_debug_masked_image, is_within_mask
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Generate QA (Questions and Answers) for Path Guided VQA')   

    parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()

    return args

def set_path_logger(path_to_log):
    if not os.path.exists(os.path.split(path_to_log)[0]):
        os.makedirs(os.path.split(path_to_log)[0])

    logFormatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s', datefmt='%Y/%m/%d %p %I:%M:%S, ')
    logFileHandler = logging.FileHandler(path_to_log)
    logConsoleHandler = logging.StreamHandler(sys.stdout)

    logFileHandler.setFormatter(logFormatter)
    logConsoleHandler.setFormatter(logFormatter)

    logging.getLogger().addHandler(logFileHandler)
    logging.getLogger().addHandler(logConsoleHandler)
    logging.getLogger().setLevel(logging.DEBUG)

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

def get_points_array(json_path, depth_width, depth_height):
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


def generate_points(img_wh):
    image_width, image_height = img_wh

    # 이미지의 가로 중간 지점
    x_center = image_width // 2
    
    # 세로는 위에서 2/3 지점
    y_start = (2 * image_height) // 3
    
    # 포인트 배열 생성: 시작점부터 이미지 하단까지
    points_xy = [[x_center/image_width, y/image_height] for y in range(y_start, image_height)]
    
    return points_xy

def interpolate_points(x1, y1, x2, y2, n):
    # x1, y1에서 x2, y2까지 n개의 점을 생성
    x_values = np.linspace(x1, x2, n)
    y_values = np.linspace(y1, y2, n)
    # 각 x값과 y값을 쌍으로 묶어 배열 형태로 반환
    points = np.column_stack((x_values, y_values))

    return points.tolist()

def expand_points(points, n):
    if len(points) >= n:
        return points  # 이미 n개 이상의 점이 있으면 그대로 반환

    # 결과 리스트 초기화
    result = []
    
    # 입력 리스트에서 순차적으로 점들을 가져와 보간 수행
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        
        # 목표하는 총 점의 수에 따라 현재 점 사이에 필요한 점의 수를 계산
        # 목표 점 수까지 남은 점의 수를 고려하여 보간 간격 결정
        points_left = n - len(result)
        segments_needed = len(points) - i
        points_in_segment = max(2, int(np.ceil(points_left / segments_needed)))
        
        # 선형 보간으로 점 생성
        x_values = np.linspace(x1, x2, points_in_segment, endpoint=False)
        y_values = np.linspace(y1, y2, points_in_segment, endpoint=False)
        
        # 생성된 점들을 결과 리스트에 추가 (마지막 점 제외)
        result.extend((x, y) for x, y in zip(x_values[:-1], y_values[:-1]))
    
    # 마지막 점 추가
    result.append(points[-1])
    
    # 결과 리스트가 여전히 n보다 작을 경우, 재귀적으로 처리
    if len(result) < n:
        return expand_points(result, n)
    
    return result

def find_depth_difference_point(depth_image, depth_difference=5.0, crop_width=10, set_lower_middle_as_zero=True, max_search_degree=0):
    # Get the dimensions of the depth image
    height, width = depth_image.shape
    
    # Calculate the center line (vertical center)
    center_line = width // 2
    
    start_x = max(center_line - crop_width // 2, 0)
    end_x = min(center_line + crop_width // 2 + 1, width)

    if set_lower_middle_as_zero:
        lower_middle_avg = 0.0
    else:
        # Crop a region around the lower middle of the image
        lower_middle_crop = depth_image[height-1-2*crop_width:height-1, start_x:end_x]
        
        # Calculate the average depth value of the cropped region
        lower_middle_avg = np.mean(lower_middle_crop)
    
    # Target depth value
    target_depth = lower_middle_avg + depth_difference

    
    if max_search_degree == 0:
        # Iterate through the center line from bottom to top to find the depth difference point
        for y in range(height-1, -1, -1):
            if abs(depth_image[y, center_line] - lower_middle_avg) >= depth_difference:
                return (y / height, center_line / width)
    else:
        # 랜덤 방향 선택 (-45도에서 45도 사이)
        np.random.seed(int(time.time()))
        angle = np.random.uniform(-max_search_degree, max_search_degree)        
        # random.seed(int(time.time()))
        # angle = random.randint(-max_search_degree, max_search_degree)
        angle_rad = np.radians(angle)
        
        logging.debug(f'>>>>> random angle: {angle} in max_search_degree +/-{max_search_degree}')

        # 방향에 따른 이동 벡터 계산
        dx = np.sin(angle_rad)
        dy = -np.cos(angle_rad)
    
        # 현재 위치를 이미지 하단 중앙으로 설정
        x, y = center_line, height - 1
        
        # 주어진 방향으로 이동하며 깊이 차이 지점을 찾음
        while 0 <= x < width and 0 <= y < height:
            if abs(depth_image[int(y), int(x)] - lower_middle_avg) >= depth_difference:
                return (y / height, x / width)
            x += dx
            y += dy
    
    # If no point is found, return None
    return None

def get_normalized_click(image):
    """
    Displays the given NumPy image, captures one click from the user, 
    and returns the normalized coordinates (x, y) relative to the image dimensions.
    
    Parameters:
    image (numpy.ndarray): The image to display.
    
    Returns:
    tuple: Normalized coordinates (x, y) of the click.
    """
    # Define the event handler for mouse click
    def onclick(event):
        nonlocal click_coords
        click_coords = (event.xdata, event.ydata)
        plt.close()  # Close the figure after one click

    click_coords = None
    
    # Display the image
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    # Wait for a click and return the normalized coordinates
    if click_coords is not None:
        x, y = click_coords
        height, width = image.shape[:2]
        normalized_x = x / width
        normalized_y = y / height
        return normalized_y, normalized_x
    else:
        raise ValueError("No click detected")


# Assisted by ChatGPT 4
def main():
    args = parse_args()

    with open(args.config) as fid:
        conf = yaml.load(fid, Loader=yaml.FullLoader)

    # set a log file path        
    path_to_log = os.path.join(conf['output_dir'], conf['task_name'], 'generate_masked_data.log')
    set_path_logger(path_to_log)

    logging.info('generate_masked_data.py: Start')
    logging.info(conf)

    # Set info from config file
    image_path = os.path.join(conf['db']['base_dir'], conf['db']['images'])
    depth_base = os.path.join(conf['db']['base_dir'], conf['db']['depths'])
    anno_path1 = os.path.join(conf['db']['base_dir'], conf['db']['det_anno_pred'])
    anno_path_gt = os.path.join(conf['db']['base_dir'], conf['db']['det_anno_gt'])
    anno_path_gp = os.path.join(conf['db']['base_dir'], conf['db']['gp_info']) # annotation has GP in txt file
    path_to_path = os.path.join(conf['db']['base_dir'], conf['db']['paths'])

    label_path_removal = os.path.join(conf['db']['base_dir'], 'removal_labels.txt')    # if you want to remove some classes in the det_anno.
    label_path_gp = os.path.join(conf['db']['base_dir'], 'gp_labels.txt')

    llava_model_base_path = conf['vlm']['llava_model_base_dir']
    llava_model_path = conf['vlm']['llava_model_dir']
    prompt_id = conf['prompt_id']
    vlm_model_name = conf['vlm']['model_name']

    yolo_dynamic_path = conf['yolo']['dynamic_det_path']
    yolo_static_path = conf['yolo']['static_det_path']

    output_path = os.path.join(conf['output_dir'], conf['task_name'])

    depth_anything_encoder = conf['depth_anything']['encoder']
    depth_anything_dataset = conf['depth_anything']['dataset']
    depth_anything_max_depth = conf['depth_anything']['max_depth']

    # gp, anno, det, set_forward
    gp_method = conf['gp']['method']
    gp_method_set_front_type = conf['gp']['method_set_front_type']
    max_search_degree = conf['gp']['max_search_degree']

    lpp_method = conf['local_path_planning']

    apply_cam_intrinsic = conf['apply_cam_intrinsic']

    llm_model_name = conf['llm']['model_name']
    gpt_model_name = conf['gpt']['model_name']

    use_llm_decision = conf['llm']['use_decision']
    use_llm_summary = conf['llm']['use_summary']

    use_gpt_decision = conf['gpt']['use_decision']
    use_gpt_summary = conf['gpt']['use_summary']

    np_rnd_seed = conf['gp']['np_rnd_seed']

    dst_masking_depth = conf['dest']['masking_depth']
    dst_depth_meter = conf['dest']['depth_meter']
    dst_draw_point = conf['dest']['draw_point']
    dst_draw_circle = conf['dest']['draw_circle']
    dst_masking_circle = conf['dest']['masking_circle']
    dst_draw_bbox = conf['dest']['draw_bbox']
    dst_circle_ratio = conf['dest']['circle_ratio_w']
    
    if gp_method == 'load_anno':
        assert os.path.exists(anno_path_gp)

    if gp_method == 'select_det':
        assert os.path.exists(label_path_gp)

        list_goal_names = read_text(label_path_gp)
        logging.info(f'@main - list_gp_det_names: {list_goal_names}')

  
    # related to output
    output_path_qa = os.path.join(output_path, 'qa')
    output_path_debug = os.path.join(output_path, 'debug')

    if not os.path.exists(output_path_qa):
        os.makedirs(output_path_qa)

    if not os.path.exists(output_path_debug):
        os.makedirs(output_path_debug)

    # option: detection info
    use_det_info = True

    # depth anything v2
    dep_any = depth_anything(depth_anything_encoder, 
                             depth_anything_dataset, 
                             depth_anything_max_depth)

    yolo_dynamic = YOLO(yolo_dynamic_path, task='detect')
    yolo_static = YOLO(yolo_static_path, task='detect')


    list_ex_images = [] 
    list_ex_prompt = []

    logging.info(f'@main - prompt_id: {prompt_id}')
    lvm = LargeMultimodalModels(vlm_model_name, llava_model_base_path=llava_model_base_path, llava_model_path=llava_model_path)

    if use_llm_decision or use_llm_summary:
        llm_model = llm_wrapper(llm_model_name)

    if use_gpt_decision or use_gpt_summary:
        gpt_model = gpt_wrapper(gpt_model_name)

    # 폴더 내의 모든 파일 목록을 가져옴
    files = os.listdir(image_path)

    # 이미지 파일들만 필터링
    image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
    image_files.sort()  # return itself

    
    # 각 이미지 파일에 대해
    for img_file in image_files:
        # 이미지 파일의 전체 경로
        img_path = os.path.join(image_path, img_file)
        img_file_wo_ext = os.path.splitext(img_file)[0]
        logging.info('==============================')
        logging.info(f'@main - processing {img_path}...')

        # XML 파일의 전체 경로 (파일 이름은 같지만 확장자만 xml로 변경)
        xml_path1 = os.path.join(anno_path1, img_file_wo_ext + '.xml')
        xml_path_gt = os.path.join(anno_path_gt, img_file_wo_ext + '.xml')
     
        # 이미지를 열고
        img = Image.open(img_path).convert('RGB')
        whole_width, whole_height = img.size

        cv_org_img = cv2.imread(img_path)

        depth_path = os.path.join(depth_base, img_file_wo_ext + '.npy')
        if os.path.exists(depth_path):
            depth_image = np.load(depth_path)       # h x w
        else:
            logging.info(f'{depth_path} is not exists. Generate and Save it.')
            depth_image = dep_any.infer_image(cv_org_img)
            np.save(depth_path, depth_image)
        
        depth_image = cv2.resize(depth_image, dsize=(whole_width, whole_height), interpolation=cv2.INTER_CUBIC)


        # 1. read annotation and convert into bboxes with label info.
        # XML 파일을 파싱하여 Bounding Box 정보를 가져옴 (if rescaling to 0 ~ 1)
        bboxes1 = read_anno(xml_path1, rescaling=True, filtering_score=0.7) # list of [label_name, [x_min, y_min, x_max, y_max], score]
        bboxes = bboxes1

        bboxes_gt = read_anno(xml_path_gt, rescaling=True)
        bboxes.extend(bboxes_gt)

        # add yolov8
        if not os.path.exists(xml_path1) or not os.path.exists(xml_path_gt):
            res_dynamic = yolo_dynamic(img, conf=0.8)
            
            for item in res_dynamic:
                for ith in range(len(item.boxes.cls)):
                    bbox_xyxy = item.boxes.xyxyn[ith, :].detach().cpu().numpy().tolist()
                    bbox_conf = item.boxes.conf[ith].detach().cpu().numpy().tolist()
                    bbox_cls = item.names[int(item.boxes.cls[ith])]

                    bboxes.append([bbox_cls, bbox_xyxy, bbox_conf])

            res_static = yolo_static(img, conf=0.8)


            for item in res_static:
                for ith in range(len(item.boxes.cls)):
                    bbox_xyxy = item.boxes.xyxyn[ith, :].detach().cpu().numpy().tolist()
                    bbox_conf = item.boxes.conf[ith].detach().cpu().numpy().tolist()
                    bbox_cls = item.names[int(item.boxes.cls[ith])]

                    bboxes.append([bbox_cls, bbox_xyxy, bbox_conf])


        # removal specific classes
        list_labels_removal = read_text(label_path_removal)
        bboxes = [item for item in bboxes if item[0] not in list_labels_removal]

        # 2. set goal position
        list_labels_gps = []
        if gp_method == 'select_det':
            list_labels_gps = get_gp(bboxes, list_goal_names, return_as_bbox=False)  # list of [label_name, [cx, cy]]

            if len(list_labels_gps) > 0:
                # select one random gp when many gps are detected
                list_labels_gps = [random.choice(list_labels_gps)]
                logging.info(f'Set GP {list_labels_gps[0][0]}, {list_labels_gps[0][1]} from detection bboxes')
            else:
                logging.info(f'No GPs from detection bboxes')

        elif gp_method == 'load_anno':
            path_to_gp = os.path.join(anno_path_gp, img_file_wo_ext + '.txt')
            if os.path.exists(path_to_gp):
                with open(path_to_gp) as fid:
                    lines = fid.read().splitlines()
                cx, cy = lines[0].split(' ')
                cx = float(cx) / whole_width
                cy = float(cy) / whole_height
                list_labels_gps = [['point', [cx, cy]]]
                logging.info(f'Set GP {list_labels_gps[0][0]}, {list_labels_gps[0][1]} from anno file, {path_to_gp}')
            else:
                logging.info(f'No GPs due to No anno file, {path_to_gp}')


        if len(list_labels_gps) == 0:
            res_gp = None

            path_to_gp = os.path.join(anno_path_gp, f'{img_file_wo_ext}_{gp_method_set_front_type}.pickle')


            if os.path.exists(path_to_gp):
                with open(path_to_gp, 'rb') as fid_gp:
                    res_gp = pickle.load(fid_gp)
            else:
                if gp_method_set_front_type == 'manual':
                    res_gp = get_normalized_click(cv_org_img)   # y and x
                    
                if gp_method_set_front_type == 'ten_meters_from_below_point':
                    res_gp = find_depth_difference_point(depth_image, depth_difference=10.0, 
                                                        crop_width=10, set_lower_middle_as_zero=False, max_search_degree=max_search_degree)
                if gp_method_set_front_type == 'ten_meters_from_depth':
                    res_gp = find_depth_difference_point(depth_image, depth_difference=10.0, 
                                                        crop_width=10, set_lower_middle_as_zero=True, max_search_degree=max_search_degree)
                    
                with open(path_to_gp, 'wb') as fid_gp:
                    pickle.dump(res_gp, fid_gp)

            if res_gp is None or gp_method_set_front_type == 'three_quarters':
                cx = 0.5
                cy = 0.75
                list_labels_gps = [['point', [cx, cy]]]
                logging.info(f'Set GP {list_labels_gps[0][0]}, {list_labels_gps[0][1]} using gp_method, three_quarters')
            else:
                cy, cx = res_gp

                if dst_draw_circle:
                    pt_name = 'circle'
                else:
                    pt_name = 'point'
                list_labels_gps = [[pt_name, [cx, cy]]]
                logging.info(f'Set GP {list_labels_gps[0][0]}, {list_labels_gps[0][1]} using gp_method, {gp_method}')

        list_queries = []
        list_descriptions = []

        # 3. for each gp in one image
        for i_gp, goal_label_cxcy in enumerate(list_labels_gps):
            logging.info(f'@main - the goal info: {goal_label_cxcy[0]}, {goal_label_cxcy[1]}')

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

            # draw point
            cv_org_img_pt = cv_org_img.copy()
            mid_point_draw = (int(goal_cxcy[0] * whole_width), int(goal_cxcy[1] * whole_height))
            cv2.circle(cv_org_img_pt, mid_point_draw, radius=20, color=(0, 165, 255), thickness=-1)
            # cv2.circle(cv_org_img_pt, (int(whole_width/2), whole_height-20), radius=20, color=(0, 255, 255), thickness=-1)

            if dst_draw_bbox:
                # draw bbox
                cv_org_img_bbox = cv_org_img.copy()
                mid_point_draw_tl = (int(goal_cxcy[0] * whole_width) - int(whole_width*dst_circle_ratio), int(goal_cxcy[1] * whole_height) - int(whole_width*dst_circle_ratio))
                mid_point_draw_br = (int(goal_cxcy[0] * whole_width) + int(whole_width*dst_circle_ratio), int(goal_cxcy[1] * whole_height) + int(whole_width*dst_circle_ratio))
                cv2.rectangle(cv_org_img_bbox, mid_point_draw_tl, mid_point_draw_br, color=(0, 165, 255), thickness=20)
                # cv2.circle(cv_org_img_bbox, (int(whole_width/2), whole_height-20), radius=20, color=(0, 255, 255), thickness=-1)

            if dst_draw_circle:
                # draw circle
                cv_org_img_circle = cv_org_img.copy()
                mid_point_draw = (int(goal_cxcy[0] * whole_width), int(goal_cxcy[1] * whole_height))
                cv2.circle(cv_org_img_circle, mid_point_draw, radius=int(whole_width*dst_circle_ratio), color=(0, 165, 255), thickness=20)
                # cv2.circle(cv_org_img_circle, (int(whole_width/2), whole_height-20), radius=20, color=(0, 255, 255), thickness=-1)

            # read point array from file
            path_suc = False
            if lpp_method == 'depth_lpp':
                json_path = os.path.join(path_to_path, img_file_wo_ext + '.json')

                if os.path.exists(json_path):
                    pdb.set_trace() # depth_with / height right?
                    path_suc, path_array = get_points_array(json_path, depth_width=518, depth_height=392)
                else:
                    res_dict_path = depth_lpp(depth_image, goal_cxcy)    # depth_sized_path is returned

                    # # Save to dict
                    # res_dict = {
                    #     'res': path_res,                    # Boolean: True or False
                    #     'reason': path_str,                 # String: Success or Failure with a reason
                    #     'depth_path_yx': path_yx,           # List: (y, x) in depth image w/o normalization
                    #     'depth_start_yx': depth_start_yx,   # pt: start point (y, x) in depth image w/o normalization
                    #     'depth_goal_yx': depth_goal_yx,     # pt: goal point (y, x) in depth image w/o normalization
                    #     'n_path_xy': n_path_xy              # List: (x, y) in depth image w/ normalization
                    # }

                    path_suc = res_dict_path['res']
                    depth_start_yx = res_dict_path['depth_start_yx']
                    depth_goal_yx = res_dict_path['depth_goal_yx']

                    if path_suc:
                        path_array = res_dict_path['n_path_xy']
                    else:
                        path_array = None

            if path_suc is False or lpp_method == 'line':
                # no path in the file, then set the center as a path line
                # path_array = generate_points((whole_width, whole_height))
                path_array = interpolate_points(0.5, 1.0, goal_cxcy[0], goal_cxcy[1], 500)
                depth_start_yx = [int(1.0*whole_height), int(0.5*whole_width)]
                depth_goal_yx = [int(goal_cxcy[1]*whole_height), int(goal_cxcy[0]*whole_width)]

            # # print('before expand_points: ', len(path_array))
            # # print(path_array)
            # path_array = expand_points(path_array, 50)
            # # print('after expand_points: ', len(path_array))
            # # print(path_array)

            path_array_xy = [[
                np.clip(int(x*whole_width), 0, whole_width-1), 
                np.clip(int(y*whole_height), 0, whole_height-1)
                ] for x, y in path_array]
            







            # generate masks with left all and right all masks
            dict_masks = generate_mask(cv_org_img_pt, path_array_xy)    # ['L', 'R'], all left and right area along with path.

            # generate mask using depth image on the path with physical radius
            if apply_cam_intrinsic:
                fx_r, fy_r, cx_r, cy_r = get_intrinsic_ratio(img_file_wo_ext)
                camera_intrinsics = (whole_width*fx_r, whole_height*fy_r, whole_width*cx_r, whole_height*cy_r)  # fx, fy, cx, cy
            else:
                camera_intrinsics = (whole_width*0.6, whole_height*0.6, whole_width/2, whole_height/2)  # fx, fy, cx, cy

            # generate mask of destination (not masked rgb image)
            # dict_masks['D'] = create_trapezoid_mask(cv_org_img_pt, [int(goal_cxcy[0]*whole_width), int(goal_cxcy[1]*whole_height)+50], r=50)
            depth_mask_c10 = create_circle_mask(cv_org_img_pt, [int(goal_cxcy[0]*whole_width), int(goal_cxcy[1]*whole_height)], r=int(whole_width*dst_circle_ratio/2.0))

            # generate mask using depth image (remove far objects over the destination point)
            depth_mask = create_depth_mask(depth_image, [int(goal_cxcy[0]*whole_width), int(goal_cxcy[1]*whole_height)], 
                                           radius=25, buffer_dist=5.)

            if dst_masking_depth:
                depth_mask_behind = create_depth_mask(depth_image, [int(goal_cxcy[0]*whole_width), int(goal_cxcy[1]*whole_height)], 
                                           radius=25, buffer_dist=dst_depth_meter)
                depth_mask_front = create_depth_mask(depth_image, [int(goal_cxcy[0]*whole_width), int(goal_cxcy[1]*whole_height)], 
                                           radius=25, buffer_dist=-dst_depth_meter)
                depth_mask_region = cv2.bitwise_and(depth_mask_behind, depth_mask_front)

                depth_mask_lr, depth_mask_c = mask_dest_depth_lr(depth_image, [int(goal_cxcy[0]*whole_width), int(goal_cxcy[1]*whole_height)], camera_intrinsics, physical_half_width=1.0)
                dict_masks['D'] = cv2.bitwise_and(depth_mask_lr, depth_mask_region)
                dict_masks['D'] = cv2.bitwise_or(dict_masks['D'], depth_mask_c10)
      
                # for debugging
                dict_masks_debug2 = {
                    'Dest_behind': depth_mask_behind,
                    'Dest_front': depth_mask_front,
                    'Dest_region': depth_mask_region,
                    'Dest_LR': depth_mask_lr,
                    'Dest_C': depth_mask_c,
                    'Dest_C10': depth_mask_c10,
                }
                save_debug_masked_image(img_path, cv_org_img, dict_masks_debug2, output_path_debug)

            
            # depth_mask_near_path = mask_depth_image_path_radius(depth_image, path_array_xy, camera_intrinsics, physical_radius=2.0)
            depth_mask_near_path = mask_depth_image_using_path(depth_image, path_array_xy, camera_intrinsics, physical_half_width=1.0)

            dict_masks['P'] = depth_mask_near_path

            # depth_mask_near_path_1m = mask_depth_image_using_path(depth_image, path_array_xy, camera_intrinsics, physical_half_width=2)
            # depth_mask_near_path_4m = mask_depth_image_using_path(depth_image, path_array_xy, camera_intrinsics, physical_half_width=4)

            # for debugging
            dict_masks_debug = {
                'L1': dict_masks['L'],
                'R1': dict_masks['R'],
                'Dst1': dict_masks['D'],
                'Depth_limit': depth_mask,
                'Path_half2m': depth_mask_near_path,
                # 'Path_half1m': depth_mask_near_path_1m,
                # 'Path_half4m': depth_mask_near_path_4m
            }
            save_debug_masked_image(img_path, cv_org_img_pt, dict_masks_debug, output_path_debug)

            dict_bboxes = {}
            for key, value in dict_masks.items():
                # for L, R, Dest
                dict_masks[key] = cv2.bitwise_and(dict_masks[key], depth_mask)  # remove too far region

                if key not in ['D', 'P']:    # 'L', 'R', 'P'
                    # dict_masks[key] = cv2.bitwise_and(dict_masks[key], depth_mask_near_path_4m)
                    dict_masks[key] = cv2.bitwise_and(dict_masks[key], ~dict_masks['P'])

                dict_bboxes[key] = []

                for item in bboxes:
                    if is_within_mask(dict_masks[key], item):
                        dict_bboxes[key].append(item)


            # dict_masks includes 'L', 'R', 'D', 'P'
            # save_debug_masked_image(img_path, cv_org_img_pt, dict_masks, output_path_debug)
            save_debug_masked_image(img_path, cv_org_img, dict_masks, output_path_debug)

            if dst_draw_circle:
                save_debug_masked_image(img_path, cv_org_img_circle, {'D': None}, output_path_debug)

            if dst_draw_point:
                if not dst_masking_depth:
                    save_debug_masked_image(img_path, cv_org_img_pt, {'D': None}, output_path_debug)
                else:
                    save_debug_masked_image(img_path, cv_org_img_pt, {'D': dict_masks['D']}, output_path_debug)

            if dst_draw_circle is False and dst_draw_point is False and dst_masking_circle is False and dst_masking_depth is False:
                save_debug_masked_image(img_path, cv_org_img, {'D': None}, output_path_debug)

    







            # dict_masks_etc = {
            #     'F': depth_mask,
            # }
            # save_debug_masked_image(img_path, cv_org_img_pt, dict_masks_etc, output_path_debug)
            logging.debug(f'dict_bboxes: {dict_bboxes}')

            list_removal_tokens = ['<|startoftext|>', '<|im_end|>', '[!@#$NEXT!@#$]']

            if prompt_id == 4510:
                final_query = []
                final_description = []

                for ppt_c, ppt_id in zip(['D', 'L', 'R', 'P', ''], [45101, 45102, 45103, 45104, 45105]):
                # for ppt_c, ppt_id in zip(['D'], [45101]):
                    if ppt_id == 45105:
                        if use_llm_decision or use_gpt_decision:
                            ppt_id = 45106  # do not ask final decision to vlm, just require image description.
                            prefix_prompt = None
                        else:
                            prefix_prompt = [' '.join(final_description)]
                        ppt_id_list_img_path = list_img_path
                        ppt_bboxes = bboxes
                    else:
                        prefix_prompt = None

                        file_with_ext = os.path.split(img_path)[1]
                        filename, ext = os.path.splitext(file_with_ext)
                        ppt_id_list_img_path = [os.path.join(output_path_debug, f'{filename}_{ppt_c}{ext}')]
                        ppt_bboxes = dict_bboxes[ppt_c]
                        
                    ppt_query, ppt_desc = lvm.describe_whole_images_with_boxes(ppt_id_list_img_path, ppt_bboxes, goal_label_cxcy, 
                                                                                step_by_step=True, 
                                                                                list_example_prompt=list_example_prompt,
                                                                                prompt_id=ppt_id, prefix_prompt=prefix_prompt)
                    
                    for rem in list_removal_tokens:
                        ppt_query = ppt_query.replace(rem, '')
                        ppt_desc = ppt_desc.replace(rem, '')

                    final_query.append(ppt_query)
                    final_description.append(ppt_desc)

                    logging.info(f'filename: {ppt_id_list_img_path}')
                    logging.info(f'query: {ppt_query}')
                    logging.info(f'desc: {ppt_desc}')
                    logging.info(f'bboxes: {ppt_bboxes}')


                    if ppt_id == 45106 and (use_llm_decision or use_gpt_decision):
                        # LLM for final decision, go or wait.
                        list_prompt, list_system = get_prompt(goal_label_cxcy, ppt_bboxes, trial_num=45107, sep_system=True)
                        llm_system = list_system[0]

                        prefix_prompt = [' '.join(final_description)]

                        llm_prompt = f'The image description is following: {prefix_prompt[0]} {list_prompt[0]} Say only the answers. '

                        if use_llm_decision:
                            response = llm_model.generate_llm_response(llm_system, llm_prompt)
                        elif use_gpt_decision:
                            response = gpt_model.generate_llm_response(llm_system, llm_prompt)
                        else:
                            raise AssertionError('No llm model!')
                        
                        # for rem in list_removal_tokens:
                        #     llm_prompt = llm_prompt.replace(rem, '')
                        #     response = response.replace(rem, '')

                        final_query.append(llm_prompt)
                        final_description.append(response)

                        logging.info(f'filename: LLM')
                        logging.info(f'query: {llm_prompt}')
                        logging.info(f'desc: {response}')
                        logging.info(f'bboxes: {ppt_bboxes}')

                        # LLM for summary
                        llm_prompt_system = 'A chat between a human and an AI that understands visuals in English. '
                        llm_prompt_summary_cmd = 'Summarize the following sentences into one sentence. '\
                                                 'The summarized sentence should include what is at the destination, on the left, on the right, and on the path, the recommended action, and its reason. '\
                                                 'Answer with the summarized content only. '
                        llm_prompt_summary = f'{llm_prompt_summary_cmd} This is sentences: {final_description[0]} {final_description[1]} {final_description[2]} {final_description[3]} {final_description[5]}'

                        if use_llm_summary:
                            response_summary = llm_model.generate_llm_response(llm_prompt_system, llm_prompt_summary)
                        elif use_gpt_summary:
                            response_summary = gpt_model.generate_llm_response(llm_prompt_system, llm_prompt_summary)
                        else:
                            raise AssertionError('No llm model for summary')
                        

                        final_query.append(llm_prompt_summary)
                        final_description.append(response_summary)

                        logging.info(f'filename: LLM Summary')
                        logging.info(f'query: {llm_prompt_summary}')
                        logging.info(f'desc: {response_summary}')
            else:
                final_query, final_description = lvm.describe_whole_images_with_boxes(list_img_path, bboxes, goal_label_cxcy, 
                                                                                        step_by_step=True, 
                                                                                        list_example_prompt=list_example_prompt,
                                                                                        prompt_id=args.prompt_id)
                
            # final_query.extend(['', '', '', '', '', '', ''])
            # final_description.extend(['', '', '', '', '', '', ''])

            output_dict = {
                'filename': img_file,
                'annotator': 'pipeline',
                'size_whc': [whole_width, whole_height, 3],
                'goal_position_xy': goal_cxcy,
                'goal_object_label': goal_label,
                # 'final_query': final_query,
                # 'answer': final_description

                'dest_query': final_query[0],
                'dest_desc': final_description[0],
                'left_query': final_query[1],
                'left_desc': final_description[1],
                'right_query': final_query[2],
                'right_desc': final_description[2],
                # 'front_desc': ,
                'path_query': final_query[3],
                'path_desc': final_description[3]
            }

            if use_llm_decision or use_gpt_decision:
                output_dict['query_before_llm'] = final_query[4]
                output_dict['desc_before_llm'] = final_description[4]

                output_dict['recommend_query'] = final_query[5]
                output_dict['recommend'] = final_description[5].replace('\n', ' ')

                output_dict['summary_query'] = final_query[6]
                output_dict['summary_answer'] = final_description[6]
            else:
                output_dict['recommend_query'] = final_query[4]
                output_dict['recommend'] = final_description[4].replace('\n', ' ')

            output_dict['bboxes'] = bboxes
            output_dict['path_array'] = path_array

            xml_all_info = dict_to_xml(output_dict, 'Annotation')
            save_xml(xml_all_info, os.path.join(output_path_qa, img_file_wo_ext + '.xml'))
            # save_json(output_dict, os.path.join(output_path_qa, img_file_wo_ext + '.json'))

            # 4.3. draw all whole original image
            # draw start, mid, and goal points and boxes
            img = Image.open(img_path).convert('RGB')
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

            # save debug image
            path_array_yx = [[y, x] for x, y in path_array_xy]
            path_to_debug = os.path.join(output_path_debug, f'{img_file_wo_ext}_path.jpg')
            draw_path_on_scaled_image(depth_image, path_array_yx, depth_start_yx, depth_goal_yx, 
                                      img, filename=path_to_debug, radius=20)


            path_to_debug_depth = os.path.join(output_path_debug, f'{img_file_wo_ext}_d.jpg')
            # draw_path_on_image(depth_img, path, depth_start_yx, depth_goal_yx, filename=path_to_debug_depth)
            save_and_visualize_depth_map(depth_image, path_to_debug_depth)

    convert_xml_files(output_path_qa, output_path_qa + '_json')

    return

if __name__ == '__main__':
    main()