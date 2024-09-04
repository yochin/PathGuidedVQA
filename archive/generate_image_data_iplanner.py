import os
from PIL import Image, ImageDraw, ImageFont, ImageStat
import matplotlib.pyplot as plt
import pdb
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import heapq
import networkx as nx

import json

import numpy as np
import open3d as o3d

from setGP import read_anno, get_gp, split_images, remove_outer_bbox, clamp, reorigin_bbox_point
from tools import read_text, dict_to_xml, save_xml
import argparse

def parse_args():
     parser = argparse.ArgumentParser(
        description='Generate QA (Questions and Answers) for Path Guided VQA')   

     parser.add_argument(
         '--db-dir', metavar='DIRECTORY', required=False,
         help='directory which contains images and object properties')
     
     parser.add_argument(
         '--model-name', default = None, required=False,
         help='model name')

     # lora 사용시에는 llava-model-base-dir이 있어야함
     # ex) llava-v1.5-7b-lora 사용시 base model은 llava-v1.5-7b, model은 llava-v1.5-7b-lora
     parser.add_argument(
         '--llava-model-base-dir', default = None, metavar='DIRECTORY', 
         help='directory for LLaVa checkpoints ')

     parser.add_argument(
         '--llava-model-dir', metavar='DIRECTORY', required=False,
         help='directory for LLaVa checkpoints ')
     
     parser.add_argument(
         '--output', metavar='DIRECTORY', required=False,
         help='directory for output ')
     
     parser.add_argument(
         '--ex-dir', metavar='DIRECTORY', default=None,
         help='directory which contains examples')
     
     parser.add_argument(
         '--prompt-id', default=18, type=int)

     return parser






def draw_path_on_image(depth_map, path, start, goal, filename="path_on_depth_image.png"):
    # Depth map을 흑백 이미지로 변환
    img = Image.fromarray(np.uint8(depth_map))
    draw = ImageDraw.Draw(img)
    
    # 경로 그리기
    if path and isinstance(path, list):
        # 경로 포인트 간 선 연결
        for i in range(len(path) - 1):
            draw.line([path[i][::-1], path[i + 1][::-1]], fill='red', width=2)
    
    # 시작점과 목표점 그리기
    draw.ellipse([(start[1] - 3, start[0] - 3), (start[1] + 3, start[0] + 3)], fill='green')
    draw.ellipse([(goal[1] - 3, goal[0] - 3), (goal[1] + 3, goal[0] + 3)], fill='blue')
    
    # 이미지 파일로 저장
    img.save(filename)


def rescale_point(point_yx, original_size_hw, target_size_hw):
    # 원본 이미지 크기 대비 목표 이미지 크기에 맞게 점의 위치 조정
    oy, ox = point_yx
    scale_x = target_size_hw[1] / original_size_hw[1]
    scale_y = target_size_hw[0] / original_size_hw[0]
    
    return (int(oy * scale_y), int(ox * scale_x))


def validate_point(point, max_x, max_y, margin=0):
    # 주어진 점을 이미지 범위 내로 제한
    x, y = point
    x = max(0+margin, min(x, max_x - 1 - margin))
    y = max(0+margin, min(y, max_y - 1 - margin))
    
    return (x, y)


def draw_path_on_scaled_image(depth_map, path, start, goal, rgb_image, filename="path_on_depth_image.png", radius=2):
    # 이미지 크기에 따라 점 위치 확인 및 조정
    height, width = depth_map.shape
    img_size_wh = rgb_image.size
    img_size_hw = [img_size_wh[1], img_size_wh[0]]
    start = validate_point(rescale_point(start, (height, width), img_size_hw), img_size_hw[0], img_size_hw[1], radius)
    goal = validate_point(rescale_point(goal, (height, width), img_size_hw), img_size_hw[0], img_size_hw[1], radius)

    # 새 이미지 크기로 이미지 생성
    draw = ImageDraw.Draw(rgb_image)

    # 경로 그리기
    if path and isinstance(path, list):
        scaled_path = [rescale_point(p, (height, width), img_size_hw) for p in path]
        # 경로 포인트 간 선 연결
        for i in range(len(scaled_path) - 1):
            draw.line([scaled_path[i][::-1], scaled_path[i + 1][::-1]], fill='red', width=radius)

    # 시작점과 목표점 그리기
    start_dot_lu = (start[1] - radius, start[0] - radius)
    start_dot_br = (start[1] + radius, start[0] + radius)

    goal_dot_lu = (goal[1] - radius, goal[0] - radius)
    goal_dot_br = (goal[1] + radius, goal[0] + radius)

    # draw start as yellow point
    draw.ellipse([start_dot_lu, start_dot_br], fill='yellow')
    # draw GP as orange point
    draw.ellipse([goal_dot_lu, goal_dot_br], fill='orange')

    # 이미지 파일로 저장
    rgb_image.save(filename)


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


import mplcursors
def save_and_visualize_depth_map(depth_map, filename='depth_map.png', plt_show=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    image = ax.imshow(depth_map, cmap='viridis')  # 'viridis' 컬러 맵을 사용
    ax.set_title('Depth Map Visualization')
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    plt.colorbar(image, ax=ax, orientation='vertical', label='Depth Value')

    # mplcursors를 사용하여 호버 시 depth 값 보기
    cursor = mplcursors.cursor(hover=True)
    @cursor.connect("add")
    def on_add(sel):
        # x, y 좌표를 올바르게 조정
        x, y = int(sel.target[0]), int(sel.target[1])
        depth_value = depth_map[y, x]  # y, x 순서로 배열 접근
        sel.annotation.set_text(f'X: {x}, Y: {y}, Depth: {depth_value}')
        sel.annotation.xy = (x, y)
        sel.annotation.get_bbox_patch().set_alpha(0.7)

    plt.savefig(filename)

    if plt_show:
        plt.show()


# # Global settings
# FL = 715.0873
# FY = 256 * 0.6
# FX = 256 * 0.6
# NYU_DATA = False
# FINAL_HEIGHT = 256
# FINAL_WIDTH = 256

def ThreeD_to_yx(list_path_3d):
    focal_length_x, focal_length_y = (FX, FY) if not NYU_DATA else (FL, FL)

    list_yx = []

    for x, y, z in list_path_3d:
        x = (x / z) * focal_length_x + FINAL_WIDTH / 2
        y = (y / z) * focal_length_y + FINAL_HEIGHT / 2

        list_yx.append([y / FINAL_HEIGHT, x / FINAL_WIDTH])

    return list_yx

def depth_to_3d(color_image, depth_image, depth_start_yx=None, depth_goal_yx=None):
    # Resize color image and depth to final size
    resized_color_image = color_image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS)
    resized_pred = Image.fromarray(depth_image).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)

    focal_length_x, focal_length_y = (FX, FY) if not NYU_DATA else (FL, FL)
    x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
    x = (x - FINAL_WIDTH / 2) / focal_length_x
    y = (y - FINAL_HEIGHT / 2) / focal_length_y
    z = np.array(resized_pred)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    colors = np.array(resized_color_image).reshape(-1, 3) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(image_path))[0] + ".ply"), pcd)

    if depth_start_yx is not None:
        d_start_y = depth_start_yx[0] / depth_image.shape[0] * FINAL_HEIGHT
        d_start_x = depth_start_yx[1] / depth_image.shape[1] * FINAL_WIDTH

        d_start_x = (int(d_start_x) - FINAL_WIDTH / 2) / focal_length_x
        d_start_y = (int(d_start_y) - FINAL_HEIGHT / 2) / focal_length_y
        d_start_z = depth_image[depth_start_yx[0], depth_start_yx[1]]

        pcd_start = [d_start_x * d_start_z, d_start_y * d_start_z, d_start_z]
    else:
        pcd_start = [-1, -1, -1]

    if depth_goal_yx is not None:
        d_goal_y = depth_goal_yx[0] / depth_image.shape[0] * FINAL_HEIGHT
        d_goal_x = depth_goal_yx[1] / depth_image.shape[1] * FINAL_WIDTH

        d_goal_x = (int(d_goal_x) - FINAL_WIDTH / 2) / focal_length_x
        d_goal_y = (int(d_goal_y) - FINAL_HEIGHT / 2) / focal_length_y
        d_goal_z = depth_image[depth_goal_yx[0], depth_goal_yx[1]]

        pcd_goal = [d_goal_x * d_goal_z, d_goal_y * d_goal_z, d_goal_z]
    else:
        pcd_goal = [-1, -1, -1]


    return pcd, pcd_start, pcd_goal


# Assisted by ChatGPT 4
def main():
    parser = parse_args()
    args = parser.parse_args()

    args.db_dir = '../val100'
    args.output = '../output_images'

    # 이미지가 저장된 폴더 경로
    image_path = os.path.join(args.db_dir, 'images')

    depth_path = os.path.join(args.db_dir, 'depth_anything')

    anno_path1 = os.path.join(args.db_dir, 'det_anno_pred')
    # anno_path2 = os.path.join(args.db_dir, 'det_anno_toomuch')
    anno_path2 = None
    anno_path_gt = os.path.join(args.db_dir, 'det_anno_gt')

    label_path_removal = os.path.join(args.db_dir, 'removal_labels.txt')
    label_path_gp = os.path.join(args.db_dir, 'gp_labels.txt')
    anno_path_gp = os.path.join(args.db_dir, 'anno')

    use_gp = 'anno'
    if not os.path.exists(label_path_gp):
        assert os.path.exists(anno_path_gp)
        use_gp = 'anno'

    if not os.path.exists(anno_path_gp):
        assert os.path.exists(label_path_gp)
        use_gp = 'det'
    

    output_path = args.output

    # option: treat point as bbox
    return_as_bbox = True

    # option: detection info
    use_det_info = True

    choose_one_random_gp = True     # select one random gp when many gps are detected

    assert choose_one_random_gp     # treat the output filename related to several gps



    # iplanner
    from iplanner.ip_algo import IPlannerAlgo
    import torch

    model_save = '/home/yochin/Desktop/PathGuidedVQA_Base/PathGuidedVQA/iplanner/models/plannernet.pt'
    crop_size = [360, 640]
    sensor_offset_x = 0.0
    sensor_offset_y = 0.0
    iplanner_algo = IPlannerAlgo(model_save, crop_size, sensor_offset_x, sensor_offset_y)



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
        img_as = Image.open(img_path)
        img_as = ImageOps.exif_transpose(img_as)
        img = img_as.convert('RGB')     # RGBA to RGB
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


            # depth image
            dep_path = os.path.join(depth_path, img_file_wo_ext + '.npy')
            depth_img = np.load(dep_path)       # h x w
            depth_img_h, depth_img_w = depth_img.shape

            # start and goal point to 3d
            depth_start_yx = (int(1.0 * depth_img.shape[0]), int(0.5 * depth_img.shape[1]))
            depth_goal_yx = (int(goal_cxcy[1] * depth_img.shape[0]), int(goal_cxcy[0] * depth_img.shape[1]))
            
            depth_start_yx = validate_point(depth_start_yx, depth_img_h, depth_img_w, margin=1)
            depth_goal_yx = validate_point(depth_goal_yx, depth_img_h, depth_img_w, margin=1)


            pcd, pcd_start, pcd_goal = depth_to_3d(img, depth_img, depth_start_yx, depth_goal_yx)

            # RANSAC을 사용하여 바닥 평면 모델 추정
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                    ransac_n=3,
                                                    num_iterations=1000)
            
            # 평면 모델의 계수 (a, b, c, d) : ax + by + cz + d = 0
            a, b, c, d = plane_model
            print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
            
            # 평면의 이상치를 제외한 포인트 클라우드 추출
            objects = pcd.select_by_index(inliers, invert=True)
            
            # 바닥으로부터 2m 이내 포인트 필터링
            object_points = np.asarray(objects.points)
            object_colors = np.asarray(objects.colors)  # 포인트의 색상 데이터 추출
            distances = a * object_points[:, 0] + b * object_points[:, 1] + c * object_points[:, 2] + d
            distances = np.abs(distances / np.sqrt(a**2 + b**2 + c**2))
            mask = distances <= 2000
            objects_below_2m = object_points[mask]
            colors_below_2m = object_colors[mask]

            # pdb.set_trace()
            # original: to-right in x, to-bottom in y, to-forward in z (start: 0.0, 4.0, 5.0) (goal: 1.7, 3.7, 17.7)
            # NN: 
            pcd_start_for_NN = [pcd_start[0], pcd_start[2], pcd_start[1]] # similar to [0, 0, 0]
            pcd_goal_for_NN = [pcd_goal[0], pcd_goal[2], pcd_goal[1]]
            
            # check x, y, z along with depth
            # check meter, mmter, cm
            # goal_robot_frame = torch.tensor([goal_robot_frame.point.x, goal_robot_f
            # rame.point.y, goal_robot_frame.point.z], dtype=torch.float32)[None, ...]
            AAA_GOAL = torch.tensor(pcd_goal_for_NN, dtype=torch.float32)[None, ...]
            AAA_START = torch.tensor(pcd_start_for_NN, dtype=torch.float32)[None, ...]
            goal_robot_frame = AAA_GOAL - AAA_START
            goal_rb = goal_robot_frame

            # 360 x 640
            # pdb.set_trace()
            # goal_rb[0, :] = torch.tensor([0, 5, 0])       # center-bottom = (0, 0),         [0, 0, 0] is center-bottom and start point. meter.
            goal_rb[0, -1] = torch.tensor(0)
            preds, waypoints, fear_output, _ = iplanner_algo.plan(depth_img, goal_rb)

            # preds: 1, 5, 3
            # waypoints: 1, 51, 3
            # fear_output: 1, 1
            path_points = waypoints[0].cpu().numpy()     # [1, 51, 3]
            path_points[:, [1, 2]] = path_points[:, [2, 1]]
            path_points = path_points + pcd_start
            # return None or path_points

            path_to_path = os.path.join(output_path_debug, f'{img_file_wo_ext}.json')

            if path_points is None: # No path from start to goal
                # path_array = np.array((-1, -1), dtype=np.int32)
                path_array = (-1, -1, -1)
                path_str = 'No path.'
            else:
                # Convert path to numpy array
                # path_array = np.array(path, dtype=np.int32)                
                path_array = path_points.tolist()
                path_str = "Path saved successfully."

            print(path_str)

            # Save to json
            res_dict_path = {
                'result': path_str,
                'path': path_array
            }
            with open(path_to_path, 'w') as f_json:
                json.dump(res_dict_path, f_json)
            # np.save(path_to_path, path_array)
                

            # 결과 플롯
            plt.figure(figsize=(8, 8))
            plt.scatter(objects_below_2m[:, 0], objects_below_2m[:, 2], s=1, c=colors_below_2m, alpha=0.5)
            plt.scatter(pcd_start[0], pcd_start[2], s=2, c='yellow', alpha=0.5)
            plt.scatter(pcd_goal[0], pcd_goal[2], s=2, c='orange', alpha=0.5)
            if path_points is not None:
                plt.scatter(path_points[:, 0], path_points[:, 2], s=2, c='red', alpha=0.5)
            plt.axis('equal')
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('Top View Projection of Objects Below 2m from the Floor')
            # plt.show()

            path_to_debug = os.path.join(output_path_debug, f'{img_file_wo_ext}_topview.jpg')
            plt.savefig(path_to_debug)


            # red dot & blue dot image
            img_as = Image.open(img_path)
            img_as = ImageOps.exif_transpose(img_as)
            img_draw = img_as.convert('RGB')     # RGBA to RGB
            draw = ImageDraw.Draw(img_draw)
            radius = 20

            # 텍스트 폰트 설정 (경로와 폰트 사이즈를 조정해야 할 수도 있습니다)
            font_size = int(whole_height * 0.1)
            font = ImageFont.truetype('arial.ttf', size=font_size)

            image_size = (whole_width, whole_height)

            if path_points is not None:
                path_yx = ThreeD_to_yx(path_array)
                path_yx = [[y * depth_img_h, x * depth_img_w] for y, x in path_yx]
            else:
                path_yx = None

            path_to_debug = os.path.join(output_path_debug, f'{img_file_wo_ext}.jpg')
            draw_path_on_scaled_image(depth_img, path_yx, depth_start_yx, depth_goal_yx, img, filename=path_to_debug, radius=radius)

            path_to_debug_depth = os.path.join(output_path_debug, f'{img_file_wo_ext}_d.jpg')
            # draw_path_on_image(depth_img, path, depth_start_yx, depth_goal_yx, filename=path_to_debug_depth)
            save_and_visualize_depth_map(depth_img, path_to_debug_depth)

            # 기존 Point Cloud에 점 추가
            pcd.points.extend(o3d.utility.Vector3dVector(np.array([pcd_start, 
                                                                   pcd_goal]) + 0.01))
            pcd.colors.extend(o3d.utility.Vector3dVector(np.array([[1, 1, 0],
                                                                   [1, 0.5, 0]])))
            
            if path_points is not None:
                pcd.points.extend(o3d.utility.Vector3dVector(np.array(path_points) + 0.01))
                # 모든 점을 색으로 설정
                red_color = [1, 0, 0]  # RGB 색상 (0~1 사이의 값)
                new_colors = np.array([red_color for _ in range(len(path_points))])  # 새 점의 개수만큼 빨간색 배열 생성
                pcd.colors.extend(o3d.utility.Vector3dVector(np.array(new_colors)))
            
            # # resulted 포인트 클라우드를 시각화합니다.
            # read_and_visualize_ply_with_click(pcd)

            # 수정된 PCD 파일 저장
            path_to_pcd = os.path.join(output_path_debug, f'{img_file_wo_ext}_pcd.ply')
            o3d.io.write_point_cloud(path_to_pcd, pcd)

    return



if __name__ == '__main__':
    main()
