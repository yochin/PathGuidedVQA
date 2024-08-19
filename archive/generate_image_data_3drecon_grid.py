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
    if path is not None:
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

from colorsys import rgb_to_hls, hls_to_rgb

def enhance_contrast(color):
    """
    배경색에 대해 눈에 띄게 대비되면서도 색상이 풍부한 색상을 반환합니다.
    이 함수는 HSL 색공간을 사용하여 명도를 반전시키고, 채도를 최대화합니다.
    """
    # RGB 색상을 HSL 색공간으로 변환
    h, l, s = rgb_to_hls(color[0]/255.0, color[1]/255.0, color[2]/255.0)
    
    # 명도 반전 및 채도 증가
    l = 1.0 - l  # 명도 반전
    s = 1.0  # 채도 최대화
    
    # 변경된 HSL 색상을 RGB로 다시 변환
    enhanced_rgb = hls_to_rgb(h, l, s)
    
    # RGB 색상 스케일 조정
    return tuple(int(i * 255) for i in enhanced_rgb)



def invert_color(color):
    """
    주어진 색상의 반전 색상을 계산합니다.
    이 방법은 배경색에 관계없이 더 다양한 대비 색상을 제공합니다.
    """
    # 색상의 반전 값을 계산
    inverted_color = (255 - color[0], 255 - color[1], 255 - color[2])
    return inverted_color


def get_contrasting_color(color):
    # RGB 값을 사용한 명도 계산: 0.299*R + 0.587*G + 0.114*B
    luminance = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
    # 배경 명도가 128보다 크면 어두운 색상(검은색)을, 그렇지 않으면 밝은 색상(흰색)을 반환
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)


# 'A' 문자 주변의 배경 색상을 분석하여 가장 대비되는 글자 색상 계산
def get_contrasting_text_color(image, position, text_size):
    # 'A' 문자가 그려질 범위를 대상으로 평균 색상 계산
    bbox = [position[0] - text_size // 2, position[1] - text_size // 2,
            position[0] + text_size // 2, position[1] + text_size // 2]
    region = image.crop(bbox)
    avg_color = ImageStat.Stat(region).mean  # RGB 평균 색상
    # 평균 색상에 대한 대비되는 색상 반환
    return enhance_contrast((int(avg_color[0]), int(avg_color[1]), int(avg_color[2])))

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


def extract_floor(points, distance_threshold=0.05, ransac_n=3, num_iterations=1000):
    plane_model, inliers = points.segment_plane(distance_threshold, ransac_n, num_iterations)
    floor = points.select_by_index(inliers)
    return floor, plane_model

def filter_objects_above_floor(points, floor_plane, height_threshold=2.0):
    all_points = np.asarray(points.points)
    distances = all_points[:, 2] - (floor_plane[0] * all_points[:, 0] + floor_plane[1] * all_points[:, 1] + floor_plane[3])

    mask = distances <= height_threshold
    filtered_points = points.select_by_index(np.where(mask)[0])
    return filtered_points

def create_top_view_image(points):
    points = np.asarray(points.points)
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=1)
    ax.set_aspect('equal', adjustable='datalim')
    plt.show()


def read_and_visualize_ply_with_click(pcd):    
    # 시각화
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    
    # 사용자가 포인트를 선택할 때까지 대기
    print("Please pick points in the window (Press SHIFT + Left mouse click)")
    vis.run()  # 이 함수는 사용자가 모든 포인트를 선택할 때까지 대기합니다.
    vis.destroy_window()
    
    # 선택한 포인트의 인덱스를 받아옴
    picked_points = vis.get_picked_points()
    
    if picked_points:
        # 선택한 포인트의 인덱스를 사용하여 좌표를 가져옴
        print("Picked points:")
        for index in picked_points:
            point = pcd.points[index]
            print(f"Point index {index}: Coordinate {point}")
    else:
        print("No points picked.")


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

import heapq

def a_star_search(start, goal, grid):
    def heuristic(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # 경로를 역순으로 반환하여 시작점에서 목표점 순으로 정렬

        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0]][neighbor[1]] == 0:  # 장애물이 있거나 이동 불가능한 셀
                    continue
            else:
                continue  # 그리드 범위 밖

            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return None  # 경로를 찾지 못한 경우


# 포인트 클라우드의 경계를 찾는 함수
def get_bounds(point_cloud):
    min_bound = point_cloud.get_min_bound()  # 최소 경계
    max_bound = point_cloud.get_max_bound()  # 최대 경계
    return min_bound, max_bound

def plot_grid_map_with_path(grid_map, path, filename="grid_map_with_path.png", pt_start=None, pt_goal=None):
    # # grid_map을 시각화하기 위한 임시 배열 생성
    # visual_map = np.full_like(grid_map, fill_value=255)  # 전체를 흰색으로 초기화 (보행 가능 영역)
    # for x in range(grid_map.shape[0]):
    #     for y in range(grid_map.shape[1]):
    #         if grid_map[x, y] == 0:  # 장애물 위치
    #             visual_map[x, y] = 0  # 검은색

    # # 경로에 해당하는 위치는 빨간색으로 설정 (matplotlib에서는 RGB 채널이 필요하므로 색상 변경 필요)
    # if path:
    #     for (x, y) in path:
    #         visual_map[x, y] = 255  # 먼저 흰색으로 설정

    # RGB 이미지로 변환
    rgb_image = np.zeros((grid_map.shape[0], grid_map.shape[1], 3), dtype=np.uint8)
    for i in range(rgb_image.shape[0]):
        for j in range(rgb_image.shape[1]):
            if grid_map[i, j] == 1:  # 보행 가능 경로
                rgb_image[i, j] = [255, 255, 255]  # 흰색
            elif grid_map[i, j] == 0:  # 장애물
                rgb_image[i, j] = [0, 0, 0]  # 검은색

    # 경로 빨간색으로 표시
    if path:
        for (x, y) in path:
            rgb_image[x, y] = [255, 0, 0]  # 빨간색

    if pt_start:
        rgb_image[pt_start] = [255, 255, 0] # yeollow

    if pt_goal:
        rgb_image[pt_goal] = [255, 128, 0]  # orange

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image, interpolation='nearest', origin='lower')  # 'lower'로 origin 설정
    plt.title('white-walk, black-obs, red-path')
    plt.show()
    plt.savefig(filename)
    plt.close()


def classify_points(base_floor, remaining_pcd, plane_model, height_threshold=1.0):
    # 바닥으로부터 2m 이내 포인트 필터링
    object_points = np.asarray(remaining_pcd.points)
    object_colors = np.asarray(remaining_pcd.colors)  # 포인트의 색상 데이터 추출
    a, b, c, d = plane_model
    distances = a * object_points[:, 0] + b * object_points[:, 1] + c * object_points[:, 2] + d
    distances = np.abs(distances / np.sqrt(a**2 + b**2 + c**2))
    mask = distances <= height_threshold
    # objects_below_2m = object_points[mask]
    # colors_below_2m = object_colors[mask]

    # 바닥면 이내의 점들과 장애물 분리
    floor_points = np.asarray(base_floor.points)  # 초기 바닥면 포인트
    floor_colors = np.asarray(base_floor.colors)  # 초기 바닥면 색상
    floor_points = np.vstack([floor_points, object_points[mask]])
    floor_colors = np.vstack([floor_colors, object_colors[mask]])
                              
    obstacle_points = np.asarray(object_points[~mask])
    obstacle_colors = np.asarray(object_colors[~mask])

    # 각각의 포인트 클라우드 생성
    floor_pcd = o3d.geometry.PointCloud()
    floor_pcd.points = o3d.utility.Vector3dVector(floor_points)
    floor_pcd.colors = o3d.utility.Vector3dVector(floor_colors)
    
    obstacle_pcd = o3d.geometry.PointCloud()
    obstacle_pcd.points = o3d.utility.Vector3dVector(obstacle_points)
    obstacle_pcd.colors = o3d.utility.Vector3dVector(obstacle_colors)
    
    return floor_pcd, obstacle_pcd


# 포인트 클라우드를 그리드 맵에 매핑하는 함수
def map_point_to_grid_point_idx(point, min_x, min_z, grid_size):
    x_index = int((point[0] - min_x) / grid_size)
    z_index = int((point[2] - min_z) / grid_size)

    return x_index, z_index

def map_points_to_grid(point_cloud, grid_map, min_x, min_z, grid_size, is_obstacle=True):
    for point in point_cloud.points:
        x_index, z_index = map_point_to_grid_point_idx(point, min_x, min_z, grid_size)
        if is_obstacle:  # 예를 들어 색상을 기반으로 장애물 판단
            grid_map[x_index][z_index] = 0  # 장애물
        else:
            grid_map[x_index][z_index] = 1  # 이동 가능 영역
            
def apply_buffer_to_grid(grid_map, grid_size, robot_radius):
    buffer_size = int(np.ceil(robot_radius / grid_size))  # 로봇 반지름에 해당하는 그리드 크기 계산
    buffer_grid = np.copy(grid_map)
    
    for x in range(grid_map.shape[0]):
        for y in range(grid_map.shape[1]):
            if grid_map[x, y] == 0:  # 장애물이 있는 위치
                # 장애물 주변에 버퍼 적용
                for dx in range(-buffer_size, buffer_size + 1):
                    for dy in range(-buffer_size, buffer_size + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grid_map.shape[0] and 0 <= ny < grid_map.shape[1]:
                            buffer_grid[nx, ny] = 0  # 버퍼 영역을 장애물로 설정
    return buffer_grid

# Assisted by ChatGPT 4
def main():
    parser = parse_args()
    args = parser.parse_args()

    args.db_dir = '../val100'
    args.output = '../output_images_grid'

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
    
    voxel_size = 0.05
    grid_size = 0.5  # 그리드의 크기 (미터 단위)
    
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
            # ply_path = os.path.join(depth_path, img_file_wo_ext + '.ply')
            # pcd = o3d.io.read_point_cloud(ply_path)

            # # 포인트 클라우드 다운샘플링
            downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)    # 0.05 meter has 1 pixel

            num_pcd = np.asarray(pcd.points).shape[0]
            num_downpcd = np.asarray(downpcd.points).shape[0]
            print(f'pcd {num_pcd} is sampled to downpcd {num_downpcd}.')

            # add start and end point to downpcd
            downpcd.points.extend(o3d.utility.Vector3dVector(np.array([pcd_start, pcd_goal])))
            downpcd.colors.extend(o3d.utility.Vector3dVector(np.array([[1, 1, 0], [1, 0.5, 0]])))
            
            # # 읽은 포인트 클라우드를 시각화합니다.
            # read_and_visualize_ply_with_click(downpcd)

            # RANSAC을 사용하여 지면과 장애물 분리
            plane_model, inliers = downpcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            floor = downpcd.select_by_index(inliers)
            not_floor = downpcd.select_by_index(inliers, invert=True)

            # 바닥면과 장애물 포인트 클라우드 분류
            # obstacles = downpcd.select_by_index(inliers, invert=True)
            floor, obstacles = classify_points(floor, not_floor, plane_model, height_threshold=2.0)


            
            




            num_floor = np.asarray(floor.points).shape[0]
            num_obstacles = np.asarray(obstacles.points).shape[0]
            print(f'downpcd {num_downpcd} = floor {num_floor} + obstacles {num_obstacles}.')

            # 평면 모델의 계수 (a, b, c, d) : ax + by + cz + d = 0
            a, b, c, d = plane_model
            print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

            # # temp 결과 플롯
            # plt.figure(figsize=(10, 10))
            # plt_ax = plt.axes(projection="3d")

            # floor_points = np.asarray(floor.points)
            # floor_colors = np.asarray(floor.colors)
            # plt_ax.scatter(floor_points[:, 0], floor_points[:, 1], floor_points[:, 2], s=1, c=floor_colors, alpha=0.5)

            # obstacles_points = np.asarray(obstacles.points)
            # obstacles_colors = np.asarray(obstacles.colors)
            # plt_ax.scatter(obstacles_points[:, 0], obstacles_points[:, 1], obstacles_points[:, 2], s=1, c='green', alpha=0.5)
            # plt_ax.scatter(pcd_start[0], pcd_start[1], pcd_start[2], s=2, c='yellow', alpha=0.5)
            # plt_ax.scatter(pcd_goal[0], pcd_goal[1], pcd_goal[2], s=2, c='orange', alpha=0.5)
            
            # # plt.axis('equal')
            # plt_ax.set_xlabel('X (m)')
            # plt_ax.set_ylabel('Y (m)')
            # plt_ax.set_zlabel('Z (m)')
            # plt.title('Top View Projection (green is obstacle)')
            # plt.show()

            # 그리드 맵 생성
            # 그리드 맵 초기화
            min_bound, max_bound = get_bounds(downpcd)

            # 포인트 클라우드에서 x와 z의 최소 및 최대 좌표를 추출
            min_x, min_z = min_bound[0], min_bound[2]
            max_x, max_z = max_bound[0], max_bound[2]

            grid_width = int((max_x - min_x) / grid_size) + 1  # 그리드의 너비
            grid_height = int((max_z - min_z) / grid_size) + 1  # 그리드의 높이
            grid_map = np.zeros((grid_width, grid_height))  # zero = not-walkable

            # 포인트 클라우드를 그리드 맵에 매핑
            map_points_to_grid(obstacles, grid_map, min_x, min_z, grid_size, is_obstacle=True)
            map_points_to_grid(floor, grid_map, min_x, min_z, grid_size, is_obstacle=False)

            # 예시 사용
            # robot_radius = 0.5  # 로봇의 반지름 (미터)
            # grid_map_with_buffer = apply_buffer_to_grid(grid_map, grid_size, robot_radius)
            grid_map_with_buffer = grid_map

            path_to_grid = os.path.join(output_path_debug, f'{img_file_wo_ext}_grid.jpg')
            

            grid_start_point = map_point_to_grid_point_idx(pcd_start, min_x, min_z, grid_size)
            grid_goal_point = map_point_to_grid_point_idx(pcd_goal, min_x, min_z, grid_size)
            grid_map_with_buffer[grid_start_point] = 1
            grid_map_with_buffer[grid_goal_point] = 1

            # 경로 계산
            grid_path = a_star_search(grid_start_point, grid_goal_point, grid_map_with_buffer)

            def convert_path_to_world_coordinates(path, grid_size, origin=(0, 0, 0)):
                """
                그리드 맵 상의 경로를 실제 세계 좌표로 변환합니다.
                
                :param path: 그리드 맵 상의 경로 (리스트 형태의 튜플)
                :param grid_size: 그리드 한 칸의 실제 크기 (미터 단위)
                :param origin: 그리드 맵의 원점 좌표 (실제 세계에서의 위치, 미터 단위)
                :return: 실제 세계 좌표로 변환된 경로
                """
                world_path = []
                for (x, z) in path:
                    real_x = origin[0] + (x + 0.5) * grid_size  # 그리드 셀 중앙을 실제 좌표로 변환
                    real_y = origin[1]
                    real_z = origin[2] + (z + 0.5) * grid_size  # 그리드 셀 중앙을 실제 좌표로 변환
                    
                    world_path.append((real_x, real_y, real_z))
                return np.array(world_path)
            
            if grid_path:
                path_str = "Path saved successfully."
                # 그리드 사이즈와 원점 설정 (예: 그리드 사이즈는 0.1 미터, 원점은 (0,0))
                pcd_path = convert_path_to_world_coordinates(grid_path, grid_size=grid_size, origin=(0, 0, 0))

                offset_start = pcd_start - pcd_path[0]
                # offset_goal = pcd_path[-1] - pcd_goal
                pcd_path = convert_path_to_world_coordinates(grid_path, grid_size=grid_size, origin=offset_start)

                pcd_path_save = pcd_path.tolist()
            else:
                path_str = 'No path.'
                pcd_path_save = None
                pcd_path = None

            print(path_str)

            # Save to json
            path_to_path = os.path.join(output_path_debug, f'{img_file_wo_ext}.json')
            res_dict_path = {
                'result': path_str,
                'pcd_path': pcd_path_save,
                'grid_path': grid_path

            }
            with open(path_to_path, 'w') as f_json:
                json.dump(res_dict_path, f_json)
                
            # 결과 플롯
            plt.figure(figsize=(8, 8))
            floor_points = np.asarray(floor.points)
            floor_colors = np.asarray(floor.colors)
            plt.scatter(floor_points[:, 0], floor_points[:, 2], s=1, c=floor_colors, alpha=0.5)

            obstacles_points = np.asarray(obstacles.points)
            plt.scatter(obstacles_points[:, 0], obstacles_points[:, 2], s=1, c='green', alpha=0.5)
            plt.scatter(pcd_start[0], pcd_start[2], s=2, c='yellow', alpha=0.5)
            plt.scatter(pcd_goal[0], pcd_goal[2], s=2, c='orange', alpha=0.5)
            if grid_path is not None:
                plt.scatter(pcd_path[:, 0], pcd_path[:, 2], s=2, c='red', alpha=0.5)
            plt.axis('equal')
            plt.xlabel('X (m)')
            plt.ylabel('Z (m)')
            plt.title('Top View Projection of Objects Below 2m from the Floor')
            # plt.show()

            path_to_debug = os.path.join(output_path_debug, f'{img_file_wo_ext}_topview.jpg')
            plt.savefig(path_to_debug)

            path_to_grid = os.path.join(output_path_debug, f'{img_file_wo_ext}_grid.jpg')
            plot_grid_map_with_path(grid_map_with_buffer, grid_path, path_to_grid, grid_start_point, grid_goal_point)

            # # red dot & blue dot image
            # img_as = Image.open(img_path)
            # img_draw = img_as.convert('RGB')     # RGBA to RGB
            # draw = ImageDraw.Draw(img_draw)
            # radius = 20

            # # 텍스트 폰트 설정 (경로와 폰트 사이즈를 조정해야 할 수도 있습니다)
            # font_size = int(whole_height * 0.1)
            # font = ImageFont.truetype('arial.ttf', size=font_size)

            # image_size = (whole_width, whole_height)

            # if pcd_path is not None:
            #     path_yx = ThreeD_to_yx(pcd_path)
            #     path_yx = [[y * depth_img_h, x * depth_img_w] for y, x in path_yx]
            # else:
            #     path_yx = None

            # path_to_debug = os.path.join(output_path_debug, f'{img_file_wo_ext}.jpg')
            # draw_path_on_scaled_image(depth_img, path_yx, depth_start_yx, depth_goal_yx, img, filename=path_to_debug, radius=radius)

            # path_to_debug_depth = os.path.join(output_path_debug, f'{img_file_wo_ext}_d.jpg')
            # # draw_path_on_image(depth_img, path, depth_start_yx, depth_goal_yx, filename=path_to_debug_depth)
            # save_and_visualize_depth_map(depth_img, path_to_debug_depth)

            # # 기존 Point Cloud에 점 추가
            # pcd.points.extend(o3d.utility.Vector3dVector(np.array([pcd_start, 
            #                                                        pcd_goal]) + 0.01))
            # pcd.colors.extend(o3d.utility.Vector3dVector(np.array([[1, 1, 0],
            #                                                        [1, 0.5, 0]])))
            
            # if pcd_path is not None:
            #     pcd.points.extend(o3d.utility.Vector3dVector(np.array(pcd_path) + 0.01))
            #     # 모든 점을 색으로 설정
            #     red_color = [1, 0, 0]  # RGB 색상 (0~1 사이의 값)
            #     new_colors = np.array([red_color for _ in range(len(pcd_path))])  # 새 점의 개수만큼 빨간색 배열 생성
            #     pcd.colors.extend(o3d.utility.Vector3dVector(np.array(new_colors)))
            
            # # # resulted 포인트 클라우드를 시각화합니다.
            # # read_and_visualize_ply_with_click(pcd)

            # # 수정된 PCD 파일 저장
            # path_to_pcd = os.path.join(output_path_debug, f'{img_file_wo_ext}_pcd.ply')
            # o3d.io.write_point_cloud(path_to_pcd, pcd)

    return



if __name__ == '__main__':
    main()
