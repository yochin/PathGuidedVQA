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


def find_path(pcd, start, goal, radius=0.5, height_limit=1):
    # 포인트 클라우드를 kd-tree로 변환
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # 그래프 생성
    G = nx.Graph()

    # 포인트 클라우드에서 각 포인트를 노드로 추가
    points = np.asarray(pcd.points)
    for i, point in enumerate(points):
        G.add_node(i, pos=point)

    # 각 노드간에 엣지 생성
    for i in range(len(points)):
        [k, idx, dists] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
        if k > 1:
            avg_dist = np.mean(np.sqrt(dists[1:]))  # 자기 자신을 제외한 점들의 거리 평균
            current_height = points[i][1]		# x, y, z
            for j in idx[1:]:  # 자기 자신을 제외하고 연결
                if i != j:
                    # 높이 차이 확인
                    height_change = np.abs(current_height - points[j][1])
                    if height_change <= height_limit:
                        G.add_edge(i, j, weight=avg_dist)  # 높이 제한 조건을 만족하는 경우에만 엣지 생성

    # 시작점과 종료점 인덱스 찾기
    start_idx = np.argmin(np.linalg.norm(points - np.array(start), axis=1))
    goal_idx = np.argmin(np.linalg.norm(points - np.array(goal), axis=1))

    # 최단 경로 찾기 (Dijkstra 알고리즘 사용)
    path = nx.dijkstra_path(G, source=start_idx, target=goal_idx, weight='weight')

    # 경로를 포인트 클라우드 인덱스로 반환
    path_points = points[path]
    return path_points

def find_path_directFirst(pcd, start, goal, radius=0.5, height_limit=1):
    # 포인트 클라우드를 kd-tree로 변환
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # 그래프 생성
    G = nx.Graph()

    # 포인트 클라우드에서 각 포인트를 노드로 추가
    points = np.asarray(pcd.points)
    for i, point in enumerate(points):
        G.add_node(i, pos=point)

    # 시작점과 종료점 벡터 생성 (x, z만 고려)
    direction_vector = np.array([goal[0] - start[0], 0, goal[2] - start[2]])
    direction_norm = np.linalg.norm(direction_vector)

    # 각 노드간에 엣지 생성
    for i in range(len(points)):
        [k, idx, dists] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
        if k > 1:
            current_height = points[i][1]		# x, y, z
            for j in idx[1:]:  # 자기 자신을 제외하고 연결
                if i != j:
                    # 높이 차이 확인
                    height_change = np.abs(current_height - points[j][1])
                    if height_change <= height_limit:
                        segment_vector = np.array([points[j][0] - points[i][0], 0, points[j][2] - points[i][2]])
                        # 직선 경로와의 방향 일치도를 측정하여 가중치 계산
                        dot_product = np.dot(segment_vector, direction_vector)
                        angle_cosine = dot_product / (np.linalg.norm(segment_vector) * direction_norm)
                        # 직선 경로와의 일치도를 가중치에 반영
                        weight = np.sqrt(dists[idx.index(j)]) * (1 - angle_cosine**2)
                        G.add_edge(i, j, weight=weight)

    # 시작점과 종료점 인덱스 찾기
    start_idx = np.argmin(np.linalg.norm(points - np.array(start), axis=1))
    goal_idx = np.argmin(np.linalg.norm(points - np.array(goal), axis=1))

    # 최단 경로 찾기 (Dijkstra 알고리즘 사용)
    path = nx.dijkstra_path(G, source=start_idx, target=goal_idx, weight='weight')

    # 경로를 포인트 클라우드 인덱스로 반환
    path_points = points[path]
    return path_points

def find_path_directFirst_except(pcd, start, goal, radius=0.5, height_limit=1.):
    # # 시작점과 종료점 벡터 생성 (x, z만 고려)
    # direction_vector = np.array([goal[0] - start[0], 0, goal[2] - start[2]])
    # direction_norm = np.linalg.norm(direction_vector)
    # if direction_norm == 0:
    #     print("Direction vector btw start and goal has zero length.")
    #     return None

    # 목표지점 벡터 계산
    goal_vector = np.array(goal) - np.array(start)
    goal_norm = np.linalg.norm(goal_vector)
    if goal_norm == 0:
        print("Goal vector has zero length.")
        return None

    # 포인트 클라우드를 kd-tree로 변환
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # 그래프 생성
    G = nx.Graph()

    # 포인트 클라우드에서 각 포인트를 노드로 추가
    points = np.asarray(pcd.points)     # only x, y, z is considered
    for i, point in enumerate(points):
        G.add_node(i, pos=point)

    # 각 노드간에 엣지 생성
    for i in range(len(points)):
        [k, idx, dists] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)   # num_pts, index, distances

        if k > 1:
            idx = list(idx)  # IntVector를 Python 리스트로 변환
            dists = list(dists)  # DoubleVector를 Python 리스트로 변환
            current_height = points[i][1]		# x, y, z
            for j in idx[1:]:  # 자기 자신을 제외하고 연결
                if i != j:
                    # 높이 차이 확인
                    height_change = np.abs(current_height - points[j][1])
                    if height_change <= height_limit:
                        segment_vector = np.array([points[j][0] - points[i][0], 0, points[j][2] - points[i][2]])
                        segment_norm = np.linalg.norm(segment_vector)
                        if segment_norm == 0:
                            continue  # Ignore zero-length segments

                        # # 직선 경로와의 방향 일치도를 측정하여 가중치 계산
                        # dot_product = np.dot(segment_vector, direction_vector)
                        # angle_cosine = dot_product / (segment_norm * direction_norm)
                        # # 직선 경로와의 일치도를 가중치에 반영
                        # weight = np.sqrt(dists[idx.index(j)]) * (1 - angle_cosine**2)

                        # 목표지점까지의 벡터와 현재 이동 벡터의 각도 계산
                        dot_product = np.dot(segment_vector, goal_vector)
                        angle_cosine = dot_product / (segment_norm * goal_norm)
                        # 각도에 기반한 가중치 및 이동 거리 추가
                        weight = segment_norm * (1 + (1 - angle_cosine**2))
                        
                        G.add_edge(i, j, weight=weight)

    # 시작점과 종료점 인덱스 찾기
    start_idx = np.argmin(np.linalg.norm(points - np.array(start), axis=1))
    goal_idx = np.argmin(np.linalg.norm(points - np.array(goal), axis=1))

    # 최단 경로 찾기 (Dijkstra 알고리즘 사용)
    try:
        path = nx.dijkstra_path(G, source=start_idx, target=goal_idx, weight='weight')
        path_points = points[path]

        return path_points
    except nx.NetworkXNoPath:
        print("No path could be found between the specified start and goal points.")

        return None
    
from concurrent.futures import ThreadPoolExecutor

def calculate_weight(args):
    i, j, points, goal_vector, goal_norm = args
    segment_vector = points[j] - points[i]
    segment_norm = np.linalg.norm(segment_vector)
    if segment_norm == 0:
        return None
    
    # # check the aling btw next node and goal node
    # dot_product = np.dot(segment_vector, goal_vector)
    # angle_cosine = dot_product / (segment_norm * goal_norm)
    # weight = segment_norm * (1 + (1 - angle_cosine**2))

    # consider only distance
    weight = segment_norm

    return i, j, weight

def find_path_thread(pcd, start, goal, k=10, radius=0.1, height_limit=1.):
    assert k * radius < 0

    # 포인트 클라우드를 kd-tree로 변환
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # 그래프 생성
    G = nx.Graph()

    # 포인트 클라우드에서 각 포인트를 노드로 추가
    points = np.asarray(pcd.points)
    for i in range(len(points)):
        G.add_node(i, pos=points[i])

    # 목표지점 벡터 계산
    goal_vector = np.array(goal) - np.array(start)
    goal_norm = np.linalg.norm(goal_vector)
    if goal_norm == 0:
        print("Goal vector has zero length.")
        return None

    # 엣지 추가를 위한 병렬 처리
    with ThreadPoolExecutor() as executor:
        futures = []

        if k > 0:
            for i in range(len(points)):
                [k, idx, dists] = pcd_tree.search_knn_vector_3d(pcd.points[i], k)
                for j in idx[1:]:
                    if i != j:
                        height_change = np.abs(points[i][1] - points[j][1])     # height
                        if height_change <= height_limit:
                            futures.append(executor.submit(calculate_weight, (i, j, points, goal_vector, goal_norm)))

        if radius > 0:
            for i in range(len(points)):
                [k, idx, dists] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
                idx = list(idx)
                dists = list(dists)
                if k > 1:
                    current_height = points[i][1]
                    for j in idx[1:]:
                        if i != j:
                            height_change = np.abs(current_height - points[j][1])
                            if height_change <= height_limit:
                                futures.append(executor.submit(calculate_weight, (i, j, points, goal_vector, goal_norm)))

        
        for future in futures:
            result = future.result()
            if result:
                i, j, weight = result
                G.add_edge(i, j, weight=weight)

    # 시작점과 종료점 인덱스 찾기
    start_idx = np.argmin(np.linalg.norm(points - np.array(start), axis=1))
    goal_idx = np.argmin(np.linalg.norm(points - np.array(goal), axis=1))

    # 최단 경로 찾기
    try:
        path = nx.dijkstra_path(G, source=start_idx, target=goal_idx, weight='weight')
        path_points = points[path]
        return path_points
    except nx.NetworkXNoPath:
        print("No path could be found between the specified start and goal points.")
        return None
    


from heapq import heappop, heappush

def heuristic_cost_estimate(start, goal, weight_y=10.0):
    # consider x, y, z, plane
    # return np.linalg.norm(np.array(goal) - np.array(start))

    # Heuristic is simply the Euclidean distance in the x-z plane
    # return np.sqrt((goal[0] - start[0])**2 + (goal[2] - start[2])**2)

     # Calculate differences in each dimension
    dx = abs(goal[0] - start[0])
    dy = abs(goal[1] - start[1])
    dz = abs(goal[2] - start[2])

    # Apply higher cost weight to the y dimension
    cost = (dx ** 2 + (weight_y * dy) ** 2 + dz ** 2) ** 0.5

    return cost

def a_star_search(G, start, goal):
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in G.nodes}
    g_score[start] = 0
    f_score = {node: float('inf') for node in G.nodes}
    f_score[start] = heuristic_cost_estimate(G.nodes[start]['pos'], G.nodes[goal]['pos'])   # estimated cost
    
    while open_set:
        current = heappop(open_set)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(current)
            path.reverse()
            return path

        for neighbor in G.neighbors(current):
            tentative_g_score = g_score[current] + G[current][neighbor]['weight']
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic_cost_estimate(G.nodes[neighbor]['pos'], G.nodes[goal]['pos'])
                heappush(open_set, (f_score[neighbor], neighbor))
    return None

def find_path_astar(pcd_org, start, goal, voxel_size=None, radius=0.5, knn=5, height_limit=0.1):
    if voxel_size is not None:
        # 포인트 클라우드를 voxel_down_sample로 다운샘플링
        pcd = pcd_org.voxel_down_sample(voxel_size)
    else:
        pcd = pcd_org

    # 포인트 클라우드를 kd-tree로 변환
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # 그래프 생성
    G = nx.Graph()

    # 포인트 클라우드에서 각 포인트를 노드로 추가
    points = np.asarray(pcd.points)
    for i in range(len(points)):
        G.add_node(i, pos=points[i])

    goal_vector = np.array(goal) - np.array(start)
    goal_norm = np.linalg.norm(goal_vector)

    # 각 노드간에 엣지 생성
    for i in range(len(points)):
        # # find all point in radius
        # depth = points[i][2]  # 깊이 값으로 z 좌표 사용
        
        # print('Radius search:')
        [k, idx, dists] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
        # print(f"Found {k} points within a radius of {radius}")

        if k == 0 and knn > 0:
            k, idx, dists = pcd_tree.search_knn_vector_3d(pcd.points[i], knn)

        if k > 0:
            idx = list(idx)
            dists = list(dists)
            current_point = points[i]
            for j in range(k):
                neighbor_index = idx[j]
                if neighbor_index != i:
                    neighbor_point = points[neighbor_index]
                    height_change = np.abs(current_point[1] - neighbor_point[1])    # x, y, z
                    if height_change <= height_limit:
                        segment_vector = neighbor_point - current_point
                        segment_norm = np.linalg.norm(segment_vector)
                        if segment_norm > 0:
                            direction_cosine = np.dot(segment_vector, goal_vector) / (segment_norm * goal_norm)
                            # distance = np.linalg.norm(current_point - neighbor_point)
                            distance = heuristic_cost_estimate(current_point, neighbor_point)
                            weight = distance * (1 + (1 - direction_cosine**2))
                            G.add_edge(i, neighbor_index, weight=weight)


    # 시작점과 종료점 인덱스 찾기
    start_idx = np.argmin(np.linalg.norm(points - np.array(start), axis=1))
    goal_idx = np.argmin(np.linalg.norm(points - np.array(goal), axis=1))

    # 최단 경로 찾기 (A* 알고리즘 사용)
    path_indices = a_star_search(G, start_idx, goal_idx)
    if path_indices is not None:
        path_points = points[path_indices]
        return path_points
    else:
        print("No path could be found between the specified start and goal points.")
        return None

def pcd_lpp(depth_img, goal_cxcy, output_path_debug):

    # start and goal point to 3d
    depth_start_yx = (int(1.0 * depth_img.shape[0]), int(0.5 * depth_img.shape[1]))
    depth_goal_yx = (int(goal_cxcy[1] * depth_img.shape[0]), int(goal_cxcy[0] * depth_img.shape[1]))
    
    depth_start_yx = validate_point(depth_start_yx, depth_img_h, depth_img_w, margin=1)
    depth_goal_yx = validate_point(depth_goal_yx, depth_img_h, depth_img_w, margin=1)


    pcd, pcd_start, pcd_goal = depth_to_3d(img, depth_img, depth_start_yx, depth_goal_yx)

    # ply_path = os.path.join(depth_path, img_file_wo_ext + '.ply')
    # pcd = o3d.io.read_point_cloud(ply_path)

    # # # 읽은 포인트 클라우드를 시각화합니다.
    # read_and_visualize_ply_with_click(pcd)

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

    # TODO: floor to pcd 

    # # Y 방향으로 바닥 추정 (가장 낮은 10%의 Y 값을 평균낸다)
    # y_floor = np.percentile(object_points[:, 1], 10)
    
    # # 바닥면에서 2m 이하인 포인트만 필터링
    # mask = (object_points[:, 1] <= y_floor + 2)
    # objects_below_2m = object_points[mask]
    # colors_below_2m = object_colors[mask]


    # astar and get 3d path -> convert to 2d and save
    # path_points = find_path_directFirst_except(pcd, pcd_start, pcd_goal, radius=0.5, height_limit=0.75)
    # path_points = find_path_thread(pcd, pcd_start, pcd_goal, k=-1, radius=1.0, height_limit=0.75)		# Dijkstra algorithm - slow
    path_points = find_path_astar(pcd, pcd_start, pcd_goal, voxel_size=0.1, radius=1.5, knn=0, height_limit=0.25)				# A* algorithm
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

            pcd_lpp(depth_img, goal_cxcy, output_path_debug)
    return



if __name__ == '__main__':
    main()
