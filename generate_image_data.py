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

import json

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


# def heuristic(a, b):
#     return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def heuristic(a, b, weight=1.5):
    return np.linalg.norm(np.array(a) - np.array(b)) * weight


def smooth_cost(depth, current, neighbor):
    # return abs(depth[current] - depth[neighbor])
    # Calculate the depth difference and square it to emphasize larger differences
    return (depth[current] - depth[neighbor]) ** 2


def astar(depth_map, start, goal):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    open_heap = []
    heapq.heappush(open_heap, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while open_heap:
        current_f, current_cost, current = heapq.heappop(open_heap)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        for move in neighbors:
            neighbor = (current[0] + move[0], current[1] + move[1])
            if 0 <= neighbor[0] < depth_map.shape[0] and 0 <= neighbor[1] < depth_map.shape[1]:
                new_cost = current_cost + 1 + smooth_cost(depth_map, current, neighbor)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
    
    return None


def average_depth_change(depth_map, center, radius):
    r, c = center
    rows, cols = depth_map.shape
    total_depth_change = 0
    count = 0
    
    # Calculate the average depth change within a circle of given radius
    for dr in range(-radius, radius+1):
        for dc in range(-radius, radius+1):
            if dr**2 + dc**2 <= radius**2:  # Within the circle
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    total_depth_change += (depth_map[r, c] - depth_map[nr, nc])**2
                    count += 1
    
    return total_depth_change / count if count else 0


def astar_r(depth_map, start, goal, radius):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    open_heap = []
    heapq.heappush(open_heap, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while open_heap:
        current_f, current_cost, current = heapq.heappop(open_heap)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        for move in neighbors:
            neighbor = (current[0] + move[0], current[1] + move[1])
            if 0 <= neighbor[0] < depth_map.shape[0] and 0 <= neighbor[1] < depth_map.shape[1]:
                new_cost = current_cost + 1 + average_depth_change(depth_map, neighbor, radius)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
    
    return None


def sample_average_depth_change(depth_map, center, radius, sample_count=10):
    r, c = center
    rows, cols = depth_map.shape
    total_depth_change = 0
    samples = []
    
    # Generate samples within the circle
    while len(samples) < sample_count:
        dr = random.randint(-radius, radius)
        dc = random.randint(-radius, radius)
        if dr**2 + dc**2 <= radius**2:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                samples.append((nr, nc))
    
    # Calculate the average depth change for the samples
    for (nr, nc) in samples:
        total_depth_change += (depth_map[r, c] - depth_map[nr, nc]) ** 2
    
    return total_depth_change / len(samples) if samples else 0

def astar_r_fast(depth_map, start, goal, radius):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    open_heap = []
    heapq.heappush(open_heap, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while open_heap:
        current_f, current_cost, current = heapq.heappop(open_heap)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        for move in neighbors:
            neighbor = (current[0] + move[0], current[1] + move[1])
            if 0 <= neighbor[0] < depth_map.shape[0] and 0 <= neighbor[1] < depth_map.shape[1]:
                new_cost = current_cost + 1 + sample_average_depth_change(depth_map, neighbor, radius)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
    
    return None

def sample_average_depth_change_thresh(depth_map, center, radius, sample_count=10, threshold_sq=float('inf')):
    r, c = center
    rows, cols = depth_map.shape
    total_depth_change = 0
    samples = []
    
    # Generate samples within the circle
    while len(samples) < sample_count:
        dr = random.randint(-radius, radius)
        dc = random.randint(-radius, radius)
        if dr**2 + dc**2 <= radius**2:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                samples.append((nr, nc))
    
    # Calculate the average depth change for the samples
    for (nr, nc) in samples:
        change = (depth_map[r, c] - depth_map[nr, nc]) ** 2
        if change > threshold_sq:  # Check against the threshold
            return float('inf')  # Prohibitively high cost to avoid this path
        total_depth_change += change
    
    return total_depth_change / len(samples) if samples else 0

def astar_r_fast_thresh(depth_map, start, goal, radius, threshold_sq):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    open_heap = []
    heapq.heappush(open_heap, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {}
    visited = set()
    came_from[start] = None
    cost_so_far[start] = 0
    
    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        if current in visited:
            continue
        visited.add(current)
        
        for move in neighbors:
            neighbor = (current[0] + move[0], current[1] + move[1])
            if 0 <= neighbor[0] < depth_map.shape[0] and 0 <= neighbor[1] < depth_map.shape[1]:
                new_cost = current_cost + 1 + sample_average_depth_change_thresh(depth_map, neighbor, radius, 10, threshold_sq)
                if new_cost == float('inf'):  # Check if path is prohibited
                    continue  # Skip adding this neighbor to the heap
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

        if not open_heap:
            return "No path found - goal is unreachable with the given threshold."

    return "No path found - exhausted all possibilities."

def get_sorted_neighbors(current, goal):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    vector_to_goal = np.array(goal) - np.array(current)
    if np.linalg.norm(vector_to_goal) > 0:
        vector_to_goal = vector_to_goal / np.linalg.norm(vector_to_goal)
    directions.sort(key=lambda d: -np.dot(vector_to_goal, d))
    return directions

def astar_r_fast_thresh_sorted(depth_map, start, goal, radius, threshold_sq):
    open_heap = []
    heapq.heappush(open_heap, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {}
    visited = set()
    came_from[start] = None
    cost_so_far[start] = 0

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        if current in visited:
            continue
        visited.add(current)
        
        neighbors = get_sorted_neighbors(current, goal)
        for move in neighbors:
            neighbor = (current[0] + move[0], current[1] + move[1])
            if 0 <= neighbor[0] < depth_map.shape[0] and 0 <= neighbor[1] < depth_map.shape[1]:
                new_cost = current_cost + 1 + sample_average_depth_change_thresh(depth_map, neighbor, radius, 10, threshold_sq)
                if new_cost == float('inf'):
                    continue
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

        if not open_heap:
            return "No path found - goal is unreachable with the given threshold."

    return "No path found - exhausted all possibilities."

def calculate_dynamic_radius(depth_value, max_radius=5, min_radius=1, max_depth=20):
    # Linear scaling of radius based on depth
    scale = (max_radius - min_radius) / max_depth
    dynamic_radius = int(max_radius - scale * depth_value)
    return max(min_radius, dynamic_radius)


def sample_average_depth_change_thresh_adaptR(depth_map, center, sample_count=10, threshold_sq=float('inf'), threshold_depth_limit=100.):
    r, c = center
    rows, cols = depth_map.shape
    depth_value = depth_map[r, c]
    radius = calculate_dynamic_radius(depth_value)
    total_depth_change = 0
    samples = []
    
    # Generate samples within the circle
    while len(samples) < sample_count:
        dr = random.randint(-radius, radius)
        dc = random.randint(-radius, radius)
        if dr**2 + dc**2 <= radius**2:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                samples.append((nr, nc))
    
    # Calculate the average depth change for the samples
    for (nr, nc) in samples:
        change = (depth_map[r, c] - depth_map[nr, nc]) ** 2
        if depth_map[r, c] < threshold_depth_limit or depth_map[nr, nc] < threshold_depth_limit:
            if change > threshold_sq:  # Check against the threshold
                return float('inf')  # Prohibitively high cost to avoid this path
        total_depth_change += change
    
    return total_depth_change / len(samples) if samples else 0


# find a path array using astar algorithm
#   threshold_sq: if depth change is larget than threshold_sq, its cost is inf.
#   heuristic_weight: weight for pixel moving
#   threshold_depth_limit: depth limit for threshold_sq. threshold_sq is applied in threshold_depth_limit.
def astar_adaptR_fast_thresh_sorted(depth_map, start, goal, threshold_sq, heuristic_weight=1.5, threshold_depth_limit=100.):
    open_heap = []
    heapq.heappush(open_heap, (0 + heuristic(start, goal, heuristic_weight), 0, start))
    came_from = {}
    cost_so_far = {}
    visited = set()
    came_from[start] = None
    cost_so_far[start] = 0

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        if current in visited:
            continue
        visited.add(current)
        
        neighbors = get_sorted_neighbors(current, goal)
        for move in neighbors:
            neighbor = (current[0] + move[0], current[1] + move[1])
            if 0 <= neighbor[0] < depth_map.shape[0] and 0 <= neighbor[1] < depth_map.shape[1]:
                new_cost = current_cost + 1 + sample_average_depth_change_thresh_adaptR(depth_map, neighbor, 10, 
                                                                                        threshold_sq, threshold_depth_limit)
                if new_cost == float('inf'):
                    continue
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal, heuristic_weight)
                    heapq.heappush(open_heap, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

        if not open_heap:
            return "No path found - goal is unreachable with the given threshold."

    return "No path found - exhausted all possibilities."



def calculate_dynamic_threshold(depth_value, base_threshold=1.0, max_depth=10.):
    scale = base_threshold / max_depth
    dynamic_threshold = base_threshold + scale * depth_value
    return dynamic_threshold

def sample_average_depth_change_adaptTH_adaptR(depth_map, center, sample_count, base_threshold):
    r, c = center
    rows, cols = depth_map.shape
    depth_value = depth_map[r, c]
    radius = calculate_dynamic_radius(depth_value)
    threshold = calculate_dynamic_threshold(depth_value, base_threshold)
    total_depth_change = 0
    samples = []

    while len(samples) < sample_count:
        dr = random.randint(-radius, radius)
        dc = random.randint(-radius, radius)
        if dr**2 + dc**2 <= radius**2:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                samples.append((nr, nc))

    for (nr, nc) in samples:
        change = (depth_map[r, c] - depth_map[nr, nc]) ** 2
        if change > threshold:
            return float('inf')
        total_depth_change += change

    return total_depth_change / len(samples) if samples else 0

def astar_adaptR_fast_adaptTH_sorted(depth_map, start, goal, base_threshold=1.0, heuristic_weight=1.5):
    open_heap = []
    heapq.heappush(open_heap, (0 + heuristic(start, goal, heuristic_weight), 0, start))
    came_from = {}
    cost_so_far = {}
    visited = set()
    came_from[start] = None
    cost_so_far[start] = 0

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        if current in visited:
            continue
        visited.add(current)
        
        neighbors = get_sorted_neighbors(current, goal)
        for move in neighbors:
            neighbor = (current[0] + move[0], current[1] + move[1])
            if 0 <= neighbor[0] < depth_map.shape[0] and 0 <= neighbor[1] < depth_map.shape[1]:
                new_cost = current_cost + 1 + sample_average_depth_change_adaptTH_adaptR(depth_map, neighbor, 10, base_threshold)
                if new_cost == float('inf'):
                    continue
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal, heuristic_weight)
                    heapq.heappush(open_heap, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

        if not open_heap:
            return "No path found - goal is unreachable with the given threshold."

    return "No path found - exhausted all possibilities."

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


def depth_lpp(depth_img, goal_cxcy):
    # img_to_dep_ratio_x = img.size[0] / depth_img.shape[1]
    # img_to_dep_ratio_y = img.size[1] / depth_img.shape[0]

    depth_img_h, depth_img_w = depth_img.shape

    depth_start_yx_ = (int(1.0 * depth_img.shape[0]), int(0.5 * depth_img.shape[1]))
    depth_goal_yx_ = (int(goal_cxcy[1] * depth_img.shape[0]), int(goal_cxcy[0] * depth_img.shape[1]))
    
    depth_start_yx = validate_point(depth_start_yx_, depth_img_h, depth_img_w, margin=1)
    depth_goal_yx = validate_point(depth_goal_yx_, depth_img_h, depth_img_w, margin=1)
                
    # path = astar(depth_img, depth_start_yx, depth_goal_yx)
    # path = astar_r_fast(depth_img, depth_start_yx, depth_goal_yx, radius=5)
    threshold_sq = 1.0 * 1.0 # meter x meter
    # path = astar_r_fast_thresh(depth_img, depth_start_yx, depth_goal_yx, radius=5, threshold_sq=threshold_sq)
    # path = astar_r_fast_thresh_sorted(depth_img, depth_start_yx, depth_goal_yx, radius=5, threshold_sq=threshold_sq)
    path = astar_adaptR_fast_thresh_sorted(depth_img, depth_start_yx, depth_goal_yx, threshold_sq=threshold_sq,
                                        heuristic_weight=1.0, threshold_depth_limit=10.)
    # path = astar_adaptR_fast_adaptTH_sorted(depth_img, depth_start_yx, depth_goal_yx, base_threshold=threshold_sq,
    #                                         heuristic_weight=2.0)

    if isinstance(path, list):
        # Convert path to numpy array
        # path_array = np.array(path, dtype=np.int32)                
        path_res = True
        path_str = "Path is planned successfully."
        path_yx = path
        path_yx.insert(0, depth_start_yx_)    # add starting point
        n_path_xy = [(item_yx[1]/depth_img_w, item_yx[0]/depth_img_h) for item_yx in path]
        
    else:
        # path_array = np.array((-1, -1), dtype=np.int32)
        path_res = False
        path_str = path
        path_yx = (-1, -1)
        n_path_xy = (-1, -1)

    # Save to dict
    res_dict = {
        'res': path_res,                    # Boolean: True or False
        'reason': path_str,                 # String: Success or Failure with a reason
        'depth_path_yx': path_yx,           # List: (y, x) in depth image w/o normalization
        'depth_start_yx': depth_start_yx,   # pt: start point (y, x) in depth image w/o normalization
        'depth_goal_yx': depth_goal_yx,     # pt: goal point (y, x) in depth image w/o normalization
        'n_path_xy': n_path_xy              # List: (x, y) in depth image w/ normalization
    }

    return res_dict


if __name__ == '__main__':
    # depth image for local path planning
    img_file_wo_ext = 'MP_SEL_003070'
    depth_path = f'/home/yochin/Desktop/PathGuidedVQA_Base/val20k/depth_anything_v2/{img_file_wo_ext}.npy'
    image_path = f'/home/yochin/Desktop/PathGuidedVQA_Base/val20k/original_images/{img_file_wo_ext}.jpg'
    debug_path = '.'
    goal_cxcy = [0.5, 0.5]  # normalized goal position is the center of the image

    # load the depth image
    depth_img = np.load(depth_path)       # height x width

    # load the color image for debugging
    color_img = Image.open(image_path).convert('RGB')     # RGBA to RGB
    whole_width, whole_height = color_img.size

    res_dict = depth_lpp(depth_img, goal_cxcy)

    # res_dict = {
    #     'res': path_res,                    # Boolean: True or False
    #     'reason': path_str,                 # String: Success or Failure with a reason
    #     'depth_path_yx': path_yx,           # List: (y, x) in depth image w/o normalization
    #     'depth_start_yx': depth_start_yx,   # pt: start point (y, x) in depth image w/o normalization
    #     'depth_goal_yx': depth_goal_yx,     # pt: goal point (y, x) in depth image w/o normalization
    #     'n_path_xy': n_path_xy              # List: (x, y) in depth image w/ normalization
    # }

    path_suc = res_dict['res']

    if path_suc:
        path_array = res_dict['n_path_xy']
        
        
        path_array_xy = [[
            np.clip(int(x*whole_width), 0, whole_width-1), 
            np.clip(int(y*whole_height), 0, whole_height-1)
            ] for x, y in path_array]
        
        # save the result images
        path_to_debug = os.path.join(debug_path, f'{img_file_wo_ext}_path.jpg')
        draw_path_on_scaled_image(depth_img, 
                                  res_dict['depth_path_yx'], 
                                  res_dict['depth_start_yx'], 
                                  res_dict['depth_goal_yx'], 
                                  color_img, filename=path_to_debug, radius=20)

        path_to_debug_depth = os.path.join(debug_path, f'{img_file_wo_ext}_d.jpg')
        save_and_visualize_depth_map(depth_img, path_to_debug_depth)
