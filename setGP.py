import xml.etree.ElementTree as ET
import numpy as np
import pdb


#   calculate_divided_points between two points
#   implemented by ChatGPT 4
def calculate_divided_points(x1, y1, x2, y2, num_divisions):
    # 두 점 사이의 거리 계산
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # 나누는 간격 계산 (num_divisions가 선을 나누는 횟수이므로, 간격은 num_divisions + 1로 나눔)
    interval = distance / (num_divisions + 1)

    # 두 점을 연결하는 선의 방향 벡터 계산
    dx = (x2 - x1) / (num_divisions + 1)
    dy = (y2 - y1) / (num_divisions + 1)

    # 각 점의 좌표 계산
    points = [[x1 + i * dx, y1 + i * dy] for i in range(1, num_divisions + 1)]

    return points


#   adjust bbox out of the image
#   implemented by ChatGPT 4
def adjust_rectangle_within_bounds(input_box, width, height):
    x1, y1, x2, y2 = input_box

    # 사각형의 폭과 높이 계산
    rect_width = x2 - x1
    rect_height = y2 - y1

    # 사각형이 경계값을 벗어나지 않도록 조정
    if x1 < 0:
        x1 = 0
        x2 = rect_width
    elif x2 > width:
        x1 = width - rect_width
        x2 = width

    if y1 < 0:
        y1 = 0
        y2 = rect_height
    elif y2 > height:
        y1 = height - rect_height
        y2 = height

    return [x1, y1, x2, y2]
            

#   implemented by ChatGPT 4
def clamp(n, min_value, max_value):
    """
    주어진 숫자 n을 min_value와 max_value 사이로 제한합니다.

    Args:
    n (float or int): 제한할 숫자
    min_value (float or int): 최소값
    max_value (float or int): 최대값

    Returns:
    float or int: 제한된 값
    """
    return max(min_value, min(n, max_value))


# read xml and return the bbox information as a list
# one bbox information has [label_name, [x_min, y_min, x_max, y_max], score] info.
def read_anno(path_to_xml, rescaling=False, filtering_score=-1.0):
    res_bboxes = []

    # XML 파일을 파싱하여 Bounding Box 정보를 가져옴
    tree = ET.parse(path_to_xml)
    root = tree.getroot()

    if rescaling:
        w_org = float(root.find('size').find('width').text)
        h_org = float(root.find('size').find('height').text)

    for obj in root.iter('object'):
        bbox = obj.find('bndbox')
        x_min = int(bbox.find('xmin').text)
        y_min = int(bbox.find('ymin').text)
        x_max = int(bbox.find('xmax').text)
        y_max = int(bbox.find('ymax').text)

        if rescaling:
            x_min = clamp((float(x_min) / w_org), 0, 1)
            x_max = clamp((float(x_max) / w_org), 0, 1)
            y_min = clamp((float(y_min) / h_org), 0, 1)
            y_max = clamp((float(y_max) / h_org), 0, 1)

        label = obj.find('name').text

        if obj.find('score') is not None:
            score = float(obj.find('score').text)
        else:
            score = 1.0

        if score > filtering_score:
            # add to the return list
            res_bboxes.append([label, [x_min, y_min, x_max, y_max], score])

    return res_bboxes



# filter the object from list_bboxes info, then generate the list of label_name and goal positions (x, y)
# return: list of [label_name, [cx, cy]]
def get_gp(list_bboxes, list_goal_objects):
    res_gps = []

    for bbox_info in list_bboxes:
        label = bbox_info[0]
        bbox = bbox_info[1]

        if label in list_goal_objects:
            res_gps.append([label, [(float(bbox[0]+bbox[2])/2.), float(bbox[1]+bbox[3])/2.]])

    return res_gps


# split images into sub images
def split_images(goal_point_cxcy, whole_width, whole_height, pil_image=None, sub_image_ratio=0.5, num_divisions=1):
    # sub-image size
    sub_half_width = (whole_width * sub_image_ratio / 2.0)
    sub_half_height = (whole_height * sub_image_ratio / 2.0)

    # generate a starting center point
    start_cx = (whole_width / 2.0)
    start_cy = whole_height - sub_half_height

    
    # start_cxcy과 goal_point_cxcy을 연결하는 선을 num_divisions번 자른 점들의 위치 계산
    list_division_points = calculate_divided_points(start_cx, start_cy, goal_point_cxcy[0], goal_point_cxcy[1], num_divisions=num_divisions)

    list_points_on_path = [[start_cx, start_cy]]
    list_points_on_path.extend(list_division_points)
    list_points_on_path.append(goal_point_cxcy)

    # from point to box
    list_cropped_images = []
    list_boxes_on_path = []    
    for point_cxcy in list_points_on_path:
        new_box = [point_cxcy[0] - sub_half_width, point_cxcy[1] - sub_half_height, 
                   point_cxcy[0] + sub_half_width, point_cxcy[1] + sub_half_height]
        
        adjusted_box = adjust_rectangle_within_bounds(new_box, width=whole_width, height=whole_height)
        list_boxes_on_path.append(adjusted_box)

        if list_cropped_images is not None:
            list_cropped_images.append(pil_image.crop(adjusted_box))

    return list_boxes_on_path, list_points_on_path, list_cropped_images

