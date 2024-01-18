import xml.etree.ElementTree as ET
import numpy as np
import pdb


def calculate_iou(box1, box2):
    """
    두 사각형 영역의 IOU를 계산합니다.

    Args:
    box1 (list): 첫 번째 사각형 영역 [x1, y1, x2, y2]
    box2 (list): 두 번째 사각형 영역 [xx1, yy1, xx2, yy2]

    Returns:
    float: 계산된 IOU 값
    """
    # 각 사각형의 (x1, y1, x2, y2) 추출
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2

    # 교차 영역 (intersection)의 (x, y) 좌표 계산
    x_left = max(x1, xx1)
    y_top = max(y1, yy1)
    x_right = min(x2, xx2)
    y_bottom = min(y2, yy2)

    # 교차 영역이 있는지 확인
    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0.0
    else:
        # 교차 영역의 넓이
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 각 사각형의 영역
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (xx2 - xx1) * (yy2 - yy1)

    # 합집합 영역
    union_area = area_box1 + area_box2 - intersection_area

    # IOU 계산
    iou = intersection_area / union_area

    return iou, intersection_area, area_box1, area_box2, union_area


def reorigin_bbox_point(list_bboxes, point_xy, boundary_bbox):
    res_list_reorigin = []

    rescale_x = 1.0 / float(boundary_bbox[2] - boundary_bbox[0])
    rescale_y = 1.0 / float(boundary_bbox[3] - boundary_bbox[1])

    for label_name, bbox, score in list_bboxes:
            bbox_reorigin = [(bbox[0]-boundary_bbox[0]) * rescale_x, 
                             (bbox[1]-boundary_bbox[1]) * rescale_y, 
                             (bbox[2]-boundary_bbox[0]) * rescale_x, 
                             (bbox[3]-boundary_bbox[1]) * rescale_y]
            bbox_reorigin_clamp = [clamp(item, 0., 1.) for item in bbox_reorigin]
            res_list_reorigin.append([label_name, bbox_reorigin_clamp, score])

    res_xy_reorigin = [clamp((point_xy[0]-boundary_bbox[0]) * rescale_x, 0., 1.), 
                       clamp((point_xy[1]-boundary_bbox[1]) * rescale_y, 0., 1.)]


    return res_list_reorigin, res_xy_reorigin


def remove_outer_bbox(list_bboxes, boundary_bbox, thresh_intersect_over_bbox=0.5):
    res_list_original = []

    for label_name, bbox, score in list_bboxes:
        # print('bbox: ', bbox)
        # print('boundary_bbox: ', boundary_bbox)
        iou, intersection_area, area_box1, area_box2, union_area = calculate_iou(bbox, boundary_bbox)

        if (intersection_area / area_box1) > thresh_intersect_over_bbox:
            res_list_original.append([label_name, bbox, score])

    return res_list_original


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
    
    if pil_image is not None:
        real_width, real_height = pil_image.size

    for point_cxcy in list_points_on_path:
        new_box = [point_cxcy[0] - sub_half_width, point_cxcy[1] - sub_half_height, 
                   point_cxcy[0] + sub_half_width, point_cxcy[1] + sub_half_height]
        
        adjusted_box = adjust_rectangle_within_bounds(new_box, width=whole_width, height=whole_height)
        list_boxes_on_path.append(adjusted_box)

        if list_cropped_images is not None:
            adjusted_box_real_size = [adjusted_box[0]*real_width, adjusted_box[1]*real_height, adjusted_box[2]*real_width, adjusted_box[3]*real_height]
            list_cropped_images.append(pil_image.crop(adjusted_box_real_size))

    return list_boxes_on_path, list_points_on_path, list_cropped_images

