import os
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from depth_anything_wrapper import depth_anything
from generate_image_data import save_and_visualize_depth_map

def print_intrinsic_from_FOV():
    imageWidth = 1920
    imageHeight = 1080
    cameraFOV_H = 84
    cameraFOV_V = 54

    # x, y축 FOV 계산 (단위: radians)
    fov_x = math.radians(cameraFOV_H)
    # fov_y = 2 * math.atan(imageHeight/imageWidth * math.tan(fov_x/2))
    fov_y = math.radians(cameraFOV_V)

    # Intrinsic Matrix 구함
    fx = (imageWidth/2) / math.tan(fov_x/2)
    fy = (imageHeight/2) / math.tan(fov_y/2)
    fx_r = (1./2.) / math.tan(fov_x/2)
    fy_r = (1./2.) / math.tan(fov_y/2)
    cx = imageWidth/2
    cy = imageHeight/2

    print(fx, fy, cx, cy, fx_r, fy_r)


def create_circular_mask(image, points, radius):
    # Initialize mask with zeros (black)
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Draw white filled circles on the mask at each point
    for point in points:
        cv2.circle(mask, center=tuple(point), radius=radius, color=(255, 255, 255), thickness=-1)
    
    return mask


def gen_mask_v1(cv_img, path_array_xy):
    img_h, img_w, _ = cv_img.shape

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


def generate_mask(cv_img, path_array_xy):
    points = np.array(path_array_xy, dtype=np.int32)

    mask_left, mask_right = gen_mask_v1(cv_img, path_array_xy)

    # mask_path = create_thick_line_mask(cv_img, points, thickness=line_thickness)

    return {'L': mask_left, 
            'R': mask_right, 
            # 'P': mask_path
            }

def create_circle_mask(cv_img, target_point, r):
    height, width = cv_img.shape[:2]

    # Create a mask
    mask = np.zeros_like(cv_img, dtype=np.uint8)
    
    # Draw a filled circle (white) on the mask
    cv2.circle(mask, target_point, r, (255, 255, 255), thickness=-1)
        
    return mask


def create_trapezoid_mask(cv_img, target_point, r):
    height, width = cv_img.shape[:2]
    x, y = target_point

    # Define the trapezoid points
    top_left = (0, 0)
    top_right = (width, 0)
    bottom_left = (max(x - r, 0), y)
    bottom_right = (min(x + r, width), y)
    
    # Create a mask
    mask = np.zeros_like(cv_img, dtype=np.uint8)
    
    # Define the trapezoid as a polygon
    trapezoid = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    
    # Fill the trapezoid area in the mask
    cv2.fillPoly(mask, [trapezoid], (255, 255, 255))
        
    return mask
    

def save_debug_masked_image(img_path, cv_img, dict_masks, debug_img_folder):
    file_with_ext = os.path.split(img_path)[1]
    filename, ext = os.path.splitext(file_with_ext)

    if not os.path.exists(debug_img_folder):
        os.makedirs(debug_img_folder)

    for key, mask in dict_masks.items():
        path_to_save = os.path.join(debug_img_folder, f'{filename}_{key}{ext}')
        if mask is None:
            debug_image = cv_img.copy()
        else:
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
    
    # Create a mask where depth is greater than the average depth
    mask = np.zeros_like(depth_image, dtype=np.uint8)
    
    # Calculate the average depth in the region of interest
    avg_depth = np.max(roi)
    thresh_depth = avg_depth + buffer_dist
    
    if buffer_dist > 0:
        mask[depth_image < thresh_depth] = 255
    else:
        mask[depth_image > thresh_depth] = 255

    mask = np.stack((mask,)*3, axis=-1)
    
    return mask

def is_within_mask(mask, bbox_info, threshold=0.1):    
    _, box, _ = bbox_info
    x1, y1, x2, y2 = box

    h, w, _ = mask.shape

    x1 = int(x1 * w)
    x2 = int(x2 * w)
    
    y1 = int(y1 * h)
    y2 = int(y2 * h)

    w = x2 - x1
    h = y2 - y1

    bbox_region = mask[y1:y2, x1:x2, 0]
    overlap_area = np.sum(bbox_region == 255)
    bbox_area = w * h

    return (overlap_area / bbox_area) >= threshold

def calculate_boundaries(x, y, depth, physical_width, fx, fy, image_width):
    """
    주어진 위치와 깊이에 기반하여 수평 경계의 x 좌표를 계산합니다.
    """    
    # pixel_width = int((physical_width / depth) * ((fx + fy) / 2))
    pixel_width = int((physical_width / depth) * fx)
    left_x = max(x - pixel_width, 0)
    right_x = min(x + pixel_width, image_width - 1)
    # print(x, y, depth, left_x, right_x)

    return left_x, right_x

def smooth_boundaries(boundaries, smoothing_factor=5):
    lefts = np.convolve([b[0] for b in boundaries], np.ones(smoothing_factor) / smoothing_factor, mode='same')
    rights = np.convolve([b[1] for b in boundaries], np.ones(smoothing_factor) / smoothing_factor, mode='same')

    return list(zip(lefts.astype(int), rights.astype(int)))

def mask_dest_depth_lr(depth_image, gp_xy, camera_intrinsics, physical_half_width):
    fx, fy, cx, cy = camera_intrinsics
    mask = np.zeros_like(depth_image, dtype=np.uint8)
    mask_c = np.zeros_like(depth_image, dtype=np.uint8)
    image_width = depth_image.shape[1]

    x, y = gp_xy
    boundaries = calculate_boundaries(int(x), int(y), depth_image[int(y), int(x)], physical_half_width, fx, fy, image_width)

    # 실제 세계에서의 반경을 픽셀 반경으로 변환
    radius = int((boundaries[1] - boundaries[0]) / 2)    
    mask[:, gp_xy[0]-radius:gp_xy[0]+radius] = 255

    cv2.circle(mask_c, gp_xy, radius, (255,), -1)

    # masked_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)
    mask = np.stack((mask,)*3, axis=-1)
    mask_c = np.stack((mask_c,)*3, axis=-1)

    return mask, mask_c


def mask_depth_image_using_path(depth_image, path_points, camera_intrinsics, physical_half_width):
    """
    경로의 좌우 경계를 계산하고 연결하여 마스킹 영역을 생성합니다.
    
    :param depth_image: 깊이 이미지 (numpy array)
    :param path_points: 경로 좌표와 깊이 [(x, y, depth), ...]
    :param camera_intrinsics: 카메라 내부 파라미터 (fx, fy, cx, cy)
    :param physical_width: 경로 너비의 절반 (미터)
    :return: 마스킹된 이미지
    """
    fx, fy, cx, cy = camera_intrinsics
    mask = np.zeros_like(depth_image, dtype=np.uint8)
    image_width = depth_image.shape[1]

    # 모든 경로 포인트에 대해 왼쪽과 오른쪽 경계 계산    
    boundaries = [calculate_boundaries(int(x), int(y), depth_image[int(y), int(x)], physical_half_width, fx, fy, image_width) for x, y in path_points]

    # 경계 평활화 적용
    # smoothed_boundaries = smooth_boundaries(boundaries)
    smoothed_boundaries = boundaries

    # 경계선 연결을 위해 각 세그먼트를 직사각형으로 그립니다.
    for i in range(len(smoothed_boundaries) - 1):
        (left_x1, right_x1), (left_x2, right_x2) = smoothed_boundaries[i], smoothed_boundaries[i + 1]
        y1, y2 = path_points[i][1], path_points[i + 1][1]

        # 직사각형의 네 꼭짓점 정의
        top_left = (left_x1, min(y1, y2))
        top_right = (right_x1, min(y1, y2))
        bottom_left = (left_x2, max(y1, y2))
        bottom_right = (right_x2, max(y1, y2))

        # 두 꼭짓점을 이용하여 직사각형을 그림
        cv2.rectangle(mask, top_left, bottom_right, (255,), thickness=cv2.FILLED)

        # print(top_left, bottom_right)

    # 실제 세계에서의 반경을 픽셀 반경으로 변환
    radius = int((smoothed_boundaries[-1][1] - smoothed_boundaries[-1][0]) / 2)
    cv2.circle(mask, path_points[-1], radius, (255,), -1)

    # masked_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)
    mask = np.stack((mask,)*3, axis=-1)

    return mask

def calculate_pixel_radius(depth, physical_radius, fx, fy):
    """
    주어진 물리적 반경(미터)을 픽셀 반경으로 변환합니다.
    """
    # 물리적 반경을 픽셀 반경으로 변환 (근사적으로 단순화)
    pixel_radius = int((physical_radius / depth) * ((fx + fy) / 2))
    return pixel_radius

def mask_depth_image_path_radius(depth_image, path_points, camera_intrinsics, physical_radius):
    """
    깊이에 따라 경로 주변을 일정 비율로 마스킹합니다.
    
    :param depth_image: 깊이 이미지 (numpy array)
    :param path_points: 경로 좌표의 리스트 [(x, y), ...]
    :param camera_intrinsics: 카메라 내부 파라미터 (fx, fy, cx, cy)
    :param physical_radius: 마스킹을 적용할 물리적 거리 (미터)
    :return: 마스킹된 이미지
    """
    fx, fy, cx, cy = camera_intrinsics
    mask = np.zeros_like(depth_image, dtype=np.uint8)

    for x, y in path_points:
        y = int(y)
        x = int(x)
        depth = depth_image[y, x]
        
        # 실제 세계에서의 반경을 픽셀 반경으로 변환
        radius = calculate_pixel_radius(depth, physical_radius, fx, fy)
        cv2.circle(mask, (x, y), radius, (255,), -1)
    
    # masked_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)
    mask = np.stack((mask,)*3, axis=-1)

    return mask

def get_intrinsic_ratio(filename):
    # default
    fx_r = 0.6
    fy_r = 0.6
    cx_r = 0.5
    cy_r = 0.5

    # if 'MP_' in filename:   # MobilePhone from image sensor size
    #     fx_r = 4.3 / 6.17       # 0.6969
    #     fy_r = 4.3 / 4.55       # 0.9451
    
    # elif 'ZED' in filename: # ZED from func call
    #     fx_r = 700. / 1280.     # 0.5469
    #     fy_r = 700. / 720.      # 0.9722
    if 'MP_' in filename:   # MobilePhone from FOV
        correction_factor = 1.5 * 3.0/2.5     # 3 meter -> 2.5 m
        fx_r = 0.6285861494594773 * correction_factor
        fy_r = 1.1174864879279598 * correction_factor
    elif 'ZED' in filename: # ZED from FOV
        correction_factor = 1.5 * 3.0/1.5     # 3 meter -> 1.5 m
        fx_r = 0.5553062574145965 * correction_factor
        fy_r = 0.9813052527525753 * correction_factor
    elif '_R' in filename: # ricoh from calibration
        correction_factor = 1.5 * 3.0/1.0     # 3 meter -> 1.0 m
        fx_r = 0.49078759599228894 * correction_factor  # 706.73413822889609 / 1440 
        fy_r = 0.7024368112553278 * correction_factor   # 758.631756155754 / 1080
    elif '_GOPR' in filename: # gopro from calibration
        correction_factor = 2.0 * 3.0/1.5     # 3 meter -> 1.5 m
        fx_r = 0.4830019884684673 * correction_factor   # 2689.3550717924259 / 5568
        fy_r = 0.5517242397258202 * correction_factor   # 2688.0004959441958 / 4872
    else:
        raise AssertionError('Cannot get camera intrinsic from unsupported filename.')
        # fx_r = 0.6
        # fy_r = 0.95

    return fx_r, fy_r, cx_r, cy_r

def interpolate_points(x1, y1, x2, y2, n):
    # x1, y1에서 x2, y2까지 n개의 점을 생성
    x_values = np.linspace(x1, x2, n)
    y_values = np.linspace(y1, y2, n)
    # 각 x값과 y값을 쌍으로 묶어 배열 형태로 반환
    points = np.column_stack((x_values, y_values))

    return points.tolist()


if __name__ == '__main__':
    # print_intrinsic_from_FOV()    # get_intrinsic_from_FOV

    test_image_path = 'samples_masking/images'
    output_path_debug = 'samples_masking/debugs'
    goal_cxcy = [0.5, 0.75]
    dst_depth_meter = 5.0
    dst_circle_ratio = 0.05

    # depth anything v2
    dep_any = depth_anything('vitl', 
                             'vkitti', 
                             80)

    # 폴더 내의 모든 파일 목록을 가져옴
    files = os.listdir(test_image_path)

    # 이미지 파일들만 필터링
    image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
    image_files.sort()  # return itself

    # load all images from camera types
    for img_file in image_files:
        print(img_file)
        # 이미지 파일의 전체 경로
        img_path = os.path.join(test_image_path, img_file)
        img_file_wo_ext = os.path.splitext(img_file)[0]

        img = Image.open(img_path).convert('RGB')
        whole_width, whole_height = img.size

        cv_org_img = cv2.imread(img_path)
        cv_org_img_pt = cv_org_img.copy()


        depth_path = os.path.join(test_image_path, img_file_wo_ext + '.npy')

        if os.path.exists(depth_path):
            depth_image = np.load(depth_path)       # h x w
        else:
            depth_image = dep_any.infer_image(cv_org_img)
            np.save(depth_path, depth_image)

        # depth_image = cv2.resize(depth_image, dsize=(whole_width, whole_height), interpolation=cv2.INTER_CUBIC)

        # # vkitti intrsic
        # fx_vkitti = 725.
        # fy_vkitti = 725.
        # w_vkitti = 1242.
        # h_vkitti = 375.
        # cx_vkitti = 620.5
        # cy_vkitti = 187.0

        # # default intrinsic in depth anything
        # fx_vkitti = 256*0.6
        # fy_vkitti = 256*0.6
        # w_vkitti = 256
        # h_vkitti = 256
        # cx_vkitti = 256*0.5
        # cy_vkitti = 256*0.5

        # to remove the bias in depth anything in metric version
        # assume the bottom depth as 1 meter

        # generate mask using depth image on the path with physical radius
        fx_r, fy_r, cx_r, cy_r = get_intrinsic_ratio(img_file_wo_ext)
        camera_intrinsics = (whole_width*fx_r, whole_height*fy_r, whole_width*cx_r, whole_height*cy_r)  # fx, fy, cx, cy

        # scale = (fx_r * fy_r) / (0.6 * 0.6)

        # depth_image = depth_image * scale

        # dist_at_bottom = np.median(depth_image[whole_height-10:, :])
        # bias = 1.0 - dist_at_bottom
        # depth_image = depth_image + bias

        # draw depth meter
        gp_pos_xy = [int(goal_cxcy[0]*whole_width), int(goal_cxcy[1]*whole_height)]
        depth_meter_at_gp = depth_image[gp_pos_xy[1], gp_pos_xy[0]]

        
        TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
        TEXT_SCALE = 1.5
        TEXT_THICKNESS = 2
        TEXT = f"{depth_meter_at_gp:.2f}"

        text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (int(gp_pos_xy[0] - text_size[0] / 2), int(gp_pos_xy[1] + text_size[1] / 2))


        cv2.circle(cv_org_img_pt, gp_pos_xy, 10, (255, 0, 0), -1)
        cv2.putText(cv_org_img_pt, TEXT, text_origin, TEXT_FACE, TEXT_SCALE, (127,255,127), TEXT_THICKNESS, cv2.LINE_AA)

        cv2.circle(cv_org_img, gp_pos_xy, 10, (255, 0, 0), -1)
        cv2.putText(cv_org_img, TEXT, text_origin, TEXT_FACE, TEXT_SCALE, (127,255,127), TEXT_THICKNESS, cv2.LINE_AA)



        
        path_array = interpolate_points(0.5, 1.0, goal_cxcy[0], goal_cxcy[1], 500)
        depth_start_yx = [int(1.0*whole_height), int(0.5*whole_width)]
        depth_goal_yx = [int(goal_cxcy[1]*whole_height), int(goal_cxcy[0]*whole_width)]

        path_array_xy = [[
            np.clip(int(x*whole_width), 0, whole_width-1), 
            np.clip(int(y*whole_height), 0, whole_height-1)
            ] for x, y in path_array]
        

        # generate masks with left all and right all masks
        dict_masks = generate_mask(cv_org_img_pt, path_array_xy)    # ['L', 'R'], all left and right area along with path line.

        # generate mask of destination (not masked rgb image)
        # dict_masks['D'] = create_trapezoid_mask(cv_org_img_pt, [int(goal_cxcy[0]*whole_width), int(goal_cxcy[1]*whole_height)+50], r=50)
        depth_mask_c10 = create_circle_mask(cv_org_img_pt, [int(goal_cxcy[0]*whole_width), int(goal_cxcy[1]*whole_height)], r=int(whole_width*dst_circle_ratio))

        # generate mask using depth image (remove far objects over the destination point)
        depth_mask = create_depth_mask(depth_image, [int(goal_cxcy[0]*whole_width), int(goal_cxcy[1]*whole_height)], 
                                        radius=25, buffer_dist=5.)

        if True:    # remove befind and front of the destination
            depth_mask_behind = create_depth_mask(depth_image, [int(goal_cxcy[0]*whole_width), int(goal_cxcy[1]*whole_height)], 
                                        radius=25, buffer_dist=dst_depth_meter)
            depth_mask_front = create_depth_mask(depth_image, [int(goal_cxcy[0]*whole_width), int(goal_cxcy[1]*whole_height)], 
                                        radius=25, buffer_dist=-dst_depth_meter)
            depth_mask_region = cv2.bitwise_and(depth_mask_behind, depth_mask_front)

            depth_mask_lr, depth_mask_c = mask_dest_depth_lr(depth_image, [int(goal_cxcy[0]*whole_width), int(goal_cxcy[1]*whole_height)], 
                                                            camera_intrinsics, physical_half_width=1)
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
        depth_mask_near_path = mask_depth_image_using_path(depth_image, path_array_xy, camera_intrinsics, physical_half_width=1)

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

        for key, value in dict_masks.items():
            # for L, R, Dest
            dict_masks[key] = cv2.bitwise_and(dict_masks[key], depth_mask)  # remove too far region

            if key not in ['D', 'P']:    # 'L', 'R', 'P'
                # dict_masks[key] = cv2.bitwise_and(dict_masks[key], depth_mask_near_path_4m)
                dict_masks[key] = cv2.bitwise_and(dict_masks[key], ~dict_masks['P'])

        # dict_masks includes 'L', 'R', 'D', 'P'
        # save_debug_masked_image(img_path, cv_org_img_pt, dict_masks, output_path_debug)
        save_debug_masked_image(img_path, cv_org_img, dict_masks, output_path_debug)

        # if dst_draw_circle:
        #     save_debug_masked_image(img_path, cv_org_img_circle, {'D': None}, output_path_debug)

        # if dst_draw_point:
        #     if not dst_masking_depth:
        #         save_debug_masked_image(img_path, cv_org_img_pt, {'D': None}, output_path_debug)
        #     else:
        #         save_debug_masked_image(img_path, cv_org_img_pt, {'D': dict_masks['D']}, output_path_debug)

        # if dst_draw_circle is False and dst_draw_point is False and dst_masking_circle is False and dst_masking_depth is False:
        #     save_debug_masked_image(img_path, cv_org_img, {'D': None}, output_path_debug)

        path_to_debug_depth = os.path.join(output_path_debug, f'{img_file_wo_ext}_d.jpg')
        save_and_visualize_depth_map(depth_image, path_to_debug_depth)