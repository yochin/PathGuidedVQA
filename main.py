import os
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pdb
import os
import openai
import base64

from setGP import read_anno, get_gp, split_images

OPENAI_API_KEY = "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"



def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def describe_all_bboxes_with_chatgpt(image_path, bboxes, goal_label_cxcy):
    # 이미지를 base64로 인코딩
    encoded_image = encode_image_to_base64(image_path)

    # 각 바운딩 박스에 대한 설명 구성
    bbox_descriptions = [f"{label} at ({x_min}, {y_min}, {x_max}, {y_max})" for label, (x_min, y_min, x_max, y_max), _ in bboxes]
    bbox_list_str = ", ".join(bbox_descriptions)
    goal_label, goal_cxcy = goal_label_cxcy
    dest_descriptions = f"{goal_label} at ({goal_cxcy[0]}, {goal_cxcy[1]})"

    # GPT-4에 대한 프롬프트 구성
    prompt = f"""
        Obstacle Name at (bounding box): {bbox_list_str}.
        Destination Name at (point): {dest_descriptions}.
        Describe the following obstacles to the destination in a natural and simple way for a visually impaired person in Korean.
        Don't talk about detailed image coordinates.

    """
    print("[PROMPT]: ", prompt)
    # OpenAI API 키 설정 (환경 변수에서 가져옴)
    openai.api_key = OPENAI_API_KEY
    completion = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        #messages=[
        #    {
        #        "role": "user",
        #        "content": prompt,
        #    },
        messages=[
            {"role": "system", "content": "This is an image-based task."},
            {"role": "user", "content": encoded_image}, #, "mimetype": "image/jpeg"
            {"role": "user", "content": prompt},
        ],
        #max_tokens=1000,
    )

    answer = completion.choices[0].message.content

    print("[ANSWER]: ", answer)

    return answer


# Read text, return list
# Coded by ChatGPT 4
def read_text(file_path):
    lines = []

    # 파일을 열고 각 줄을 읽어 리스트에 추가합니다
    with open(file_path, 'r') as file:
        for line in file:
            # strip() 함수를 사용하여 줄바꿈 문자를 제거합니다
            lines.append(line.strip())

    return lines
    

# Assisted by ChatGPT 4
def main():
    # 이미지가 저장된 폴더 경로
    image_path = 'samples/images'
    anno_path1 = 'samples/anno_aihub'
    anno_path2 = 'samples/anno_toomuch'
    anno_path_gt = 'samples/anno_gt'
    label_path_gt = 'samples/default_labels.txt'

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
        print(f'processing {img_path}...\n')
        # XML 파일의 전체 경로 (파일 이름은 같지만 확장자만 xml로 변경)
        xml_path1 = os.path.join(anno_path1, os.path.splitext(img_file)[0] + '.xml')
        xml_path2 = os.path.join(anno_path2, os.path.splitext(img_file)[0] + '.xml')
        xml_path_gt = os.path.join(anno_path_gt, os.path.splitext(img_file)[0] + '.xml')
        

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
        bboxes1 = read_anno(xml_path1, rescaling=True, filtering_score=0.8) # list of [label_name, [x_min, y_min, x_max, y_max], score]
        bboxes2 = read_anno(xml_path2, rescaling=True, filtering_score=0.8)
        bboxes = bboxes1
        bboxes.extend(bboxes2)

        bboxes_gt = read_anno(xml_path_gt, rescaling=True)
        bboxes.extend(bboxes_gt)

        # 1.2. set goal position
        list_labels_gps = get_gp(bboxes, list_goal_names)  # list of [label_name, [cx, cy]]

        # 1.3. split images into sub-images
        for goal_label_cxcy in list_labels_gps:
            print('the goal info:', goal_label_cxcy)
            goal_label, goal_cxcy = goal_label_cxcy
            list_subimage_boxes_on_path, list_subimage_centerpoints_on_path, list_cropped_images = split_images(goal_cxcy, 1.0, 1.0, pil_image=img, sub_image_ratio=0.5, num_divisions=1)
        
    # #     # 2. generate answers 1 and 2 using LLM (byungok.han)
    
            # 결과 문장 생성
            description = describe_all_bboxes_with_chatgpt(img_path, bboxes, goal_label_cxcy)

    #     # 3. merge answers into the final answer (later)


            # 4. draw all
            # draw start, mid, and goal points and boxes
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype('arial.ttf', size=40)
            radius = 20
            
            for mid_box, mid_point in zip(list_subimage_boxes_on_path, list_subimage_centerpoints_on_path):
                mid_point_draw = [mid_point[0] * whole_width, mid_point[1] * whole_height]
                mid_box_draw = [mid_box[0] * whole_width, mid_box[1] * whole_height, mid_box[2] * whole_width, mid_box[3] * whole_height]

                mid_point_lu_draw = (int(mid_point_draw[0]-radius), int(mid_point_draw[1]-radius))
                mid_point_rd_draw = (int(mid_point_draw[0]+radius), int(mid_point_draw[1]+radius))
                draw.ellipse([mid_point_lu_draw, mid_point_rd_draw], fill='red')
                draw.rectangle(mid_box_draw, outline='red', width=4)

            # draw detection results
            print(bboxes)
            for label_name, bbox, score in bboxes:
                # Bounding Box를 이미지에 그림
                draw_bbox = [bbox[0] * whole_width, bbox[1] * whole_height, bbox[2] * whole_width, bbox[3] * whole_height]
                draw.rectangle(draw_bbox, outline='yellow', width=2)
                draw.text(draw_bbox[:2], label_name, fill='white', font=font)

            # 이미지 및 Bounding Box 표시
            plt.imshow(img)
            plt.axis('off')
            plt.title(img_file + f', goal_label:{goal_label}')
            plt.show()



    

    return

if __name__ == '__main__':
    main()