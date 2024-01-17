import os
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pdb

from setGP import read_anno, get_gp, split_images


import requests
import json



def describe_all_bboxes_with_chatgpt(bboxes):
    # 모든 장애물 정보를 하나의 문자열로 구성
    bbox_descriptions = [f"{label} at ({x_min}, {y_min}, {x_max}, {y_max})" for label, (x_min, y_min, x_max, y_max), _ in bboxes]
    bbox_list_str = ", ".join(bbox_descriptions)

    prompt = f"Describe the following obstacles in a natural and detailed way for a visually impaired person: {bbox_list_str}."

    # ChatGPT API 호출
    response = requests.post(
        "https://api.openai.com/v1/engines/davinci-codex/completions",
        headers={
            "Authorization": f"Bearer YOUR_API_KEY",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "prompt": prompt,
            "max_tokens": 200  # 필요에 따라 조정
        })
    )

    # 응답 처리
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["text"].strip()
    else:
        return f"Error: Unable to describe obstacles. Response Code: {response.status_code}"




# Assisted by ChatGPT 4

def main():
    # 이미지가 저장된 폴더 경로
    image_path = 'samples/images'
    anno_path1 = 'samples/anno_aihub'
    anno_path2 = 'samples/anno_toomuch'

    # 폴더 내의 모든 파일 목록을 가져옴
    files = os.listdir(image_path)

    # 이미지 파일들만 필터링
    image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 각 이미지 파일에 대해
    for img_file in image_files:
        # 이미지 파일의 전체 경로
        img_path = os.path.join(image_path, img_file)
        # XML 파일의 전체 경로 (파일 이름은 같지만 확장자만 xml로 변경)
        xml_path1 = os.path.join(anno_path1, os.path.splitext(img_file)[0] + '.xml')
        xml_path2 = os.path.join(anno_path2, os.path.splitext(img_file)[0] + '.xml')

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


        # 0. Definition
        list_goal_names = ['stairs', 'door', 'elevator']
        whole_width, whole_height = img.size

        # 1. Split input images into cropped images along with the goal path (yochin)
        # 1.1. read annotation and convert into bboxes with label info.
        # XML 파일을 파싱하여 Bounding Box 정보를 가져옴
        bboxes1 = read_anno(xml_path1) # list of [label_name, [x_min, y_min, x_max, y_max], score]
        bboxes2 = read_anno(xml_path2)
        bboxes = bboxes1
        bboxes.extend(bboxes2)

        # 1.2. set goal position
        list_labels_gps = get_gp(bboxes, list_goal_names)  # list of [label_name, [cx, cy]]

        # 1.3. split images into sub-images
        for goal_label_cxcy in list_labels_gps:
            goal_label, goal_cxcy = goal_label_cxcy
            list_boxes_on_path, list_points_on_path, list_cropped_images = split_images(goal_cxcy, whole_width, whole_height, pil_image=img, sub_image_ratio=0.5, num_divisions=1)

        
    #     # 2. generate answers 1 and 2 using LLM (byungok.han)
    
            # 결과 문장 생성
            description = describe_all_bboxes_with_chatgpt(bboxes_example)
            print(description)    

    #     # 3. merge answers into the final answer (later)


            # 4. draw all
            # draw start, mid, and goal points and boxes
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype('arial.ttf', size=40)
            
            for mid_point, mid_box in zip(list_points_on_path, list_boxes_on_path):
                draw.point(mid_point)
                draw.rectangle(mid_box, outline='red', width=4)

            # draw detection results
            for label_name, bbox, score in bboxes:
                # Bounding Box를 이미지에 그림
                draw.rectangle(bbox, outline='yellow', width=2)
                draw.text(bbox[:2], label_name, fill='white', font=font)

            # 이미지 및 Bounding Box 표시
            plt.imshow(img)
            plt.axis('off')
            plt.title(img_file + f', goal_label:{goal_label}')
            plt.show()



    

    return

if __name__ == '__main__':
    main()