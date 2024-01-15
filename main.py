import os
from PIL import Image
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# assisted by ChatGPT4

def main():
    # 이미지가 저장된 폴더 경로
    image_path = 'samples/images'
    anno_path = 'samples/anno_aihub'

    # 폴더 내의 모든 파일 목록을 가져옴
    files = os.listdir(image_path)

    # 이미지 파일들만 필터링
    image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 각 이미지 파일에 대해
    for img_file in image_files:
        # 이미지 파일의 전체 경로
        img_path = os.path.join(image_path, img_file)
        # XML 파일의 전체 경로 (파일 이름은 같지만 확장자만 xml로 변경)
        xml_path = os.path.join(anno_path, os.path.splitext(img_file)[0] + '.xml')


        # 이미지를 열고
        img = Image.open(img_path)
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # XML 파일을 파싱하여 Bounding Box 정보를 가져옴
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)

            # Bounding Box를 이미지에 그림
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # 이미지 및 Bounding Box 표시
        plt.axis('off')
        plt.title(img_file)
        plt.show()


        # 1. Split input images into cropped images along with the goal path (yochin)
        
        # 2. generate answers 1 and 2 using LLM (byungok.han)

        # 3. merge answers into the final answer (later)





    

    return

if __name__ == '__main__':
    main()