import os
import shutil

def copy_file(source_folder, destination_folder, file_name, do_copy=True):
    """
    주어진 파일명을 사용하여 소스 폴더에서 파일을 찾고, 찾은 경우 대상 폴더로 복사합니다.

    Parameters:
    - source_folder: 검색을 시작할 폴더의 경로
    - destination_folder: 파일을 복사할 대상 폴더의 경로
    - file_name: 검색하고 복사할 파일의 이름
    """

    # 소스 폴더에서 주어진 파일명을 검색
    for root, dirs, files in os.walk(source_folder):
        if file_name in files:
            source_file_path = os.path.join(root, file_name)
            destination_file_path = os.path.join(destination_folder, file_name)

            # 파일을 대상 폴더로 복사
            if do_copy:
                shutil.copy(source_file_path, destination_file_path)
            print(f"'{file_name}' has been copied to '{destination_folder}'.")
            break
    else:
        # 파일을 찾지 못한 경우
        print(f"'{file_name}' not found in '{source_folder}'.")


if __name__ == '__main__':   
    # source_image_folder = '../val100_yochin/images'    
    source_image_folder = '../val100_hbo/images'

    # source_anno_folder = '/home/yochin/Desktop/GLIP/OUTPUT_gd/aihub_pwalkDB_with_gd_labels_using_yolo_detector_train852/pred_pascal'
    # source_anno_folder = '/home/yochin/Desktop/GLIP/OUTPUT_gd/gdDB_with_aihub_labels_using_yolo_detector_train958/pred_pascal'
    # destination_anno_folder = '../val100_hbo/det_anno_pred'

    # source_anno_folder = '/media/NAS_GDHRI/dbs/aihub_pwalk/organized/Annotations'
    source_anno_folder = '/media/NAS_GDHRI/dbs/naeultech_231109/organized/Annotations'
    destination_anno_folder = '../val100_hbo/det_anno_gt'

    do_copy = True     # if False, just show the copy info.

    # 소스 폴더에서 파일 목록 가져오기
    list_img_files = [f for f in os.listdir(source_image_folder) if os.path.isfile(os.path.join(source_image_folder, f))]

    # 각 파일을 대상 폴더로 복사
    for file_name in list_img_files:
        # 파일 확장자 확인 (예: '.jpg', '.png')
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            filename_xml = os.path.splitext(file_name)[0] + '.xml'
            
            # 함수 호출
            copy_file(source_anno_folder, destination_anno_folder, filename_xml, do_copy=do_copy)
            