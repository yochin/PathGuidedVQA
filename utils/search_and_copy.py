import os
import shutil

def copy_file(source_folder, destination_folder, file_name, do_copy=True, then_remove=False):
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
                # shutil.copy(source_file_path, destination_file_path)    # file + permission
                shutil.copyfile(source_file_path, destination_file_path)    # file only
            if then_remove:
                os.remove(source_file_path)
            print(f"'{file_name}' has been copied to '{destination_folder}'.")
            break
    else:
        # 파일을 찾지 못한 경우
        print(f"'{file_name}' not found in '{source_folder}'.")


# 1. List up the files in the reference folder
# 2. Find and copy the files those are in the ref. from the source folder to dest folder


if __name__ == '__main__':   
    ref_file_folder = '/media/NAS_GDHRI/dbs/PathGuidedVQA/2024.08.06/DestMasking_DrawDepthPoint_FewExample_DecGPT_SumLLama38binst_DBval20k_Try2/qa'
    ref_file_ext = ('.xml')

    source_file_folder = '/home/yochin/Desktop/PathGuidedVQA_Base/val20k/original_images'     
    destination_file_folder = '/media/NAS_GDHRI/dbs/PathGuidedVQA/2024.08.06/DestMasking_DrawDepthPoint_FewExample_DecGPT_SumLLama38binst_DBval20k_Try2/original_images'
    copied_file_ext = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.JPG', '.PNG', '.JPEG')

    do_copy = True     # if False, just show the copy info.
    then_remove = False

    # ref 폴더에서 파일 목록 가져오기
    list_img_files = [f for f in os.listdir(ref_file_folder) if os.path.isfile(os.path.join(ref_file_folder, f))]

    # 각 파일을 대상 폴더로 복사
    for file_name in list_img_files:
        # 파일 확장자 확인 (예: '.jpg', '.png')
        if file_name.lower().endswith(ref_file_ext):

            file_exist = False

            for item_ext in copied_file_ext:
                copied_filename_ext = os.path.splitext(file_name)[0] + item_ext

                if os.path.exists(os.path.join(source_file_folder, copied_filename_ext)):
                    file_exist = True
                    break

            
            if file_exist:
                # 함수 호출
                copy_file(source_file_folder, destination_file_folder, copied_filename_ext, do_copy=do_copy, then_remove=then_remove)
            else:
                raise AssertionError(f'{copied_filename_ext} is not exist in {source_file_folder}')
            
