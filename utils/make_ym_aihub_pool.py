import os

# 1. A 폴더 내의 모든 txt 파일 읽기
def read_txt_files_from_folder(folder_path):
    file_names = set()  # 중복된 파일명을 제거하기 위해 set 사용
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

                print(f'start to process {file_path}')

                # 파일에서 각 라인을 읽어와 파일명을 set에 추가
                for line in lines:
                    if line.strip() in file_names:
                        print(f'{line.strip()} is in {file_name}')

                    file_names.add(line.strip())

                print('current length of file_names: ', len(file_names))
                    
    return file_names

# 2. B 폴더 내의 모든 파일명 읽기
def read_files_from_folder(folder_path):
    return set(os.listdir(folder_path))  # 중복된 파일명 처리를 위해 set 사용

# B 폴더 내의 파일명 제거
def remove_files_in_b_folder(a_file_names, b_folder_path):
    for file_name in a_file_names:
        file_path = os.path.join(b_folder_path, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")
        else:
            print(f"File not found, skipping: {file_path}")

# 1. A 폴더 내의 모든 .txt 파일을 읽어서 각 파일에 포함된 파일명을 리스트로 만듭니다.
# 2. 리스트 내부의 중복된 파일명을 제거합니다.
# 3. B 폴더 내의 모든 파일명을 읽어서, 2번에서 만든 리스트에 존재하는 파일명과 일치하는 경우 해당 파일명을 리스트에서 제거합니다.
if __name__ == '__main__':
    # Part 1. generate list file
    # a_folder_path = './aihub_file_pool_from_ym/raw'  # A 폴더 경로

    # a_file_names = read_txt_files_from_folder(a_folder_path)
    # a_file_names = sorted(a_file_names)

    # # print(a_file_names)
    # print(a_file_names[0])
    # print(a_file_names[-1])
    # with open('./aihub_file_pool_from_ym/aihub_pool.txt', 'w') as fid:
    #     fid.write('\n'.join(a_file_names))

    # Part 2. remove file
    a_file_names = set()
    with open('/media/NAS_GDHRI/dbs/PathGuidedVQA/2024.08.30_20k/aihub_pool.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        # a_file_names has "" in filename
        line = line.strip().replace('"', '')    # aaa.xml
        line = os.path.splitext(line)[0]
        a_file_names.add(line)


    # # temporary
    # a_file_names = [
    #     'ZED1_KSC_001032_L_P002065',
    #     'ZED1_KSC_001131_L_P002067',
    #     'ZED1_KSC_003135_L_P000002',
    #     'ZED1_KSC_003163_L_P000008',
    #     'ZED1_KSC_003288_L_P000022',
    #     'ZED1_KSC_003776_L_P000078',
    #     'ZED1_KSC_005109_L_P000261',
    #     'ZED1_KSC_003808_L_P000092',
    #     'ZED1_KSC_004149_L_P000130',
    #     'ZED1_KSC_004280_L_P000147',
    #     'ZED1_KSC_004999_L_P000232',
    # ]
    # a_file_names = set(a_file_names)


    # b_folder_base = '/media/NAS_GDHRI/dbs/PathGuidedVQA/2024.08.30_20k'
    b_folder_base = '/home/yochin/Desktop/PathGuidedVQA_Base/output/2024.08.30_20k'
    # b_folder_path = os.path.join(b_folder_base, 'qa')  # B 폴더 경로
    # b_folder_path = os.path.join(b_folder_base, 'original_images')  # B 폴더 경로
    b_folder_path = os.path.join(b_folder_base, 'qa_json')  # B 폴더 경로


    # b_folder_path = '/home/yochin/Desktop/PathGuidedVQA_Base/val20k/original_images'


    b_file_names = read_files_from_folder(b_folder_path)
    b_file_names = sorted(b_file_names)
    b_file_names = set(b_file_names)

    # A 폴더의 파일명 리스트에서 B 폴더에 존재하는 파일명 제거
    remaining_files = []
    delete_files = []
    for item in b_file_names:
        filename_only, filename_ext = os.path.splitext(item)

        if filename_only in a_file_names:
            delete_files.append(item)
        else:
            remaining_files.append(item)

    print("B 폴더 내 삭제 파일 리스트: ")
    for file_name in delete_files:
        print(file_name)
        os.remove(os.path.join(b_folder_path, file_name))

    print(len(delete_files))
