import os

def read_files_from_folder(folder_path):
    filenames_ext = set(os.listdir(folder_path))
    filenames = set()

    for item in filenames_ext:
        filenames.add(os.path.splitext(item)[0])

    # filenames = sorted(filenames)

    return filenames  # 중복된 파일명 처리를 위해 set 사용

if __name__ == '__main__':
    a_folder_path = '/home/yochin/Desktop/PathGuidedVQA_Base/val20k/original_images'
    a_file_names = read_files_from_folder(a_folder_path)

    # b_folder_path = '/home/yochin/Desktop/PathGuidedVQA_Base/output/2024.08.20/Masking_AddYOLO_FewExample_DescLLaVA_DecsGPT_SumLLama38binst_val20k/original_images'
    b_folder_path = '/home/yochin/Desktop/PathGuidedVQA_Base/output/2024.08.05_06_all/qa'
    b_file_names = read_files_from_folder(b_folder_path)

    a_b_file_names = a_file_names - b_file_names
    b_a_file_names = b_file_names - a_file_names

    print('a: ', len(a_file_names))
    print('b: ', len(b_file_names))
    print('a-b: ', len(a_b_file_names))
    print('b-a: ', len(b_a_file_names))

    a_file_names = list(a_file_names)
    b_file_names = list(b_file_names)
    print(a_file_names[:10])
    print(b_file_names[:10])

