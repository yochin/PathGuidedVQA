import os
import json
import argparse

from tools import read_gt_pair_from_xml


def parse_args():

     parser = argparse.ArgumentParser(
        description='Generate json file for the llava training framework')

     parser.add_argument(
         '--gt-dir', metavar='DIRECTORY for ground truth', required=True,
         help='directory which contains images and object properties')
     
     parser.add_argument(
         '--output', metavar='FILEPATH', required=True,
         help='path for an output json file')

     return parser.parse_args()
     

def main():

    args = parse_args()

    files = os.listdir(args.gt_dir)
    xml_files = [f for f in files if f.endswith(('.xml'))]


    gt_list = []
    for xml_file in xml_files:

        xml_path = os.path.join(args.gt_dir, xml_file)
        img_filename, answer = read_gt_pair_from_xml(xml_path)

        element = {
            "id": img_filename.split('.')[0],
            "image": img_filename,
            "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nDescribe the obstacles to the destination in a natural and simple way for a visually impaired person."
                    },
                    {
                        "from": "navigation assistant",
                        "value": answer
                    },
                ]
            
            }

        gt_list.append(element)

    with open(args.output, 'w', encoding='utf-8') as json_file:
        json.dump(gt_list, json_file, indent="\t", ensure_ascii=False)


    return
   

if __name__ == '__main__':
    main()