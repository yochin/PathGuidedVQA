import os
import sys
import xml.etree.ElementTree as ET
import json
import ast
import pdb

import argparse
import yaml
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='Copy image files')   

    parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()

    return args

def set_path_logger(path_to_log):
    if not os.path.exists(os.path.split(path_to_log)[0]):
        os.makedirs(os.path.split(path_to_log)[0])

    logFormatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s', datefmt='%Y/%m/%d %p %I:%M:%S, ')
    logFileHandler = logging.FileHandler(path_to_log)
    logConsoleHandler = logging.StreamHandler(sys.stdout)

    logFileHandler.setFormatter(logFormatter)
    logConsoleHandler.setFormatter(logFormatter)

    logging.getLogger().addHandler(logFileHandler)
    logging.getLogger().addHandler(logConsoleHandler)
    logging.getLogger().setLevel(logging.DEBUG)


def sample_list(input_list, n):
    if n <= 2:
        raise ValueError("N must be greater than 2 to ensure first and last elements are included.")
    if n >= len(input_list):
        return input_list[:]  # Return a copy of the entire list
    
    sampled_list = []
    sampled_list.append(input_list[0])  # Always include the first element

    if n > 2:
        step = (len(input_list) - 1) / (n - 1)  # -1 to account for the first element
        sampled_list.extend([input_list[int(round(step * i))] for i in range(1, n-1)])

    sampled_list.append(input_list[-1])  # Always include the last element
    
    return sampled_list

def xml_to_json(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # # Convert XML to JSON
    # xml_json = xml_to_json_recursive(root)

    # Skip the <Annotation> tag and directly process its children
    if root.tag == 'Annotation':
        json_data = {}
        for child in root:
            # child_json = xml_to_json_recursive(child)            
            # json_data[child.tag] = child_json

            if child.tag == 'path_array':
                path_array = ast.literal_eval(child.text)
                path_array_10 = sample_list(path_array, 10)
                assert len(path_array_10) == 10
                json_data[child.tag] = path_array_10
            elif child.tag in ['size_whc', 'goal_position_xy', 'bboxes']:
                json_data[child.tag] = ast.literal_eval(child.text)
            else:
                json_data[child.tag] = child.text
    else:
        raise ValueError("Root element must be <Annotation>.")
    
    return json_data

# def xml_to_json_recursive(element):
#     # Initialize the dictionary for this element
#     json_data = {}
    
#     # # Copy element attributes
#     # if element.attrib:
#     #     json_data["@attributes"] = element.attrib
    
#     # Copy element text if present
#     if element.text:
#         json_data["text"] = element.text
    
#     # Copy children recursively
#     for child in element:
#         child_json = xml_to_json_recursive(child)
        
#         # Check if the tag is already in the dictionary
#         if child.tag in json_data:
#             if type(json_data[child.tag]) is list:
#                 json_data[child.tag].append(child_json)
#             else:
#                 json_data[child.tag] = [json_data[child.tag], child_json]
#         else:
#             json_data[child.tag] = child_json

#     return json_data

def convert_xml_files(source_dir, target_dir):
    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # List all XML files in the source directory
    xml_files = [f for f in os.listdir(source_dir) if f.endswith('.xml')]

    # Process each XML file
    for xml_file in xml_files:
        xml_path = os.path.join(source_dir, xml_file)
        try:
            json_data = xml_to_json(xml_path)

            # Generate output JSON file path
            json_file = os.path.splitext(xml_file)[0] + '.json'
            json_path = os.path.join(target_dir, json_file)

            # Save JSON data to file
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=4)

            print(f"Converted {xml_file} to {json_file}")
        except Exception as e:
            print(f"Error converting {xml_file}: {str(e)}")


if __name__ == '__main__':
    args = parse_args()

    with open(args.config) as fid:
        conf = yaml.load(fid, Loader=yaml.FullLoader)

    # set a log file path        
    path_to_log = os.path.join(conf['output_dir'], conf['task_name'], 'xml_to_json.log')
    set_path_logger(path_to_log)

    # Example usage
    source_directory = os.path.join(conf['output_dir'], conf['task_name'], 'qa')
    target_directory = os.path.join(conf['output_dir'], conf['task_name'], 'qa_json')

    convert_xml_files(source_directory, target_directory)
