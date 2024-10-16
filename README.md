# Automated Data Generation Pipeline for Space-Aware Instruction Tuning (SAIT)

This is the official repository for automated data generation pipeline used in our project on [Space-Aware VLM](https://github.com/byungokhan/Space-awareVLM) for guide dog robots assisting the visually impaired.

## Overview
This repository contains the code for our novel automated data generation pipeline that focuses on virtual paths to destinations in 3D space and their surroundings. The pipeline is designed to enhance the spatial understanding of Vision-Language Models (VLMs) by generating training data that emphasizes spatial relationships and navigational contexts.

## Key Features and Steps
* Object Detection and Depth Estimation: Utilizes advanced models for detecting objects and estimating depth in images captured from a pedestrian's viewpoint.
* Virtual Pathfinding: Automatically identifies a virtual path from the user's position to a specified goal position within the image.
* Region Masking and Prompting: Generates region-specific prompts and masks images to focus the VLM on relevant spatial areas, improving its understanding of left, right, and path descriptions.
* Automated Description Generation: Produces concise and essential descriptions for destinations, paths, and surrounding areas to assist visually impaired users effectively.
* Path Navigability Decision: Determines whether the identified path is walkable, providing recommendations based on the descriptions of the destination, path, and surrounding areas.
* Integration with VLMs: Designed to work seamlessly with Vision-Language Models, enhancing their spatial reasoning capabilities.

## Getting Started
### Prerequisites

### Installation
LLaVA 1.6과 Yolo V8이 필요합니다.

### Setup
Install the related codes and download the pre-trained models for object detection and depth estimation:

* [YOLOv8 Object Detectors](https://github.com/ultralytics/ultralytics): Trained on VIN and SideGuide datasets.
* [Depth Anything V2 for Metric Depth Estimation](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth): For monocular metric depth estimation.
Place the downloaded yolo-v8 models in the yolo/ directory and depth_anything_v2 directory.

### Configuration
Edit the configuration file configs/tr20k/config_p3_tr20k.yaml to set the paths to the models and adjust any parameters as needed.

### Running the Pipeline
```bash
./run_gen_tr.sh
```

### Output
The pipeline will generate:

* Masked Images for debugging: Images with regions masked for left, right, path, and destination areas.
* Generated Description Files: Text descriptions for each area, virtual path coordinates, and a decision on whether the path is walkable, along with a brief explanation.

## Usage for training VLMs
Follow the instructions in [Space-Aware VLM](https://github.com/byungokhan/Space-awareVLM).

## Acknowledgments
This work was supported by the Institute of Information & communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (RS-2023-00215760, Guide Dog: Development of Navigation AI Technology of a Guidance Robot for the Visually Impaired Person). 

This research (paper) used datasets from ‘The Open AI Dataset Project (AI-Hub, S. Korea)’. All data information can be accessed through ‘AI-Hub (www.aihub.or.kr)’.

## Contact
For questions or collaborations, please contact:

* Woo-han Yun: yochin@etri.re.kr
* ByungOk Han: byungok.han@etri.re.kr

## License
We will add the license information later.