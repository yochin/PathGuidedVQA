from prompt_library_by_hbo import get_direction

def get_prompt(goal_label_cxcy, bboxes, trial_num, sep_system=False, add_LCR_prompt=False):
    # 각 바운딩 박스에 대한 설명 구성
    # bbox_descriptions = [f"{label} at ({round(x_min, 2)}, {round(y_min, 2)}, {round(x_max, 2)}, {round(y_max, 2)})" for label, (x_min, y_min, x_max, y_max), _ in bboxes]
    bbox_descriptions = [f"{label} [{round(x_min, 4)}, {round(y_min, 4)}, {round(x_max, 4)}, {round(y_max, 4)}]" for label, (x_min, y_min, x_max, y_max), _ in bboxes]
    bbox_list_str = ", ".join(bbox_descriptions)
    goal_label, goal_cxcy = goal_label_cxcy

    if len(goal_cxcy) == 2:     # point
        # dest_descriptions = f"{goal_label} at ({round(goal_cxcy[0], 2)}, {round(goal_cxcy[1], 2)})"
        dest_descriptions = f"{goal_label} [{round(goal_cxcy[0], 4)}, {round(goal_cxcy[1], 4)}]"
    elif len(goal_cxcy) == 4:   # bbox
        dest_descriptions = f"{goal_label} [{round(goal_cxcy[0], 4)}, {round(goal_cxcy[1], 4)}, {round(goal_cxcy[2], 4)}, {round(goal_cxcy[3], 4)}]"
    else:
        raise AssertionError('check ', goal_cxcy)
    
    if add_LCR_prompt:
        if goal_cxcy[0] < 0.33:
            dest_descriptions = dest_descriptions + ' on the left side of the image'
        elif goal_cxcy[0] > 0.66:
            dest_descriptions = dest_descriptions + ' on the right side of the image'
        else:
            dest_descriptions = dest_descriptions + ' on the center side of the image'
        

    # list_ids_sep = [45101, 45102, 45103, 45104, 45105, 45106, 45107] 
    list_ids_sep = ['D',   'L',   'R',   'P',   'Not', 'Desc', 'Decs']

    list_prompt = []
    list_system = []
    # one turn
    if trial_num in list_ids_sep:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '        # default
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '  # LLaVA
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '  # LLaVA
                'Image size: 1.0x1.0.'  # LLaVA
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the user is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                # "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. "
                "Rules: \n"
                "1. Do not talk about detailed image coordinates. Never use pixel positional information. "
                "2. Consider perspective view of the 2D image property. "
                # "3. Do not talk about orange point to represent the destination and yellow point to represent the starting point . "
                )
            
            if trial_num == list_ids_sep[6]:
                list_system.append(
                    '3. If the user is in front of a crosswalk, look for the pedestrian traffic light and tell me the color of the light. If there is a red pedestrian traffic light, say \'Stop and wait\' option. '
                    '4. If there is an obstacle in the path obstructing the user\'s walk, say \'Stop and wait\' option. '
                    '5. If the path leads the user into dangerous areas like roads or construction sites, say \'Stop and wait\' option. '
                    '6. If the path is clear of obstacles, say \'Follow the path\' option. '                    
                )
                list_system = [' '.join(list_system)]

            if len(bboxes) > 0:
                list_system.append((f"The image contains objects, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')
        
        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt1 = []
        if trial_num == list_ids_sep[0]:
            # Destination
            # list_prompt1.append(f'Explain the destination, represented by a orange {dest_descriptions} in 1 line. '   # orange point
            # list_prompt1.append(f'The destination is the center of the visible area. Describe the destination in one sentence. '    # only circle
            #### masking_circle
            # list_prompt1.append(f'Assume the visible area in the image is the user\'s destination. '    # only circle, same style of left and right
            #                     # 'Analyze the input image and provide a one-sentence description of what is at the end of the path. '
            #                     'Analyze the visible area and provide a one-sentence description of the destination. '
            #                     # 'If the location is not on the floor, say the destination is obscured by the object. '  # PathR
            #                     # 'Example 1) The destination is an entrance of a building. '

            # ### masking_depth
            # list_prompt1.append(f'Assume the visible area in the image is the user\'s destination. '    # only circle, same style of left and right
            #                     # 'Analyze the input image and provide a one-sentence description of what is at the end of the path. '
            #                     'Analyze the visible area and provide a one-sentence description of the destination. '
            #                     # 'If the location is not on the floor, say the destination is obscured by the object. '  # PathR
            #                     # 'Example 1) The destination is an entrance of a building. '

            ### masking_depth + Draw_point
            list_prompt1.append(f'Assume the orange {dest_descriptions} in the visible area is the user\'s destination. '    # only circle, same style of left and right
                                # 'Analyze the input image and provide a one-sentence description of what is at the end of the path. '
                                'Analyze the area and provide a one-sentence description of the destination. '
                                'Do not talk about orange point to represent the destination. '
                                # 'If the location is not on the floor, say the destination is obscured by the object. '  # PathR
                                # 'Example 1) The destination is an entrance of a building. '

            # #### draw_circle and draw_point and say_point
            # list_prompt1.append(f'Assume the orange {dest_descriptions} in the image is the user\'s destination. '    # only circle, same style of left and right
            #                     'Analyze the area and provide a one-sentence description of the destination. '
            #                     'Do not talk about orange point to represent the destination. '
            #                     # 'Do not talk about orange circle to represent the destination. '

            # list_prompt1.append(f'Explain the destination, represented by an orange {dest_descriptions}, in one sentence. '    # only circle, same style of left and right
                                # 'Example 1) The destination is on the pedestrian road. '
                                # 'Example 2) The destination is on the roadway. '
                                # 'Example 3) The destination is an entrance of a building. '
                                # 'Example 4) The destination is not on the ground and is obscured by a car. '

                                # Few Example
                                'Example 1) The destination is ahead on the sidewalk. '
                                'Example 2) The destination is ahead on the car road. '
                                'Example 3) The destination is ahead but is obscured by a truck. '
                                'Example 4) The destination is ahead at the entrance of the building. '

                                # # 'Example 4) The destination is on the crosswalk. '
                                )       # DESC1
        if trial_num == list_ids_sep[1]:
            # Left
            # list_prompt1.append(f'Assume there is a path that starts at the yellow dot at the bottom center of the image and ends at the orange {dest_descriptions} in the the image. '
            list_prompt1.append(f'Assume the visible area in the image is to the left of the user\'s path. '
                                # 'Analyze the input image and provide a one-sentence description that includes what is located closely to the left of the path. '
                                'Analyze the visible area and provide a one-sentence description that includes what is located closely to the left of the path. '
                                'If there are no special objects other than the floor, say \'nothing\'. '  # PathR
                                'Example 1) There are cars on the left side. '
                                'Example 2) There are nothing than the floor on the left side. '  # PathR
                                )  # DESC1 or 2
        if trial_num == list_ids_sep[2]:
            # Right
            # list_prompt1.append(f'Assume there is a path that starts at the yellow dot at the bottom center of the image and ends at the orange {dest_descriptions} in the the image. '
            list_prompt1.append(f'Assume the visible area in the image is to the right of the user\'s path. '
                                # 'Analyze the input image and provide a one-sentence description that includes what is located closely to the right of the path. '
                                'Analyze the visible area and provide a one-sentence description that includes what is located closely to the right of the path. '
                                'If there are no special objects other than the floor, say \'nothing\'. '  # PathR
                                'Example 1) There are people on the right side. '
                                'Example 2) There are nothing than the floor on the right side. '  # PathR
                                )  # DESC1 or 2
        if trial_num == list_ids_sep[3]:
            # Path
            # list_prompt1.append(f'Assume there is a path that starts at the yellow dot at the bottom center of the image and ends at the orange {dest_descriptions} in the the image. '
            #                       'Analyze the input image and provide a one-sentence description that includes what is located on the reference path. '
            list_prompt1.append(f'Assume the visible area in the image is the user\'s path and the user moves forward along the visible area of the image. '
                                # 'Analyze the input image and provide a one-sentence description of what objects are on the path. '
                                'Analyze the visible area and provide a one-sentence description of what objects are on the path. '
                                'Example 1) There are cars and people on the path. '
                                'Example 2) There are nothing on the path. '
                                )  # DESC1 or 2
        if trial_num == list_ids_sep[4]:
            raise AssertionError('Check the prompt id!')
            list_prompt2 = []
            list_prompt2.append('1) Based on the current description, evaluate the user is able to move along the path without collision or collide with other objects. Then, select the most appropriate action: \'Follow the path\' or \'Stop and wait\'. '
                                'If the user is in front of a crosswalk and there is a red pedestrian traffic light, say \'Stop and wait\' option. '
                                'If there is an object other than the floor on the path, say \'Stop and wait\' option. '
                                'Say only the answer. '
                            )  # DESC1,2
            list_prompt2.append('2) Then, explain the reason in 1 line. ')    # DESC1,2
            list_prompt2.append('Example 1) 1) \'Stop and wait\' 2) The parked car is on the path. Wait or find an alternative route to the destination. ')
            list_prompt2.append('Example 2) 1) \'Follow the path\' 2) There is no obstacles on the path. ')
            list_prompt1 = [' '.join(list_prompt2)]
        if trial_num == list_ids_sep[5]:
            # VLM, describe and find
            list_prompt1.append(f'Describe where the user is located based on the image. '
                                'If the user is in front of a crosswalk, look for the pedestrian traffic light and tell me the color of the light. '
                                'Example 1) The user is in front of a crosswalk. The pedestrian traffic light is red. '
                                'Example 2) The user is in front of a construction site. '
                                )
        if trial_num == list_ids_sep[6]:
            # VLM, find and decision
            list_prompt2 = []
            list_prompt2.append('Based on the current description, evaluate the user is able to move along the path without collision or collide with other objects. '
                                'Then, select the most appropriate action: \'Follow the path\' or \'Stop and wait\'. '
                                'Say only the answer, explaining the selection and the reason in two sentences. '
                                # 'Rules: \n'
                                # '1) If the user is in front of a crosswalk, look for the pedestrian traffic light and tell me the color of the light. If there is a red pedestrian traffic light, say \'Stop and wait\' option. '
                                # '2) If there is an obstacle in the path obstructing the user\'s walk, say \'Stop and wait\' option. '
                                # '3) If the path leads the user into dangerous areas like roads or construction sites, say \'Stop and wait\' option. '
                                # '4) If the path is clear of obstacles, say \'Follow the path\' option. '
                                # '5) Say only the answer with explaining the reason in 1 line. '
                            )  # DESC1,2
            # list_prompt2.append('Example 1) 1) \'Stop and wait\' 2) The parked car is on the path. Wait or find an alternative route to the destination. ')
            # list_prompt2.append('Example 2) 1) \'Follow the path\' 2) There is no obstacles on the path. ')
            list_prompt2.append('Example 1) Stop and wait. A car is on the path, so walking to the destination is impossible. ')
            list_prompt2.append('Example 2) Follow the path. The path is clear of obstacles, so walking to the destination is possible. ')
            list_prompt2.append('Example 3) Stop and wait. It is in front of the crosswalk and the pedestrian traffic light is red, so walking to the destination is impossible. ')
            list_prompt1 = [' '.join(list_prompt2)]
        list_prompt.extend(list_prompt1)
    else:
        raise AssertionError(f'{trial_num} is not supported')

    return list_prompt, list_system