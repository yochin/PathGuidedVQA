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
        

    list_ids = [1540301, 1540302, 1540303]    
    list_ids_sep = [45101, 45102, 45103, 45104, 45105, 45106, 45107]    
    #               D      L      R      P      Not    Desc   Decision

    list_prompt = []
    list_system = []
    if trial_num < 15:
        if len(bboxes) > 0:
            list_prompt.append((
                    "The image contains the following objects, which are located within bounding boxes represented by four numbers. "
                    "These four numbers correspond to the normalized pixel values for left, top, right, and bottom. "
                    f"The included objects are {bbox_list_str}.\n"
                    'Describe the overall photo from near to far.'
                    ))
        else:
            list_prompt.append('Describe the overall photo from near to far.')
        list_prompt.append(f'Explain the path to the {dest_descriptions}, which is the current destination.')
        list_prompt.append('Explain the obstacles that exist on the path, and tell us what to do to get the destination.')
    # one turn
    elif trial_num in list_ids_sep:
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
                                'Analyze the visible area and provide a one-sentence description of what objects are on the path and the destination of the path. '
                                'Example 1) The destination of the path is the entrance of the building, and there are cars and people on the path. '
                                'Example 2) The destination of the path is the sidewalk, and there are nothing on the path. '
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



        # str_dir=get_direction(goal_cxcy[0])
        # # Summarize prompt
        # list_prompt1 = []
        # if trial_num == list_ids_sep[0]:
        #     list_prompt1.append(f'Explain the destination, represented by a orange {dest_descriptions} in 1 line. '
        #                         'Example) The destination is an entrance of a building.'
        #                         )       # DESC1
        # if trial_num == list_ids_sep[1]:
        #     list_prompt1.append(f'There is a red path that starts at the yellow dot at the bottom center of the image and ends at the orange {dest_descriptions} in the the image. '
        #                         'Analyze the input image and provide a one-sentence description that includes what is located closely to the left of the red line. '
        #                         'If there are no special objects other than the floor, say \'nothing\'. '  # PathR
        #                         'Example) There are cars on the left side.'
        #                         'Example) There are nothing than the floor on the left side.'  # PathR
        #                         )  # DESC1 or 2
        # if trial_num == list_ids_sep[2]:
        #     list_prompt1.append(f'There is a red path that starts at the yellow dot at the bottom center of the image and ends at the orange {dest_descriptions} in the the image. '
        #                         'Analyze the input image and provide a one-sentence description that includes what is located closely to the right of the red line. '
        #                         'If there are no special objects other than the floor, say \'nothing\'. '  # PathR
        #                         'Example) There are people on the right side.'
        #                         'Example) There are nothing than the floor on the right side.'  # PathR
        #                         )  # DESC1 or 2
        # if trial_num == list_ids_sep[3]:
        #     list_prompt1.append(f'There is a red path that starts at the yellow dot at the bottom center of the image and ends at the orange {dest_descriptions} in the the image. '
        #                         'Analyze the input image and provide a one-sentence description that includes what is located on the reference red line. '
        #                         'Example) There are nothing on the path.'
        #                         )  # DESC1 or 2
        # if trial_num == list_ids_sep[4]:
        #     list_prompt2 = []
        #     list_prompt2.append('1) Based on the current description, evaluate the user is able to move along the red line without collision or collide with other objects. Then, select the most appropriate action: \'Follow the red line\' or \'Stop and wait\'. '
        #                         'If the user is in front of a crosswalk, look for the traffic light and tell me the color of the light. If there is a red traffic light, say \'Stop and wait\' option. '
        #                         'Say only the answer. '
        #                     )  # DESC1,2
        #     list_prompt2.append('2) Then, explain the reason in 1 line. ')    # DESC1,2
        #     list_prompt2.append('Example 1) 1) \'Stop and wait\' 2) The parked car is on the path, and there is no route to the destination. ')
        #     list_prompt2.append('Example 2) 1) \'Follow the red line\' 2) There is no obstacles on the path. ')
        #     list_prompt1 = [' '.join(list_prompt2)]

        # list_prompt.extend(list_prompt1)
    # one turn
    elif trial_num in list_ids:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '        # default
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '  
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the user is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains objects, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt1 = []
        list_prompt1.append(f'1) Explain the destination, represented by a orange {dest_descriptions} in 1 line. ')       # DESC1
        # list_prompt1.append(f'1) Where is the orange {dest_descriptions}? Do not mention the orange point or the red path. ')   # DESC2
        # dest_descriptions = dest_descriptions.replace('point', 'dot')   # DESC3
        # list_prompt1.append(f'1) Explain the destination, represented by an orange {dest_descriptions} in 1 line. Focus on the location where the orange dot is placed and the space around it. Reference the following example: Example) The destination is the building entrance. ')       # DESC3

        list_prompt1.append(f'2) There is a red path that starts at the yellow dot at the bottom center of the image and ends at the orange {dest_descriptions} in the the image. '
                            'Analyze the input image and provide a one-sentence description that includes what is located to the left, to the right, and on the reference line based on the red line. '
                            )  # DESC1 or 2
        # list_prompt1.append(f'2) There is a red path that starts at the yellow dot at the bottom center of the image and ends at the orange {dest_descriptions} in the the image. '
        #                     'Analyze the input image and provide a one-sentence description that includes what is located to the left, to the right, and on the reference path based on the red path. '
        #                     'Reference the following example: Example) There are cars on the left side of the red path, people on the right side, and nothing on the path itself. '
        #                     )    # DESC3

        list_prompt2 = []
        list_prompt2.append('3) Based on the current description, evaluate the user is able to move along the red line without collision or collide with other objects. Then, select the most appropriate action: \'Follow the red line\' or \'Stop and wait or find another route\'. '
                            'If the user is in front of a crosswalk, look for the traffic light and tell me the color of the light. If there is a red traffic light, say \'Stop and wait\' option. '
                           'Say only the answer. '
                           )  # DESC1,2
        list_prompt2.append('4) Then, explain the reason in 1 line. ')    # DESC1,2
        # list_prompt2.append('3) Based on the current description, evaluate the user is able to move along the red path without collision or collide with other objects. Then, select the most appropriate action: \'Follow the red line\' or \'Stop and wait or find another route\'. '
        #                     'If the user is in front of a crosswalk, look for the traffic light. If there is a red traffic light, say \'Stop and wait\' option. '
        #                    'Say only the answer. '
        #                    )    # DESC3
        # list_prompt2.append('4) Then, explain the reason in 1 line. If the user is in front of a crosswalk, look for the traffic light and also say the color of the light. ')  # DESC3


        if trial_num == list_ids[0]:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
        elif trial_num == list_ids[1]:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
            list_prompt = [' '.join(list_prompt)]
        elif trial_num == list_ids[2]:
            list_prompt1 = [' '.join(list_prompt1)]
            list_prompt2 = [' '.join(list_prompt2)]
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)

    elif trial_num == 9999:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                )
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        list_prompt.append("Analyze the input image and provide a one-sentence description that includes what is on the left, what is on the right, what is in front, and whether it is possible to move forward.")
    elif trial_num == 9998:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the user is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains objects, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')
        
        list_prompt.append("Analyze the input image and provide a one-sentence description that includes what is on the left, what is on the right, what is in front, and whether it is possible to move forward.")

    elif trial_num == 15:
        # Yuns prompt
        list_prompt.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains the following objects, {bbox_list_str}.\n"))
        list_prompt.append(f'After describing the overall photo from near to far, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path.')
        list_prompt = [' '.join(list_prompt)]

        # Summarize prompt
        # list_prompt.append('Summarize the path in 3 lines.')
        list_prompt.append(('Summarize the answer in 3 lines. '
                            'The first line summarizes the path to the destination using direction and distance. '
                            'The second line describes objects along the path. '
                            'In the third line, decide whether to go or stop depending on the situation and explain the reason.')
        )
    elif trial_num == 16:
        # Dr.Hans prompt
        drhans_prompt_1 = ("[Context: The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. " 
            "In this image, the user's starting point is situated below the center of the image at the point [0.5, 1.0]. "
            "Consider the starting point as the ground where the user is standing.]\n" )
        drhans_prompt_2 = (f"[Obstacle Name at (bounding box): [{bbox_list_str}].]\n")
        drhans_prompt_3 = (f"[Destination Name at (point): [{dest_descriptions}].]\n")
        drhans_prompt_4 = ("Describe the obstacles to the destination in a natural and simple way "
            "for a visually impaired person as a navigation assistant in 3 sentences. "
            "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. "
            "First, answer it in English, then translate it Korean.")
        
        list_prompt.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
        list_prompt.append(drhans_prompt_1)
        if len(bboxes) > 0:
            list_prompt.append(drhans_prompt_2)
        list_prompt.append(drhans_prompt_3)
        list_prompt.append(drhans_prompt_4)
        list_prompt = [' '.join(list_prompt)]

        list_prompt.append(('Summarize the answer in 3 lines. '
                            'The first line summarizes the path to the destination using direction and distance. '
                            'The second line describes objects along the path. '
                            'In the third line, decide whether to go or stop depending on the situation and explain the reason.')
        )
    elif trial_num == 17:
        list_prompt.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
        list_prompt.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                           "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                           "At the bottom of the image, the left point ranges from [0.0, 1.0] to [0.5, 1.0] and right point ranges from [0.5, 1.0] to [1.0, 1.0]"
                           "At the top of the image, the left point ranges from [0.0, 0.0] to [0.5, 0.0] and right point ranges from [0.5, 0.0] to [1.0, 0.0]"
                           "Consider the starting point as the ground where the user is standing.]\n"
                           )
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains obstacles, {bbox_list_str}.\n"))
        list_prompt.append(f'After describing the overall photo from near to far, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path.')
        list_prompt.append("Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. ")
        list_prompt = [' '.join(list_prompt)]

        # Summarize prompt
        list_prompt.append('Summarize the path to the destination using direction and distance in 1 line. ')
        list_prompt.append('Enumerate the obstacles along the path with its relative position to the bottom-center point of the image. ')
        list_prompt.append('Select GO or STOP. Then explain the reason in 1 line. ')
    elif trial_num == 18:
        if sep_system:
            list_system.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_system.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing.\n"
                                )
            list_system = [' '.join(list_system)]
        else:
            list_prompt.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_prompt.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing.\n"
                                )
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains obstacles, {bbox_list_str}.\n"))
        list_prompt.append(f'After describing the overall photo from near to far, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path.')
        list_prompt.append("Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. ")
        list_prompt = [' '.join(list_prompt)]

        # Summarize prompt
        list_prompt.append('Summarize the path to the destination using direction and distance in 1 line. ')
        list_prompt.append('Enumerate the obstacles only on the path with its relative position to the bottom-center point of the image. ')
        list_prompt.append('Select GO or STOP. Then explain the reason in 1 line. ')
    
    elif trial_num == 118:
        if sep_system:
            list_system.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_system.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing.\n"
                                )
            list_system = [' '.join(list_system)]
        else:
            list_prompt.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_prompt.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing.\n"
                                )
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains obstacles, {bbox_list_str}.\n"))
        list_prompt.append(f'After describing the overall photo from near to far, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path.')
        list_prompt.append("Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. ")
        list_prompt = [' '.join(list_prompt)]

        # Summarize prompt
        list_prompt.append('Summarize the path to the destination using direction and distance in 1 line. ')
        list_prompt.append('Enumerate the obstacles only on the path with its relative position to the bottom-center point of the image. ')
        list_prompt.append('Choose: go left 45 degree, go straight, go right 45 degree, or stop. Then explain the reason in 1 line. ')
    
    elif trial_num in [1118, 11118, 1158, 11158]:
        if sep_system:
            list_system.append('A chat between a human and an AI that understands visuals. '
                               'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                               'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                               'Image size: 1.0x1.0.'
                               "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                               "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image."
                               "Consider the starting point as the ground where the user is standing."
                               "Explain as if you were a navigation assistant explaining to a visually impaired person."
                               "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                               )
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains obstacles, {bbox_list_str}.\n"))
        list_prompt.append(f'Describe the overall photo from near to far.')
        list_prompt = [' '.join(list_prompt)]

        # Summarize prompt
        list_prompt.append(f'Explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. ')
        list_prompt.append('What obstacles are on the path described? Enumerate one by one. ')
        if trial_num == 1118:
            list_prompt.append('What action do you recommend? Please choose from the following options. A) Go straight, B) Go left 45, C) Go right 45, D) Stop. Then, explain the reason in 1 line. ')
        elif trial_num == 11118:
            list_prompt.append('What action do you recommend? Please choose from the following options. A) Go right 45, B) Stop, C) Go straight, D) Go left 45. Then, explain the reason in 1 line. ')
        elif trial_num == 1158:
            list_prompt.append('What action do you recommend? Please choose from the following options. Go straight, Go left 45, Go right 45, Stop. Then, explain the reason in 1 line. ')
        elif trial_num == 11158:
            list_prompt.append('What action do you recommend? Please choose from the following options. Go right 45, Stop, Go straight, Go left 45. Then, explain the reason in 1 line. ')

    
    
    
    elif trial_num == 21:
        if sep_system:
            list_system.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_system.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. \n"
                                )
            list_system = [' '.join(list_system)]
        else:
            list_prompt.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_prompt.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. \n"
                                )
        list_prompt.append('From now on, we will talk about the new image. ')
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains obstacles, {bbox_list_str}.\n"))
        list_prompt.append(f'After describing the overall photo from near to far, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path.')
        list_prompt = [' '.join(list_prompt)]

        # Summarize prompt
        list_prompt.append('Summarize the path to the destination using direction and distance in 1 line. ')
        list_prompt.append('Enumerate the obstacles only on the path with its relative position to the bottom-center point of the image. ')
        list_prompt.append('Select GO or STOP. Then explain the reason in 1 line. ')

    elif trial_num == 30:
        if sep_system:
            list_system.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_system.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. \n"
                                )
            list_system = [' '.join(list_system)]
        else:
            list_prompt.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_prompt.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. \n"
                                )
        list_prompt.append('From now on, we will talk about the new image. ')
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains obstacles, {bbox_list_str}.\n"))
        list_prompt.append(f'After describing the overall photo from near to far, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path.')
        list_prompt = [' '.join(list_prompt)]

        # Summarize prompt
        list_prompt.append('Summarize the path to the destination using direction and distance in 1 line. ')
        list_prompt.append('Select GO or STOP. Then explain the reason in 1 line. ')

    elif trial_num == 130:
        if sep_system:
            list_system.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_system.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. \n"
                                )
            list_system = [' '.join(list_system)]
        else:
            list_prompt.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_prompt.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. \n"
                                )
        list_prompt.append('From now on, we will talk about the new image. ')
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains obstacles, {bbox_list_str}.\n"))
        list_prompt.append(f'After describing the overall photo from near to far, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path.')
        list_prompt = [' '.join(list_prompt)]

        # Summarize prompt
        list_prompt.append('Summarize the path to the destination using direction and distance in 1 line. ')
        list_prompt.append('Choose: go left 45 degree, go straight, go right 45 degree, or stop. Then explain the reason in 1 line. ')

    elif trial_num == 22:
        if sep_system:
            list_system.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_system.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. \n"
                                )
            list_system = [' '.join(list_system)]
        else:
            list_prompt.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_prompt.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. \n"
                                )
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains obstacles, {bbox_list_str}.\n"))
        list_prompt.append(f'After describing the overall photo from near to far, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path.')
        list_prompt = [' '.join(list_prompt)]

        # Summarize prompt
        list_prompt.append('Summarize the path to the destination using direction and distance in 1 line. ')
        list_prompt.append('Enumerate the obstacles only on the path with its relative position to the bottom-center point of the image. ')
        list_prompt.append('Select GO or STOP. Then explain the reason in 1 line. ')
        
    elif trial_num == 19:   # one-turn query
        if sep_system:
            list_system.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_system.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                                )
            list_system = [' '.join(list_system)]
        else:
            list_prompt.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_prompt.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                                )
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains obstacles, {bbox_list_str}.\n"))
        list_prompt.append(f'First, describe the overall photo from near to far.')
        list_prompt.append(f'Second, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line.')
        list_prompt.append(f'Third, choose GO or STOP with the reason in 1 line.')
        list_prompt = [' '.join(list_prompt)]

    elif trial_num == 119:   # one-turn query
        if sep_system:
            list_system.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_system.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                                )
            list_system = [' '.join(list_system)]
        else:
            list_prompt.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_prompt.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                                )
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains obstacles, {bbox_list_str}.\n"))
        list_prompt.append(f'First, describe the overall photo from near to far.')
        list_prompt.append(f'Second, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line.')
        list_prompt.append(f'Third, choose: go left 45 degree, go straight, go right 45 degree, or stop, with the reason in 1 line.')
        list_prompt = [' '.join(list_prompt)]

    elif trial_num in [1119, 11119, 1159, 11159]:   # one-turn query
        if sep_system:
            list_system.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_system.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image."
                                "Consider the starting point as the ground where the user is standing."
                                "Explain as if you were a navigation assistant explaining to a visually impaired person."
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                                )
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')
        
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains obstacles, {bbox_list_str}.\n"))
        list_prompt.append(f'First, describe the overall photo from near to far.')
        list_prompt.append(f'Second, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line.')
        list_prompt.append(f'Third, what obstacles are on the path described? Enumerate one by one.')
        if trial_num == 1119:
            list_prompt.append(f'Fourth, what action do you recommend? Please choose from the following options. A) Go straight, B) Go left 45, C) Go right 45, D) Stop. Then, explain the reason in 1 line.')
        elif trial_num == 11119:
            list_prompt.append(f'Fourth, what action do you recommend? Please choose from the following options. A) Go right 45, B) Stop, C) Go straight, D) Go left 45. Then, explain the reason in 1 line.')
        elif trial_num == 1159:
            list_prompt.append(f'Fourth, what action do you recommend? Please choose from the following options. Go straight, Go left 45, Go right 45, Stop. Then, explain the reason in 1 line.')
        elif trial_num == 11159:
            list_prompt.append(f'Fourth, what action do you recommend? Please choose from the following options. Go right 45, Stop, Go straight, Go left 45. Then, explain the reason in 1 line.')

        list_prompt = [' '.join(list_prompt)]

    elif trial_num in [1819, 1859]:   # two-turn query
        if sep_system:
            list_system.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_system.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image."
                                "Consider the starting point as the ground where the user is standing."
                                "Explain as if you were a navigation assistant explaining to a visually impaired person."
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                                )
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')
        
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains objects, {bbox_list_str}.\n"))
        list_prompt.append(f'First, describe the overall photo from near to far.')
        list_prompt.append(f'Second, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line.')
        list_prompt = [' '.join(list_prompt)]

        if trial_num == 1819:
            list_prompt.append(f'Based on the description, choose the appropriate action to reach the destination from the following options. A) Go straight, B) Go left 45, C) Go right 45, D) Stop.')
        elif trial_num == 1859:
            list_prompt.append(f'Based on the description, choose the appropriate action to reach the destination from the following options. Go straight, Go left 45, Go right 45, Stop.')

        list_prompt.append('Based on the description, what obstacles are on the path? List one by one. Say only the answer. Use a comma as a separator. If there is no obstacles, say "no obstacles".')



    elif trial_num in [41819, 41859]:   # two-turn query
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the user is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir = get_direction(goal_cxcy[0])
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains objects, {bbox_list_str}. \n"))
        list_prompt.append(f'First, describe the overall photo from near to far. ')
        list_prompt.append(
            'The horizontal direction of the destination is ' + str_dir + ' from the user. '
            f'Second, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. '
        )
        list_prompt = [' '.join(list_prompt)]

        if trial_num == 41819:
            list_prompt.append('Based on the description, choose the appropriate action to reach the destination at the first step and at the second step from the following options: '
                               '\'A) Go straight to the center\', \'B) Go diagonally to the left\', \'C) Go diagonally to the right\', \'D) Stop and wait\'. '
				'Say only the answer. If there is a potential danger for the user, say \'D) Stop and wait\' option. '
                               )
        elif trial_num == 41859:
            list_prompt.append('Based on the description, choose the appropriate action to reach the destination at the first step and at the second step from the following options: '
                               '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', \'Stop and wait\'. '
				'Say only the answer. If there is a potential danger for the user, say \'Stop and wait\' option. '
				)

        list_prompt.append(
            'Based on the description, what obstacles are on the path? List one by one. Say only the answer. Use a comma as a separator. If there is no obstacles, say "no obstacles".'
        )
 
    # multi turn, one turn, two turn
    elif trial_num in [41118, 41128, 41138]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the user is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt.append('1) Describe the overall photo from near to far. ')
        # list_prompt.append(f'2) Explain the destination, {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt.append(f'2) Explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. '
                           'The horizontal direction of the destination is '+ str_dir +' from the user.')
        
        if trial_num == 41138:
            list_prompt = [' '.join(list_prompt)]

        list_prompt.append('3) What obstacles are on the path described? Enumerate one by one.')
        list_prompt.append('4) Which action do you recommend to the user who wants to go to the destination? '
                           'Please choose an action in the following four options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Stop and wait\'. '
                           'Say only the answer. If there is a potential danger for the user, say \'Stop and wait\' option. '
                           )
        list_prompt.append('5) Then, explain the reason in 1 line. ')

        if trial_num == 41128:
            list_prompt = [' '.join(list_prompt)]


    # multi turn, one turn, two turn
    elif trial_num in [41148, 41158, 41168]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the user is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt.append('1) Describe the overall photo from near to far. ')
        # list_prompt.append(f'2) Explain the destination, {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt.append(f'2) Explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. '
                           'The horizontal direction of the destination is '+ str_dir +' from the user.')
        
        if trial_num == 41168:
            list_prompt = [' '.join(list_prompt)]

        list_prompt.append('3) Which action do you recommend to the user who wants to go to the destination? '
                           'Please choose an action in the following four options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Stop and wait\'. '
                           'Say only the answer. If there is a potential danger for the user, say \'Stop and wait\' option. '
                           )
        list_prompt.append('4) Then, explain the reason in 1 line. ')
        list_prompt.append('5) What obstacles are on the path described? Enumerate one by one.')

        if trial_num == 41158:
            list_prompt = [' '.join(list_prompt)]


    # one turn
    elif trial_num in [41178, 41188, 41198]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the user is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt.append('1) Describe the overall photo from near to far. Focus on the objects, their relative positions in the image, and its positions with their neighbor objects. ')
        list_prompt.append(f'2) Explain the destination, {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt.append(f'3) Explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. '
                           'The horizontal direction of the destination is '+ str_dir +' from the user.')
        
        if trial_num == 41198:
            list_prompt = [' '.join(list_prompt)]

        list_prompt.append('4) Which action do you recommend to the user who wants to go to the destination? '
                           'Please choose an action in the following four options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Stop and wait\'. '
                           'Say only the answer. If there is a potential danger for the user, say \'Stop and wait\' option. '
                           )
        list_prompt.append('5) Then, explain the reason in 1 line. ')
        list_prompt.append('6) What obstacles are on the path described? Enumerate one by one.')

        if trial_num == 41188:
            list_prompt = [' '.join(list_prompt)]


    # one turn
    elif trial_num in [541178, 541188, 541198]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the user is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt.append('1) Describe the overall photo from near to far. Focus on the objects, their relative positions in the image, and its positions with their neighbor objects. ')
        list_prompt.append(f'2) Explain the destination, represented by a blue {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt.append(f'3) Describe in one sentence the red path that starts at the green dot at the bottom center of the image and ends at the blue dot {dest_descriptions} in the the image . '
                           'Focus on the direction of the path and any obstacles present along it, rather than on the green starting point, the blue endpoint, or the red line itself. '
                           'The horizontal direction of the destination is '+ str_dir +' from the user. ')

        if trial_num == 541198:
            list_prompt = [' '.join(list_prompt)]

        list_prompt.append('4) Which action do you recommend to the user who wants to go to the destination? '
                           'Please choose an action in the following four options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Stop and wait\'. '
                           'Say only the answer. If there is a potential danger for the user, say \'Stop and wait\' option. '
                           )
        list_prompt.append('5) Then, explain the reason in 1 line. ')
        list_prompt.append('6) What obstacles are on the path described? Enumerate one by one. ')

        if trial_num == 541188:
            list_prompt = [' '.join(list_prompt)]


    # one turn
    elif trial_num in [540178, 540188, 540198]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the user is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt.append('1) Describe the overall photo from near to far. Focus on the objects, their relative positions in the image, and its positions with their neighbor objects. ')
        list_prompt.append(f'2) Explain the destination, represented by a blue {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt.append(f'3) Describe in one sentence the red path that starts at the green dot at the bottom center of the image and ends at the blue dot {dest_descriptions} in the the image . '
                           'Focus on the direction of the path and any obstacles present along it, rather than on the green starting point, the blue endpoint, or the red line itself. ')

        if trial_num == 540198:
            list_prompt = [' '.join(list_prompt)]

        list_prompt.append('4) Which action do you recommend to the user who wants to go to the destination? '
                           'Please choose an action in the following four options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Stop and wait\'. '
                           'Say only the answer. If there is a potential danger for the user, say \'Stop and wait\' option. '
                           )
        list_prompt.append('5) Then, explain the reason in 1 line. ')
        list_prompt.append('6) What obstacles are on the path described? Enumerate one by one. ')

        if trial_num == 540188:
            list_prompt = [' '.join(list_prompt)]


    # one turn
    elif trial_num in [540201, 540202, 540203]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the user is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt1 = []
        list_prompt1.append('1) Describe the overall photo from near to far. '
                            'Focus on the objects, their relative positions in the image, and its positions with their neighbor objects. '
                            'If the user is in front of a crosswalk, look for the traffic light and tell me the color of the light. ')
        list_prompt1.append(f'2) Explain the destination, represented by a orange {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt1.append(f'3) Describe the red path that starts at the yellow dot at the bottom center of the image and ends at the orange dot {dest_descriptions} in the the image . '
                           'Focus on the direction of the path and any obstacles present along it, rather than on the yellow starting point, the orange endpoint, or the red line itself. '
                           'The horizontal direction of the destination is '+ str_dir +' from the user. ')

        list_prompt2 = []
        list_prompt2.append('4) Select the most appropriate action to follow the path described above from the following four options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Stop and wait\'. '
                           'Say only the answer. If there is a red traffic light, say \'Stop and wait\' option. '
                           )
        list_prompt2.append('5) Then, explain the reason in 1 line. ')
        list_prompt2.append('6) What obstacles are on the path described? Enumerate one by one. ')

        if trial_num == 540201:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
        elif trial_num == 540202:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
            list_prompt = [' '.join(list_prompt)]
        elif trial_num == 540203:
            list_prompt1 = [' '.join(list_prompt1)]
            list_prompt2 = [' '.join(list_prompt2)]
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)



    # one turn
    elif trial_num in [540211, 540212, 540213]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the user is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt1 = []
        list_prompt1.append('1) Describe the overall photo from near to far. '
                            'Focus on the objects, their relative positions in the image, and its positions with their neighbor objects. '
                            'If the user is in front of a crosswalk, look for the traffic light and tell me the color of the light. ')
        list_prompt1.append(f'2) Explain the destination, represented by a orange {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt1.append(f'3) Describe the red path that starts at the yellow dot at the bottom center of the image and ends at the orange dot {dest_descriptions} in the the image. '
                           'Focus on the direction of the path and any obstacles present along it, rather than on the yellow starting point, the orange endpoint, or the red line itself. ')

        list_prompt2 = []
        list_prompt2.append('4) Select the most appropriate action to follow the path described above from the following four options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Stop and wait\'. '
                           'Say only the answer. If there is a red traffic light, say \'Stop and wait\' option. '
                           )
        list_prompt2.append('5) Then, explain the reason in 1 line. ')
        list_prompt2.append('6) What obstacles are on the path described? Enumerate one by one. ')

        if trial_num == 540211:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
        elif trial_num == 540212:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
            list_prompt = [' '.join(list_prompt)]
        elif trial_num == 540213:
            list_prompt1 = [' '.join(list_prompt1)]
            list_prompt2 = [' '.join(list_prompt2)]
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)



    # one turn
    elif trial_num in [540301, 540302, 540303]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the user is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt1 = []
        list_prompt1.append('1) Describe the overall photo from near to far. '
                            'Focus on the objects, their relative positions in the image, and its positions with their neighbor objects. '
                            'If the user is in front of a crosswalk, look for the traffic light and tell me the color of the light. ')
        list_prompt1.append(f'2) Explain the destination, represented by a orange {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt1.append(f'3) Describe the red path that starts at the yellow dot at the bottom center of the image and ends at the orange dot {dest_descriptions} in the the image . '
                           'Focus on the direction of the path and any obstacles present along it, rather than on the yellow starting point, the orange endpoint, or the red line itself. '
                           'The horizontal direction of the destination is '+ str_dir +' from the user. ')

        list_prompt2 = []
        list_prompt2.append('4) If there is a red traffic light, say \'Stop and wait\' option. '
                            'Otherwise, select the most appropriate action to follow the path described above from the following three options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', and \'Go diagonally to the right\'. '
                           'Say only the answer. '
                           )
        list_prompt2.append('5) Then, explain the reason in 1 line. ')
        list_prompt2.append('6) What obstacles are on the path described? Enumerate one by one. ')

        if trial_num == 540301:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
        elif trial_num == 540302:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
            list_prompt = [' '.join(list_prompt)]
        elif trial_num == 540303:
            list_prompt1 = [' '.join(list_prompt1)]
            list_prompt2 = [' '.join(list_prompt2)]
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)


    # one turn
    elif trial_num in [540311, 540312, 540313]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the user is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt1 = []
        list_prompt1.append('1) Describe the overall photo from near to far. '
                            'Focus on the objects, their relative positions in the image, and its positions with their neighbor objects. '
                            'If the user is in front of a crosswalk, look for the traffic light and tell me the color of the light. ')
        list_prompt1.append(f'2) Explain the destination, represented by a orange {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt1.append(f'3) Describe the red path that starts at the yellow dot at the bottom center of the image and ends at the orange dot {dest_descriptions} in the the image. '
                           'Focus on the direction of the path and any obstacles present along it, rather than on the yellow starting point, the orange endpoint, or the red line itself. ')

        list_prompt2 = []
        list_prompt2.append('4) If there is a red traffic light, say \'Stop and wait\' option. '
                            'Otherwise, select the most appropriate action to follow the path described above from the following three options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', and \'Go diagonally to the right\'. '
                           'Say only the answer. '
                           )
        list_prompt2.append('5) Then, explain the reason in 1 line. ')
        list_prompt2.append('6) What obstacles are on the path described? Enumerate one by one. ')

        if trial_num == 540311:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
        elif trial_num == 540312:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
            list_prompt = [' '.join(list_prompt)]
        elif trial_num == 540313:
            list_prompt1 = [' '.join(list_prompt1)]
            list_prompt2 = [' '.join(list_prompt2)]
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)



    # multi turn, one turn, two turn
    elif trial_num in [41121, 41122, 41123]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The viewer is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the viewer is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt.append('1) Describe the overall photo from near to far. Focus on the objects, their relative positions within the image, and their locations in relation to other neighbor objects. ')
        list_prompt.append(f'2) What is near the viewer? Describe the object with its relative position in the image in 1 line. '
                           'Then, if that object restricts the viewer\'s movement, describe in which direction the viewer can move. ')
        list_prompt.append(f'3) Explain the destination, {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt.append('4) Is the designated destination in a restricted or inaccessible area? ')
        list_prompt.append('5) Are there any unavoidable obstacles, restricted areas, or red pedestrian traffic lights between the current location of the viewer and the designated destination? ')

        if trial_num == 41123:
            list_prompt = [' '.join(list_prompt)]

        list_prompt.append(f'6) Based on the information provied so far, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. '
                           'The horizontal direction of the destination is '+ str_dir +' from the viewer. ')
        list_prompt.append('7) Which action do you recommend to the viewer who wants to go to the destination? '
                           'Please choose an action in the following four options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Stop and wait\'. '
                           'Say only the answer. If there is a potential danger for the viewer, say \'Stop and wait\' option. '
                           )
        list_prompt.append('8) Then, explain the reason in 1 line. ')

        if trial_num == 41122:
            list_prompt = [' '.join(list_prompt)]


    elif trial_num in [10001]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.\n'
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt.append(f'Can a viewer who is a pedestrian, move forward? ' 
                           'If not possible, say \'stop and wait\'. '
                           'If possible, which direction is the most efficent to reach the destination? '
                           'The horizontal direction of the destination is '+ str_dir +' from the viewer. '
                           'Choose an action in the following four options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', and \'Go diagonally to the right\'. '
        )

    
    # multi turn, one turn, two turn
    elif trial_num in [10002]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                'The viewer\'s position in near the bottom center of the image, looking towards the center of the image. '
                "Consider the starting point as the ground where the viewer is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt.append('1) Describe the overall photo from near to far. Focus on the objects, their relative positions within the image, and their locations in relation to other neighbor objects. ')
        list_prompt.append(f'2) What is near the viewer at the bottom center of the image? Describe the object with its relative position in the image in 1 line. '
                           'Then, if that object restricts the viewer\'s movement, describe in which direction the viewer can move. '
                           'If the viewer cannot move in all directions, say \'The viewer have to stop and wait\'. ')
        list_prompt1 = [' '.join(list_prompt)]

        list_prompt = []
        list_prompt.append(f'3) Explain the destination, {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt.append('4) Is the designated destination in a restricted or inaccessible area? ')
        list_prompt.append(f'5) Based on the information provied so far, explain the path from the viewer\'s position to the designated destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. '
                           'The horizontal direction of the destination is '+ str_dir +' from the viewer. '
                           'If the viewer cannot move in all directions, say \'The viewer have to stop and wait\'. ')
        list_prompt2 = [' '.join(list_prompt)]

        list_prompt = []
        list_prompt.append('6) Based on the previous conversation, which action would be appropriate for the viewer at the bottom center of the image who wants to go to the designated destination? '
                           'Choose an action in the following four options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Stop and wait\'. '
                           'Say only the answer. If there is a potential danger for the viewer, say \'Stop and wait\' option. '
                           'The potential dangers include encountering an unavoidable obstacle near the viewer, reaching a destination in a restricted area or finding a red traffic light in the image. '
                           )
        list_prompt.append('7) Then, explain the reason in 1 line. ')

        list_prompt3 = [' '.join(list_prompt)]

        list_prompt = []
        list_prompt.extend(list_prompt1)
        list_prompt.extend(list_prompt2)
        list_prompt.extend(list_prompt3)

    # multi turn, one turn, two turn
    elif trial_num in [40001, 40002, 40003]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                'The photographer\'s position in near the bottom center of the image, looking towards the center of the image. '
                "Consider the starting point as the ground where the photographer is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        list_prompt1 = []
        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt1.append('1) Describe the overall photo from near to far for the visually imparied person. Focus on the objects and their relative positions in the image. ')
        list_prompt1.append(f'2) What is near the photographer? Describe the object with its relative position in the image in 1 line. '
                           'Then, if that object restricts the photographer\'s movement, describe in which direction the photographer can move. ')
        list_prompt1.append(f'3) Explain the destination, {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt1.append('4) Are there any obstacles, restricted areas, or red pedestrian traffic lights in the image? Where are they? ')

        list_prompt2 = []
        list_prompt2.append(f'5) Based on the description, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. '
                           'The horizontal direction of the destination is '+ str_dir +' from the photographer. ')
        list_prompt2.append('6) Which action do you recommend to the photographer who wants to go to the destination? '
                           'Please choose an action in the following eight options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Wait and find alternative path\', '
                           '\'Go diagonally to the left to avoid obstacles\', \'Go diagonally to the right to avoid obstacles\', \'Wait until changing the traffic light\', and \'Wait until removing obstacles\'. '                           
                           )
        list_prompt2.append('7) Then, explain the reason in 1 line. ')

        if trial_num == 40001:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
        elif trial_num == 40002:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
            list_prompt = [' '.join(list_prompt)]
        elif trial_num == 40003:
            list_prompt1 = [' '.join(list_prompt1)]
            list_prompt2 = [' '.join(list_prompt2)]
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)

    # multi turn, one turn, two turn
    elif trial_num in [40006, 40007, 40008]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                'The photographer\'s position in near the bottom center of the image, looking towards the center of the image. '
                "Consider the starting point as the ground where the photographer is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        list_prompt1 = []
        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt1.append('1) Describe the overall photo from near to far for the visually imparied person. Focus on the objects and their relative positions in the image. ')
        list_prompt1.append(f'2) What is near the photographer? Describe the object with its relative position in the image in 1 line. '
                           'Then, say which object in the image restricts the photographer\'s movement. ')
        list_prompt1.append(f'3) Explain the destination, {dest_descriptions}, with its relative position in the image in 1 line. ')

        list_prompt2 = []
        list_prompt2.append(f'4) Based on the description, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path. '
                            'The horizontal direction of the destination is '+ str_dir +' from the photographer. '
                            'The photographer can select and combine the following actions: '
                            '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Wait and find alternative path\', '
                            '\'Go diagonally to the left to avoid obstacles\', \'Go diagonally to the right to avoid obstacles\', \'Wait until changing the traffic light\', and \'Wait until removing obstacles\'. ')
        list_prompt2.append('5) Which action do you recommend to the photographer at the first? Then, explain the reason in 1 line.')


        if trial_num == 40006:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
        elif trial_num == 40007:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
            list_prompt = [' '.join(list_prompt)]
        elif trial_num == 40008:
            list_prompt1 = [' '.join(list_prompt1)]
            list_prompt2 = [' '.join(list_prompt2)]
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)


    # multi turn, one turn, two turn
    elif trial_num in [40036, 40037, 40038]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0. '
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                'The photographer\'s position in near the bottom center of the image, looking towards the center of the image. '
                "Consider the starting point as the ground where the photographer is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        list_prompt1 = []
        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt1.append('1) Describe the overall photo from near to far for the visually imparied person. Focus on the objects and their relative positions in the image. ')
        list_prompt1.append('If the photographer is in front of a crosswalk, look for the traffic light and tell me the color of the light. ')
        list_prompt1.append(f'2) What is near the letter \'A\' at the bottom center [0.5, 1.0]? Ignore the letters and describe the object with its relative position in the image in 1 line. '
                           'Then, say which object in the image restricts the photographer\'s movement. ')
        list_prompt1.append(f'3) Explain the destination, represented by a letter \'Z\' {dest_descriptions}, with its relative position in the image in 1 line. ')

        list_prompt2 = []
        list_prompt2.append(f'4) Say all letters from \'A\', located at the bottom center [0.5, 1.0], to \'Z\' at {dest_descriptions}, while avoiding all objects. Please say the letters in sequence. ')
        list_prompt2.append(f'5) Based on the previous letter sequence, explain the path on the sequence, paying attention to obstacles along the path. '
                            'The photographer can select and combine the following actions: '
                            '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Wait and find alternative path\', '
                            '\'Go diagonally to the left to avoid obstacles\', \'Go diagonally to the right to avoid obstacles\', \'Wait until changing the traffic light\', and \'Wait until removing obstacles\'. ')
        list_prompt2.append('6) Which action do you recommend to the photographer at the first? Then, explain the reason in 1 line.')


        if trial_num == 40036:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
        elif trial_num == 40037:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
            list_prompt = [' '.join(list_prompt)]
        elif trial_num == 40038:
            list_prompt1 = [' '.join(list_prompt1)]
            list_prompt2 = [' '.join(list_prompt2)]
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)




    # multi turn, one turn, two turn
    elif trial_num in [40011, 40012, 40013]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: left-top [0.0, 0.0], right-bottom [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                'The photographer\'s position in near the bottom center of the image, looking towards the center of the image. '
                "Consider the starting point as the ground where the photographer is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        list_prompt1 = []
        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt1.append('1) Describe the overall photo from near to far for the visually imparied person. Focus on the objects and their relative positions in the image. ')
        list_prompt1.append(f'2) What is near the photographer, represented by a yellow point at the center-bottom in the image? Describe the object with its relative position in the image in 1 line. '
                           'Then, say which object in the image restricts the photographer\'s movement. ')
        list_prompt1.append(f'3) Explain the destination, represented by a orange {dest_descriptions}, with its relative position in the image in 1 line. ')

        list_prompt2 = []
        list_prompt2.append(f'4) Based on the description, explain the path to the destination, represented by a orange {dest_descriptions}, paying attention to obstacles along the path. '
                            'The horizontal direction of the destination is '+ str_dir +' from the photographer. '
                            'The photographer can select and combine the following actions: '
                            '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Wait and find alternative path\', '
                            '\'Go diagonally to the left to avoid obstacles\', \'Go diagonally to the right to avoid obstacles\', \'Wait until changing the traffic light\', and \'Wait until removing obstacles\'. ')
        list_prompt2.append('5) Which action do you recommend to the photographer at the first? Then, explain the reason in 1 line.')


        if trial_num == 40011:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
        elif trial_num == 40012:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
            list_prompt = [' '.join(list_prompt)]
        elif trial_num == 40013:
            list_prompt1 = [' '.join(list_prompt1)]
            list_prompt2 = [' '.join(list_prompt2)]
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)

    
    # multi turn, one turn, two turn
    elif trial_num in [40016, 40017, 40018]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: left-top [0.0, 0.0], right-bottom [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                'On the right side of the image, there is a purple border. On the left side of the image, there is a pink border. '
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                'The photographer\'s position in near the bottom center of the image, looking towards the center of the image. '
                "Consider the starting point as the ground where the photographer is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        list_prompt1 = []
        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt1.append('1) Describe the overall photo from near to far for the visually imparied person. Focus on the objects and their relative positions in the image. ')
        list_prompt1.append(f'2) What is near the photographer, represented by a yellow point at the center-bottom in the image? Describe the object with its relative position in the image in 1 line. '
                           'Then, say which object in the image restricts the photographer\'s movement. ')
        list_prompt1.append(f'3) Explain the destination, represented by a orange {dest_descriptions}, with its relative position in the image in 1 line. ')

        list_prompt2 = []
        list_prompt2.append(f'4) Based on the description, explain the path to the destination, represented by a orange {dest_descriptions}, paying attention to obstacles along the path. '
                            'The horizontal direction of the destination is '+ str_dir +' from the photographer. '
                            'The photographer can select and combine the following actions: '
                            '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Wait and find alternative path\', '
                            '\'Go diagonally to the left to avoid obstacles\', \'Go diagonally to the right to avoid obstacles\', \'Wait until changing the traffic light\', and \'Wait until removing obstacles\'. '
                            'Moving in the left direction will bring the photographer closer to the pink border on the left of the image. '
                            'Moving in the right direction will bring the photographer closer to the purple border on the right of the image. ')
        list_prompt2.append('5) Which action do you recommend to the photographer at the first? Then, explain the reason in 1 line.')


        if trial_num == 40016:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
        elif trial_num == 40017:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
            list_prompt = [' '.join(list_prompt)]
        elif trial_num == 40018:
            list_prompt1 = [' '.join(list_prompt1)]
            list_prompt2 = [' '.join(list_prompt2)]
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)





    # multi turn, one turn, two turn
    elif trial_num in [40021, 40022, 40023]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0. '
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                'The photographer is looking at center [0.5, 0.5] of the image. '
                "Consider the starting point as the ground where the photographer is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. \n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        list_prompt1 = []
        list_prompt2 = []
        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt1.append('1) Describe the overall photo from near to far. ')
        list_prompt1.append(f'2) Explain the destination, represented by a orange {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt1.append(f'3) Explain the path to the destination, represented by a orange {dest_descriptions}, paying attention to obstacles along the path in 1 line. '
                           'The horizontal direction of the destination is '+ str_dir +' from the photographer.')


        list_prompt2.append('4) What obstacles are on the path described? Enumerate one by one. ')
        list_prompt2.append('5) Which action do you recommend to the photographer who wants to go to the destination? '
                            'Please choose an action in the following options: '
                            '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', \'Wait and find alternative path\', '
                            '\'Go diagonally to the left to avoid obstacles\', \'Go diagonally to the right to avoid obstacles\', \'Wait until changing the traffic light to blue\', and \'Wait until removing obstacles\'. ')
        list_prompt2.append('6) Then, explain the reason in 1 line.')


        if trial_num == 40021:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
        elif trial_num == 40022:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
            list_prompt = [' '.join(list_prompt)]
        elif trial_num == 40023:
            list_prompt1 = [' '.join(list_prompt1)]
            list_prompt2 = [' '.join(list_prompt2)]
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)


    # multi turn, one turn, two turn
    elif trial_num in [40026, 40027, 40028]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0. '
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                'The photographer is looking at center [0.5, 0.5] of the image. '
                "Consider the starting point as the ground where the photographer is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. \n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        list_prompt1 = []
        list_prompt2 = []
        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt1.append('1) Describe the overall photo from near to far. If the photographer is in front of a crosswalk, look for the traffic light and tell me the color of the light. ')
        list_prompt1.append(f'2) Explain the destination, represented by a orange {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt1.append(f'3) Explain the path to the destination, represented by a orange {dest_descriptions}, paying attention to obstacles along the path in 1 line. '
                           'The horizontal direction of the destination is '+ str_dir +' from the photographer.')


        list_prompt2.append('4) What obstacles are on the path described? Enumerate one by one. ')
        list_prompt2.append('5) Which action do you recommend to the photographer who wants to go to the destination? '
                            'Please choose an action in the following options: '
                            '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', \'Wait and find alternative path\', '
                            '\'Go diagonally to the left to avoid obstacles\', \'Go diagonally to the right to avoid obstacles\', \'Wait until changing the traffic light to blue\', \'Wait until removing obstacles\'. '
                            '\'Go forward while curving slightly to the right\', \'Go forward while curving slightly to the left\', \'Go forward while detouring to the right\', and \'Go forward while detouring to the left\'')
        list_prompt2.append('6) Then, explain the reason in 1 line.')

        if trial_num == 40026:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
        elif trial_num == 40027:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
            list_prompt = [' '.join(list_prompt)]
        elif trial_num == 40028:
            list_prompt1 = [' '.join(list_prompt1)]
            list_prompt2 = [' '.join(list_prompt2)]
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)

    # multi turn, one turn, two turn
    elif trial_num in [40031, 40032, 40033]:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right and closer to purple border; y moves down. Decreasing x moves left and closer to pink border; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0. '
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                'The photographer is looking at center [0.5, 0.5] of the image. '
                "Consider the starting point as the ground where the photographer is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. \n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        list_prompt1 = []
        list_prompt2 = []
        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt1.append('1) Describe the overall photo from near to far. If the photographer is in front of a crosswalk, look for the traffic light and tell me the color of the light. ')
        list_prompt1.append(f'2) Explain the destination, represented by a orange {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt1.append(f'3) Explain the path to the destination, represented by a orange {dest_descriptions}, paying attention to obstacles along the path in 1 line. '
                           'The horizontal direction of the destination is '+ str_dir +' from the photographer.')


        list_prompt2.append('4) What objects are on the path described? Enumerate one by one. ')
        list_prompt2.append('5) Which action do you recommend to the photographer who wants to go to the destination? '
                            'Please choose an action in the following options: '
                            '\'Go straight to the center\', \'Go diagonally to the pink border\', \'Go diagonally to the purple border\', \'Wait and find alternative path\', '
                            '\'Go diagonally to the pink border to avoid obstacles\', \'Go diagonally to the purple border to avoid obstacles\', \'Wait until changing the traffic light to blue\', \'Wait until removing obstacles\', '
                            '\'Go forward while curving slightly to the purple border\', \'Go forward while curving slightly to the pink border\', \'Go forward while detouring to the purple border\', and \'Go forward while detouring to the pink border\'. '
                            'If the traffic light with a red light is visible, then select \'Wait until changing the traffic light to blue\' ')
        list_prompt2.append('6) Then, explain the reason in 1 line.')

        if trial_num == 40031:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
        elif trial_num == 40032:
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)
            list_prompt = [' '.join(list_prompt)]
        elif trial_num == 40033:
            list_prompt1 = [' '.join(list_prompt1)]
            list_prompt2 = [' '.join(list_prompt2)]
            list_prompt.extend(list_prompt1)
            list_prompt.extend(list_prompt2)


    # multi turn, one turn, two turn
    elif trial_num in [41126, 41127]:   # 41128 is already used in other places.
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                'The viewer\'s position in near the bottom center of the image, looking towards the center of the image. '
                "Consider the starting point as the ground where the viewer is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt.append('1) Describe the overall photo from near to far. Focus on the objects, their relative positions within the image, and their locations in relation to other neighbor objects. ')
        list_prompt.append(f'2) What is near the viewer at the bottom center of the image? Describe the object with its relative position in the image in 1 line. '
                           'Then, if that object restricts the viewer\'s movement, describe in which direction the viewer can move. ')
        list_prompt.append(f'3) Explain the destination, {dest_descriptions}, with its relative position in the image in 1 line. ')
        list_prompt.append('4) Is the designated destination in a restricted or inaccessible area? ')
        list_prompt.append('5) Are there any obstacles, restricted areas, or red pedestrian traffic lights between the viewer\'s position and the designated destination? ')

        # if trial_num == 41128:
        #     list_prompt = [' '.join(list_prompt)]

        list_prompt.append(f'6) Based on the information provied so far, explain the path from the viewer\'s position to the designated destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. '
                           'The horizontal direction of the destination is '+ str_dir +' from the viewer. ')
        list_prompt.append('7) Which action do you recommend to the viewer at the bottom center of the image who wants to go to the designated destination? '
                           'Please choose an action in the following four options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Stop and wait\'. '
                           'Say only the answer. If there is a potential danger for the viewer, say \'Stop and wait\' option. '
                           'The potential dangers include encountering an unavoidable obstacle on their path, coming across a red traffic light, or reaching a destination in a restricted area. '
                           )
        list_prompt.append('8) Then, explain the reason in 1 line. ')

        if trial_num == 41127:
            list_prompt = [' '.join(list_prompt)]


    elif trial_num in [41829, 41869]:   # two-turn query
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '
                'Image size: 1.0x1.0.'
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the user is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                )
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir = get_direction(goal_cxcy[0])
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains objects, {bbox_list_str}. \n"))
        list_prompt.append(f'First, describe the overall photo from near to far. ')
        list_prompt.append(
            'The horizontal direction of the destination is ' + str_dir + ' from the user. '
            f'Second, explain the destination, {dest_descriptions}, with its relative position in the image in 1 line, then describe the path to the destination, paying attention to obstacles along the path in 1 line. '
        )
        list_prompt = [' '.join(list_prompt)]

        if trial_num == 41829:
            list_prompt.append('Based on the description, choose the appropriate action to reach the destination at the first step from the following options: '
                               '\'A) Go straight to the center\', \'B) Go diagonally to the left\', \'C) Go diagonally to the right\', \'D) Stop and wait\'. '
				'Say only the answer. If there is a potential danger for the user, say \'D) Stop and wait\' option. '
                               )
        elif trial_num == 41869:
            list_prompt.append('Based on the description, choose the appropriate action to reach the destination at the first step from the following options: '
                               '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', \'Stop and wait\'. '
				'Say only the answer. If there is a potential danger for the user, say \'Stop and wait\' option. '
				)

        list_prompt.append(
            'Based on the description, what obstacles are on the path? List one by one. Say only the answer. Use a comma as a separator. If there is no obstacles, say "no obstacles".'
        )
        



    elif trial_num == 20:   # one-turn query without description
        if sep_system:
            list_system.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_system.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                                )
            list_system = [' '.join(list_system)]
        else:
            list_prompt.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_prompt.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                                )
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains obstacles, {bbox_list_str}.\n"))
        list_prompt.append(f'First, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line.')
        list_prompt.append(f'Second, choose GO or STOP with the reason in 1 line.')
        list_prompt = [' '.join(list_prompt)]

    elif trial_num == 120:   # one-turn query without description
        if sep_system:
            list_system.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_system.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                                )
            list_system = [' '.join(list_system)]
        else:
            list_prompt.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
            list_prompt.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                                "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                                "Consider the starting point as the ground where the user is standing. "
                                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
                                )
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains obstacles, {bbox_list_str}.\n"))
        list_prompt.append(f'First, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line.')
        list_prompt.append(f'Second, choose: go left 45 degree, go straight, go right 45 degree, or stop, with the reason in 1 line.')
        list_prompt = [' '.join(list_prompt)]

    elif trial_num == 500:
        # reordering questions
        list_prompt.append('A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. Image size: 1.0x1.0.')
        list_prompt.append("The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                            "In this image, the user's starting point is situated at the bottom-center point [0.5, 1.0] of the image. "
                            "Consider the starting point as the ground where the user is standing.\n"
                            )
        if len(bboxes) > 0:
            list_prompt.append((f"The image contains obstacles, {bbox_list_str}.\n"))
        list_prompt.append(f'After describing the overall photo from near to far, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path.')
        list_prompt.append("Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. ")
        list_prompt = [' '.join(list_prompt)]

        # Summarize prompt
        list_prompt.append('Summarize the path to the destination using direction and distance in 1 line. ')
        list_prompt.append('Select GO or STOP based on common sense. Tell in 1 line about thought process leading up to the decision. ')
        list_prompt.append('If select GO, then enumerate the only obstacles on the path with its relative position to the bottom-center point of the image. If select STOP, describe the object that caused it to stop.')
        
    else:
        raise AssertionError(f'{trial_num} is not supported')

    return list_prompt, list_system