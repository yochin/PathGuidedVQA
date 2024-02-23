def get_prompt(goal_label_cxcy, bboxes, trial_num, sep_system=False):
    # 각 바운딩 박스에 대한 설명 구성
    # bbox_descriptions = [f"{label} at ({round(x_min, 2)}, {round(y_min, 2)}, {round(x_max, 2)}, {round(y_max, 2)})" for label, (x_min, y_min, x_max, y_max), _ in bboxes]
    bbox_descriptions = [f"{label} [{round(x_min, 2)}, {round(y_min, 2)}, {round(x_max, 2)}, {round(y_max, 2)}]" for label, (x_min, y_min, x_max, y_max), _ in bboxes]
    bbox_list_str = ", ".join(bbox_descriptions)
    goal_label, goal_cxcy = goal_label_cxcy

    if len(goal_cxcy) == 2:     # point
        # dest_descriptions = f"{goal_label} at ({round(goal_cxcy[0], 2)}, {round(goal_cxcy[1], 2)})"
        dest_descriptions = f"{goal_label} [{round(goal_cxcy[0], 2)}, {round(goal_cxcy[1], 2)}]"
    elif len(goal_cxcy) == 4:   # bbox
        dest_descriptions = f"{goal_label} [{round(goal_cxcy[0], 2)}, {round(goal_cxcy[1], 2)}, {round(goal_cxcy[2], 2)}, {round(goal_cxcy[3], 2)}]"
    else:
        raise AssertionError('check ', goal_cxcy)


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