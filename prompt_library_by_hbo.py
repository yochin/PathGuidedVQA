def get_prompt_by_hbo(goal_label_cxcy, bboxes, trial_num, sep_system=False):
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
    if trial_num == 1118:
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
        list_prompt.append(f'Describe the overall photo from near to far.')
        list_prompt = [' '.join(list_prompt)]

        # Summarize prompt
        list_prompt.append(f'Explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. ')
        list_prompt.append('What obstacles are on the path described? Enumerate one by one. ')
        list_prompt.append('What action do you recommend? Please choose from the following options. A) Go straight, B) Go left 45, C) Go right 45, D) Stop. Then, explain the reason in 1 line. ')

    elif trial_num == 1119:   # one-turn query
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
        list_prompt.append(f'Fourth, what action do you recommend? Please choose from the following options. A) Go straight, B) Go left 45, C) Go right 45, D) Stop. Then, explain the reason in 1 line.')
        list_prompt = [' '.join(list_prompt)]

    # by hbo
    elif trial_num == 91118:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals.'
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0].'
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2].'
                'Image size: 1.0x1.0.')
            list_system.append(
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                "In this image, an user's starting point is situated at the bottom-center point [0.5, 1.0] of the image."
                "If the user "
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
        list_prompt.append(
            f'Explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. ')
        list_prompt.append('What obstacles are on the path described? Enumerate one by one. ')
        list_prompt.append(
            'What action do you recommend to a visually impaired person? '
            'Please choose from the following four options: '
            '\'Go straight\', \'Turn slightly to the left and go\', \'Turn slightly to the right and go\', and \'Stop\'. '
            'Then, explain the reason in 1 line. '
            'For example, if the destination is [0.0, 0.5] and the path is clear, choose \'Turn slightly to the left and go\''
            'If the destination is [1.0, 0.5] and the path is clear, choose \'Turn slightly to the right and go\''
            )
    else:
        raise AssertionError(f'{trial_num} is not supported')

    return list_prompt, list_system