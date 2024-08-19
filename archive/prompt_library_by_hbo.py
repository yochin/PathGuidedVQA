def get_direction(x):
    if x <= 0.4:
        return 'diagonally to the left'
    elif x >= 0.6:
        return 'diagonally to the right'
    else:
        return 'in the center'

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
        list_prompt.append('1) Describe the overall photo from near to far.')
        list_prompt.append(f'2) Explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. '
                           'The horizontal direction of the destination is '+ str_dir +' from the user.')
        list_prompt.append('3) What obstacles are on the path described? Enumerate one by one.')
        list_prompt.append('4) Which action do you recommend to the user who wants to go to the destination? '
                           'Please choose an action in the following four options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Stop\'.')
        list_prompt.append('5) Then, explain the reason in 1 line.')
        #list_prompt = [' '.join(list_prompt)]

        # by hbo
    elif trial_num == 91128:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                "Do use only English. "
                "This input is for system, so don't generate any answer now. "
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                "Consider the starting point [0.5, 1.0] as the ground where the user (pedestrian) is standing."
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Explain as if you were a navigation assistant explaining to the user who is a visually impaired person."
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
            )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))

            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        str_dir = get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt.append(
            '1) The horizontal direction of the destination is ' + str_dir + ' from the user. '
            f'Explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. ')
        list_prompt.append('2) What obstacles are on the path described? Enumerate one by one.')
        list_prompt.append('3) Which action do you recommend to the user who wants to go to the destination? '
                           'If there\'s a possibility of danger to visually impaired people, do not hesitate to do \'Stop\'. '
                           'Choose an action in the following four options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Stop\'. '
                           )
        #list_prompt = [' '.join(list_prompt)]

    elif trial_num == 91138:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes."
                "Consider the starting point [0.5, 1.0] as the ground where the user (pedestrian) is standing."
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Explain as if you were a navigation assistant explaining to the user who is a visually impaired person."
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property.\n"
            )
            if len(bboxes) > 0:
                list_system.append((f"The image contains obstacles, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')
        list_prompt = [' '.join(list_prompt)]

        str_dir = get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt.append(
            f'1) Explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. '
            'The horizontal direction of the destination is ' + str_dir + ' from the user.')
        list_prompt.append('2) What obstacles are on the path described? Enumerate one by one.')
        list_prompt.append('3) Which action do you recommend to the user who wants to go to the destination? '
                           'Please choose an action in the following four options: '
                           '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Stop\'.')
        list_prompt = [' '.join(list_prompt)]

    elif trial_num in [91148]:  # two-turn query
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
            list_prompt.append((f"The image contains objects, {bbox_list_str}.\n"))
        list_prompt.append(f'First, describe the overall photo from near to far.')
        list_prompt.append(
            'The horizontal direction of the destination is ' + str_dir + ' from the user.'
            f'Second, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. '
        )
        list_prompt = [' '.join(list_prompt)]

        list_prompt.append(
            'Based on the description, choose the appropriate action to reach the destination at the first from the following options: '
            '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Stop\'. '
            'Say only the answer. If there\'s a potential danger for the user, do not hesitate to choose \'Stop\' option. '
        )
        list_prompt.append(
            'Based on the description, what obstacles are on the path? List one by one. Say only the answer. Use a comma as a separator. If there is no obstacles, say "no obstacles".')

    elif trial_num in [91158]:  # two-turn query
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
            list_prompt.append((f"The image contains objects, {bbox_list_str}.\n"))
        list_prompt.append(f'First, describe the overall photo from near to far.')
        list_prompt.append(
            'The horizontal direction of the destination is ' + str_dir + ' from the user.'
            f'Second, explain the path to the destination, {dest_descriptions}, paying attention to obstacles along the path in 1 line. '
        )
        list_prompt = [' '.join(list_prompt)]

        list_prompt.append(
            'Based on the description, choose the appropriate action to reach the destination from the following options: '
            '\'Go straight to the center\', \'Go diagonally to the left\', \'Go diagonally to the right\', and \'Stop\'. '
            'Say only the answer. If the user needs to wait first, the \'Stop\' option must be chosen for safety. '
        )
        list_prompt.append(
            'Based on the description, what obstacles are on the path? List one by one. Say only the answer. Use a comma as a separator. If there is no obstacles, say "no obstacles".')


    else:
        raise AssertionError(f'{trial_num} is not supported')

    return list_prompt, list_system