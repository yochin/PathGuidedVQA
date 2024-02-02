import openai
import base64
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, init_llava_model, run_llava_model
from transformers import set_seed
import pdb


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

class LargeMultimodalModels():
    def __init__(self, model_name=None, llava_model_base_path=None, llava_model_path=None, ferret_model_path=None):
        self.possible_models = ['dummy', 'llava', 'chatgpt', 'ferret']

        self.model_name = model_name

        assert self.model_name in self.possible_models

        # for gpt4
        if self.model_name == 'chatgpt':
            self.OPENAI_API_KEY = "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"
        # for llava
        elif self.model_name == 'llava':
            self.llava_model_path = llava_model_path
            self.llava_model_base_path = llava_model_base_path
            self.llava_tokenizer, self.llava_model, self.llava_image_processor, \
            self.llava_context_len, self.llava_model_name = init_llava_model(model_path=self.llava_model_path,
                                                                             model_base=self.llava_model_base_path)
        # for ferret
        elif self.model_name == 'ferret':
            self.ferret_model_path = ferret_model_path
        

    def describe_images_with_boxes(self, image_path, bboxes, goal_label_cxcy, order, num_total, merge=False, previous_descriptions=[]):
        print('\nimage_path:', image_path)
        print('bboxes:', bboxes)
        print('goal_label_cxcy:', goal_label_cxcy)
        print('order:', order)
        print('num_total:', num_total)
        print('merge:', merge)
        print('previous_descriptions:', previous_descriptions)

        res_answer = ''
        res_query = ''

        if self.model_name == 'dummy':
            res_answer = self.describe_all_bboxes_with_dummy()
        elif self.model_name in ['llava']:
            res_query, res_answer = self.describe_all_bboxes_with_llava(image_path, bboxes, goal_label_cxcy, order, num_total, merge, previous_descriptions)
        elif self.model_name == 'chatgpt':
            res_answer = self.describe_all_bboxes_with_chatgpt(image_path, bboxes, goal_label_cxcy)
        else:
            raise AssertionError(f'{self.model_name} is not supported!')
        
        return res_query, res_answer
    
    def describe_whole_images_with_boxes(self, image_path, bboxes, goal_label_cxcy, step_by_step=False):
        print('\nimage_path:', image_path)
        print('bboxes:', bboxes)
        print('goal_label_cxcy:', goal_label_cxcy)

        res_answer = ''
        res_query = ''

        if self.model_name == 'dummy':
            res_answer = self.describe_all_bboxes_with_dummy()
        elif self.model_name in ['llava', 'ferret']:
            res_query, res_answer = self.describe_all_bboxes_with_llava_ferret_in_whole_image(image_path, bboxes, goal_label_cxcy, step_by_step, self.model_name)
        elif self.model_name == 'chatgpt':
            res_answer = self.describe_all_bboxes_with_chatgpt(image_path, bboxes, goal_label_cxcy)
        else:
            raise AssertionError(f'{self.model_name} is not supported!')
        
        return res_query, res_answer


    def describe_all_bboxes_with_dummy(self):
        answer = 'hello world'

        return answer
    

    def describe_all_bboxes_with_chatgpt(self, image_path, bboxes, goal_label_cxcy):
        # 이미지를 base64로 인코딩
        encoded_image = encode_image_to_base64(image_path)

        # 각 바운딩 박스에 대한 설명 구성
        bbox_descriptions = [f"{label} at ({x_min}, {y_min}, {x_max}, {y_max})" for label, (x_min, y_min, x_max, y_max), _ in bboxes]
        bbox_list_str = ", ".join(bbox_descriptions)
        goal_label, goal_cxcy = goal_label_cxcy
        dest_descriptions = f"{goal_label} at ({goal_cxcy[0]}, {goal_cxcy[1]})"

        # 프롬프트 구성
        prompt = (  "[Context: The input image depicts the view from a pedestrian's position, " 
                    "taken at a point 80cm above the ground for pedestrian navigation purposes. " 
                    "In this image, the user's starting point is situated below the center of the image at (0.5, 1.0). "
                    "Consider the starting point as the ground where the user is standing.]\n" 
                    f"[Obstacle Name at (bounding box): [{bbox_list_str}].]\n"
                    f"[Destination Name at (point): [{dest_descriptions}].]\n"
                    "Describe the obstacles to the destination in a natural and simple way "
                    "for a visually impaired person as a navigation assistant in 3 sentences. "
                    "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. ")

    
        # print("[PROMPT]: ", prompt)
        # OpenAI API 키 설정 (환경 변수에서 가져옴)
        openai.api_key = self.OPENAI_API_KEY
        completion = openai.chat.completions.create(
            #model = "gpt-4",
            model="gpt-4-1106-preview",
            #messages=[
            #    {
            #        "role": "user",
            #        "content": prompt,
            #    },
            messages=[
                {"role": "system", "content": "This is an image-based task."},
                {"role": "user", "content": encoded_image}, #, "mimetype": "image/jpeg"
                {"role": "user", "content": prompt},
            ],
            #max_tokens=1000,
        )

        answer = completion.choices[0].message.content

        # print("[ANSWER]: ", answer)

        return answer


    def order_to_str(self, order):
        # if order == 1:
        #     str_order = 'first'
        # elif order == 2:
        #     str_order = 'second'
        # elif order == 3:
        #     str_order = 'third'

        if order == 1:
            str_order = 'a starting point'
        elif order == 2:
            str_order = 'a midpoint'
        elif order == 3:
            str_order = 'a destination'
        else:
            raise AssertionError('Not implemented')
        
        return str_order
    

    def num_to_str(self, num_total):
        if num_total == 1:
            raise AssertionError('Not implemented')
        elif num_total == 2:
            str_num_total = 'two'
        elif num_total == 3:
            str_num_total = 'three'
        else:
            raise AssertionError('Not implemented')
        
        return str_num_total
    

    def describe_all_bboxes_with_llava(self, image_path, bboxes, goal_label_cxcy, order, num_total, merge=False, previous_descriptions=[]):

        # 각 바운딩 박스에 대한 설명 구성
        bbox_descriptions = [f"{label} at ({round(x_min, 2)}, {round(y_min, 2)}, {round(x_max, 2)}, {round(y_max, 2)})" for label, (x_min, y_min, x_max, y_max), _ in bboxes]
        bbox_list_str = ", ".join(bbox_descriptions)
        goal_label, goal_cxcy = goal_label_cxcy
        dest_descriptions = f"{goal_label} at ({round(goal_cxcy[0], 2)}, {round(goal_cxcy[1], 2)})"

        str_order = self.order_to_str(order)
        str_num_total = self.num_to_str(num_total)
        
        if merge:
            previous_prompt = []
            
            if len(previous_descriptions) == 3:
                previous_prompt.append('The description at a starting point is that ')
                previous_prompt.append(previous_descriptions[0])
                previous_prompt.append(' ')
                                
                previous_prompt.append('The description at a midpoint is that ')
                previous_prompt.append(previous_descriptions[1])
                previous_prompt.append(' ')

                previous_prompt.append('The description at a destination is that ')
                previous_prompt.append(previous_descriptions[2])
                previous_prompt.append(' ')

                previous_prompt = ' '.join(previous_prompt)


            else:
                raise AssertionError('Not implemented')

            # # 프롬프트 구성
            # prompt = (  f"[Context: The input image is whole image of the route to the destination. "
            #             "The input image depicts the view from a pedestrian's position, " 
            #             "taken at a point 80cm above the ground for pedestrian navigation purposes. " 
            #             "In this image, the user's starting point is situated below the center of the image at (0.5, 1.0). "
            #             "Consider the starting point as the ground where the user is standing.]\n" 
            #             f"[Obstacle Name at (bounding box): [{bbox_list_str}].]\n"
            #             f"[Destination Name at (point): [{dest_descriptions}].]\n"
            #             f"{previous_prompt}"
            #             "Describe the obstacles to the destination in a natural and simple way "
            #             "for a visually impaired person as a navigation assistant in 3 sentences. "
            #             "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. ")

            prompt = (  f"[Context: The input image depicts the view from a pedestrian's position, " 
                        "taken at a point 80cm above the ground for pedestrian navigation purposes. " 
                        "In this image, the user's starting point is situated below the center of the image at (0.5, 1.0). "
                        "Consider the starting point as the ground where the user is standing.]\n" 
                        f"[Obstacle Name at (bounding box): [{bbox_list_str}].]\n"
                        f"[Destination Name at (point): [{dest_descriptions}].]\n"
                        f"{previous_prompt}"
                        "Describe the obstacles to the destination in a natural and simple way "
                        "for a visually impaired person as a navigation assistant in 3 sentences. "
                        "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. ")
        else:
            # 프롬프트 구성
            # prompt = (  f"[Context: The input image is the {str_order} of {str_num_total} images of the route to the destination. "
            #             "The input image depicts the view from a pedestrian's position, " 
            #             "taken at a point 80cm above the ground for pedestrian navigation purposes. " 
            #             "In this image, the user's starting point is situated below the center of the image at (0.5, 1.0). "
            #             "Consider the starting point as the ground where the user is standing.]\n" 
            #             f"[Obstacle Name at (bounding box): [{bbox_list_str}].]\n"
            #             f"[Destination Name at (point): [{dest_descriptions}].]\n"
            #             "Describe the obstacles to the destination in a natural and simple way "
            #             "for a visually impaired person as a navigation assistant in 3 sentences. "
            #             "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. ")

            prompt = (  f"[Context: The input image is taken at {str_order} on the route to the destination. "
                        "The input image depicts the view from a pedestrian's position, " 
                        "taken at a point 80cm above the ground for pedestrian navigation purposes. " 
                        "In this image, the user's starting point is situated below the center of the image at (0.5, 1.0). "
                        "Consider the starting point as the ground where the user is standing.]\n" 
                        f"[Obstacle Name at (bounding box): [{bbox_list_str}].]\n"
                        f"[Destination Name at (point): [{dest_descriptions}].]\n"
                        "Describe the obstacles to the destination in a natural and simple way "
                        "for a visually impaired person as a navigation assistant in 3 sentences. "
                        "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. ")
        

        # llm_args = type('Args', (), {
        #     "model_path": llava_model_path,
        #     "model_base": llava_model_base_path,
        #     "model_name": get_model_name_from_path(llava_model_path),
        #     "query": prompt,
        #     "conv_mode": None,
        #     "image_file": image_path,
        #     "sep": ",",
        #     "temperature": 0,
        #     "top_p": None,
        #     "num_beams": 1,
        #     "max_new_tokens": 512
        # })()
        # res_answer = eval_model(llm_args)
        res_answer = run_llava_model(tokenizer=self.llava_tokenizer, model=self.llava_model, image_processor=self.llava_image_processor, context_len=self.llava_context_len, 
                                     input_query=prompt, image_files=image_path, input_conv_mode=None, input_temperature=input_temperature, input_top_p=None, input_num_beams=1, input_max_new_tokens=512, model_name=self.llava_model_name)



        # print('query: ', prompt)
        # print('answer: ', res_answer)

        return prompt, res_answer
    

    def describe_all_bboxes_with_llava_ferret_in_whole_image(self, image_path, bboxes, goal_label_cxcy, step_by_step=False, model_name=None):
        set_seed(42)
        input_temperature = 0.6
        input_top_p = 0.9

        # 각 바운딩 박스에 대한 설명 구성
        bbox_descriptions = [f"{label} at ({round(x_min, 2)}, {round(y_min, 2)}, {round(x_max, 2)}, {round(y_max, 2)})" for label, (x_min, y_min, x_max, y_max), _ in bboxes]
        bbox_list_str = ", ".join(bbox_descriptions)
        goal_label, goal_cxcy = goal_label_cxcy

        if len(goal_cxcy) == 2:     # point
            dest_descriptions = f"{goal_label} at ({round(goal_cxcy[0], 2)}, {round(goal_cxcy[1], 2)})"
        elif len(goal_cxcy) == 4:   # bbox
            dest_descriptions = f"{goal_label} at ({round(goal_cxcy[0], 2)}, {round(goal_cxcy[1], 2)}, {round(goal_cxcy[2], 2)}, {round(goal_cxcy[3], 2)})"
        else:
            raise AssertionError('check ', goal_cxcy)

        # "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n"
        # ours beloow prompt
        # "ASSISTANT:"

        if step_by_step:
            list_prompt = []
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

            list_answer = []
            for i_prompt, prompt in enumerate(list_prompt):
                if i_prompt == 0:
                    in_prompt = prompt
                else:
                    in_prompt = ' '.join([list_answer[i_prompt-1], prompt])

                if model_name == 'llava':
                    answer = run_llava_model(tokenizer=self.llava_tokenizer, model=self.llava_model, image_processor=self.llava_image_processor, context_len=self.llava_context_len, 
                                            input_query=in_prompt, image_files=image_path, input_conv_mode=None, input_temperature=input_temperature, input_top_p=input_top_p, input_num_beams=1, 
                                            input_max_new_tokens=512, model_name=self.llava_model_name)
                elif model_name == 'ferret':
                    pdb.set_trace()

                list_answer.append(answer)

            res_answer = '[!@#$NEXT!@#$]'.join(list_answer)
            res_prompt = '[!@#$NEXT!@#$]'.join(list_prompt)
            prompt = res_prompt

            # print(res_answer)
        else:
            if len(bboxes) == 0:
                prompt = (f"After explaining the overall photo from near to far, explain the path to the {goal_label}, which is the current destination, "
                        "explain the obstacles that exist on the path, and tell us what to do. ")
            else:
                if len(goal_cxcy) == 2:
                    prompt = (
                        "The image contains the following objects, which are located within bounding boxes represented by four numbers. "
                        f"These four numbers correspond to the normalized pixel values for left, top, right, and bottom. The included objects are {bbox_list_str}.\n"
                        f"After explaining the overall photo from near to far, explain the path to the {dest_descriptions}, which is the current destination and the two numbers represent the normalized horizontal and vertical axis values of the image.\n"
                        "Explain the obstacles that exist on the path, and tell us what to do. "
                    )

                    assert step_by_step == False
                elif len(goal_cxcy) == 4:
                    prompt = (
                        "The image contains the following objects, which are located within bounding boxes represented by four numbers. "
                        f"These four numbers correspond to the normalized pixel values for left, top, right, and bottom. The included objects are {bbox_list_str}.\n"
                        f"After explaining the overall photo from near to far, explain the path to the {dest_descriptions}, which is the current destination, "
                        "explain the obstacles that exist on the path, and tell us what to do. "
                    )
                
            if model_name == 'llava':                
                res_answer = run_llava_model(tokenizer=self.llava_tokenizer, model=self.llava_model, image_processor=self.llava_image_processor, context_len=self.llava_context_len, 
                                            input_query=prompt, image_files=image_path, input_conv_mode=None, input_temperature=input_temperature, input_top_p=input_top_p, input_num_beams=1, 
                                            input_max_new_tokens=512, model_name=self.llava_model_name)
            elif model_name == 'ferret':
                    pdb.set_trace()

        # print('query: ', prompt)
        # print('answer: ', res_answer)

        return prompt, res_answer