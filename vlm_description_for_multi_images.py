import openai
import base64
from llava.eval.run_llava import init_llava16_model, run_llava16_model
from llava.serve.cli import run_llava16_model_cli
from llava.conversation import conv_templates
from transformers import set_seed
from prompt_library import get_prompt
import pdb


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

class LargeMultimodalModels():
    def __init__(self, model_name=None, llava_model_base_path=None, llava_model_path=None, ferret_model_path=None, prompt_id=18):
        self.possible_models = ['dummy', 'llava', 'chatgpt', 'ferret', 'llava16', 'llava16_cli']

        self.model_name = model_name
        self.prompt_id = prompt_id
        print('@LargeMultimodalModels - prompt_id: ', prompt_id)

        assert self.model_name in self.possible_models

        # for gpt4
        if self.model_name == 'chatgpt':
            self.OPENAI_API_KEY = "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"
        # for llava
        elif self.model_name in ['llava', 'llava16', 'llava16_cli']:
            self.llava_model_path = llava_model_path
            self.llava_model_base_path = llava_model_base_path

            if self.model_name == 'llava16':
                self.llava_tokenizer, self.llava_model, self.llava_image_processor, \
                self.llava_context_len, self.llava_model_name, self.input_conv_mode = init_llava16_model(model_path=self.llava_model_path,
                                                                                model_base=self.llava_model_base_path,
                                                                                )
            elif self.model_name == 'llava16_cli':
                self.llava_tokenizer, self.llava_model, self.llava_image_processor, \
                self.llava_context_len, self.llava_model_name, self.llava_input_conv_mode = init_llava16_model(model_path=self.llava_model_path, model_base=self.llava_model_base_path, 
                                                    input_conv_mode=None)
                print('@LargeMultimodalModels - init_llava16_model: ', self.llava_model_name)
            else:
                raise AssertionError('check the model: ', self.model_name)
            
        # for ferret
        elif self.model_name == 'ferret':
            self.ferret_model_path = ferret_model_path
        

    def describe_whole_images_with_boxes(self, image_path, bboxes, goal_label_cxcy, step_by_step=False, list_example_prompt=[]):
        print('\n')
        print('@describe_whole_images_with_boxes - image_path:', image_path)
        print('@describe_whole_images_with_boxes - bboxes:', bboxes)
        print('@describe_whole_images_with_boxes - goal_label_cxcy:', goal_label_cxcy)

        if len(list_example_prompt) > 0:
            assert self.model_name == 'llava16_cli'

        res_answer = ''
        res_query = ''

        if self.model_name == 'dummy':
            res_answer = self.describe_all_bboxes_with_dummy()
        elif self.model_name in ['llava', 'ferret', 'llava16', 'llava16_cli', 'chatgpt']:
            res_query, res_answer = self.describe_all_bboxes_with_llava(image_path, bboxes, goal_label_cxcy, step_by_step, self.model_name, list_example_prompt)
        else:
            raise AssertionError(f'{self.model_name} is not supported!')
        
        return res_query, res_answer


    def describe_all_bboxes_with_dummy(self):
        answer = 'hello world'

        return answer
    

    # def describe_all_bboxes_with_chatgpt(self, image_path, bboxes, goal_label_cxcy):
    #     # 이미지를 base64로 인코딩
    #     encoded_image = encode_image_to_base64(image_path)

    #     # 각 바운딩 박스에 대한 설명 구성
    #     bbox_descriptions = [f"{label} at ({x_min}, {y_min}, {x_max}, {y_max})" for label, (x_min, y_min, x_max, y_max), _ in bboxes]
    #     bbox_list_str = ", ".join(bbox_descriptions)
    #     goal_label, goal_cxcy = goal_label_cxcy
    #     dest_descriptions = f"{goal_label} at ({goal_cxcy[0]}, {goal_cxcy[1]})"

    #     # 프롬프트 구성
    #     prompt = (  "[Context: The input image depicts the view from a pedestrian's position, " 
    #                 "taken at a point 80cm above the ground for pedestrian navigation purposes. " 
    #                 "In this image, the user's starting point is situated below the center of the image at (0.5, 1.0). "
    #                 "Consider the starting point as the ground where the user is standing.]\n" 
    #                 f"[Obstacle Name at (bounding box): [{bbox_list_str}].]\n"
    #                 f"[Destination Name at (point): [{dest_descriptions}].]\n"
    #                 "Describe the obstacles to the destination in a natural and simple way "
    #                 "for a visually impaired person as a navigation assistant in 3 sentences. "
    #                 "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. ")

    
    #     # print("[PROMPT]: ", prompt)
    #     # OpenAI API 키 설정 (환경 변수에서 가져옴)
    #     openai.api_key = self.OPENAI_API_KEY
    #     completion = openai.chat.completions.create(
    #         #model = "gpt-4",
    #         model="gpt-4-1106-preview",
    #         #messages=[
    #         #    {
    #         #        "role": "user",
    #         #        "content": prompt,
    #         #    },
    #         messages=[
    #             {"role": "system", "content": "This is an image-based task."},
    #             {"role": "user", "content": encoded_image}, #, "mimetype": "image/jpeg"
    #             {"role": "user", "content": prompt},
    #         ],
    #         #max_tokens=1000,
    #     )

    #     answer = completion.choices[0].message.content

    #     # print("[ANSWER]: ", answer)

    #     return answer

    
    def describe_all_bboxes_with_llava(self, image_path, bboxes, goal_label_cxcy, step_by_step=False, model_name=None, list_example_prompt=[]):
        set_seed(42)
        input_temperature = 0.6
        input_top_p = 0.9

        if step_by_step:
            list_prompt, list_system = get_prompt(goal_label_cxcy, bboxes, trial_num=self.prompt_id, sep_system=True)

            if model_name == 'llava16_cli':
                list_answer = run_llava16_model_cli(self.llava_tokenizer, self.llava_model, self.llava_image_processor, self.llava_context_len, self.llava_model_name, 
                                                    image_files=image_path, list_queries=list_prompt, input_conv_mode=self.llava_input_conv_mode,
                                                    input_temperature=input_temperature, input_top_p=input_top_p, input_num_beams=1,
                                                    input_max_new_tokens=512, input_debug=True, list_ex_prompt=list_example_prompt, list_system=list_system,
                                                    use_ex_image=False)
            
            elif model_name == 'chatgpt':
                # 이미지를 base64로 인코딩
                encoded_image = encode_image_to_base64(image_path[0])
                # OpenAI API 키 설정 (환경 변수에서 가져옴)
                openai.api_key = self.OPENAI_API_KEY

                
                completion = openai.chat.completions.create(
                    #model = "gpt-4",
                    #messages=[
                    #    {
                    #        "role": "user",
                    #        "content": prompt,
                    #    },
                    # model="gpt-4-1106-preview",
                    # messages=[
                    #     {"role": "system", "content": list_system[0]},
                    #     {"role": "user", "content": encoded_image}, #, "mimetype": "image/jpeg"
                    #     {"role": "user", "content": list_prompt[0]},
                    # ],
                    model="gpt-4-vision-preview",
                    messages=[
                        {"role": "system", "content": list_system[0]},
                        {"role": "user", 
                         "content": [
                             {
                                "type": "text", 
                                "text": list_prompt[0]
                             },
                             {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}",
                                    "detail": "high"    # "low", "high", "auto"
                                }
                             } #, "mimetype": "image/jpeg"
                         ]
                        }
                    ],
                    max_tokens=1024,
                )
                answer = completion.choices[0].message.content
                list_answer = [answer]
                        
            else:
                list_answer = []
                for i_prompt, prompt in enumerate(list_prompt):
                    if i_prompt == 0:
                        in_prompt = prompt
                    else:
                        conv = conv_templates[self.input_conv_mode].copy()       # reset the conversation by copying from templates

                        in_prompt = ' '.join([conv.roles[0], '\n', list_prompt[i_prompt-1], '\n',
                                            conv.roles[1], '\n', list_answer[i_prompt-1], '\n', 
                                            conv.roles[0], '\n', prompt])

                    if model_name == 'llava16':
                        answer = run_llava16_model(tokenizer=self.llava_tokenizer, model=self.llava_model, image_processor=self.llava_image_processor, context_len=self.llava_context_len, 
                                                input_query=in_prompt, image_files=image_path, input_conv_mode=self.input_conv_mode, input_temperature=input_temperature, input_top_p=input_top_p, input_num_beams=1, 
                                                input_max_new_tokens=512, model_name=self.llava_model_name)
                    else:
                        raise AssertionError('check the model: ', model_name)

                    list_answer.append(answer)

            res_answer = '[!@#$NEXT!@#$]'.join(list_answer)
            res_prompt = '[!@#$NEXT!@#$]'.join(list_prompt)
            prompt = res_prompt

            # print(res_answer)
        else:
            raise AssertionError('No more supported')
            # if len(bboxes) == 0:
            #     prompt = (f"After explaining the overall photo from near to far, explain the path to the {goal_label}, which is the current destination, "
            #             "explain the obstacles that exist on the path, and tell us what to do. ")
            # else:
            #     if len(goal_cxcy) == 2:
            #         prompt = (
            #             "The image contains the following objects, which are located within bounding boxes represented by four numbers. "
            #             f"These four numbers correspond to the normalized pixel values for left, top, right, and bottom. The included objects are {bbox_list_str}.\n"
            #             f"After explaining the overall photo from near to far, explain the path to the {dest_descriptions}, which is the current destination and the two numbers represent the normalized horizontal and vertical axis values of the image.\n"
            #             "Explain the obstacles that exist on the path, and tell us what to do. "
            #         )

            #         assert step_by_step == False
            #     elif len(goal_cxcy) == 4:
            #         prompt = (
            #             "The image contains the following objects, which are located within bounding boxes represented by four numbers. "
            #             f"These four numbers correspond to the normalized pixel values for left, top, right, and bottom. The included objects are {bbox_list_str}.\n"
            #             f"After explaining the overall photo from near to far, explain the path to the {dest_descriptions}, which is the current destination, "
            #             "explain the obstacles that exist on the path, and tell us what to do. "
            #         )
                
            # if model_name == 'llava':
            #     res_answer = run_llava_model(tokenizer=self.llava_tokenizer, model=self.llava_model, image_processor=self.llava_image_processor, context_len=self.llava_context_len, 
            #                                 input_query=prompt, image_files=image_path, input_conv_mode=None, input_temperature=input_temperature, input_top_p=input_top_p, input_num_beams=1, 
            #                                 input_max_new_tokens=512, model_name=self.llava_model_name)
            # elif model_name == 'ferret':
            #         pdb.set_trace()

        # print('query: ', prompt)
        # print('answer: ', res_answer)

        return prompt, res_answer