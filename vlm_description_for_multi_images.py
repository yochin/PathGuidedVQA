import openai
import base64
from llava.eval.run_llava import init_llava16_model, run_llava16_model
from llava.serve.cli import init_llava16_model_cli, run_llava16_model_cli
from llava.conversation import conv_templates
from transformers import set_seed
# from prompt_library import get_prompt
# from prompt_library_by_hbo import get_prompt_by_hbo
import sys
import importlib
import pdb


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

class LargeMultimodalModels():
    def __init__(self, model_name=None, llava_model_base_path=None, llava_model_path=None, ferret_model_path=None, prompt_lib_name=None):
        self.possible_models = ['dummy', 'llava', 'chatgpt', 'ferret', 'llava16', 'llava16_cli']

        self.model_name = model_name

        assert self.model_name in self.possible_models
        assert prompt_lib_name is not None
        
        if 'prompt_lib' not in sys.path:
            sys.path.append('prompt_lib')

        self.prompt_library = importlib.import_module(prompt_lib_name)

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
                self.llava_context_len, self.llava_model_name, self.llava_input_conv_mode = init_llava16_model_cli(model_path=self.llava_model_path, model_base=self.llava_model_base_path, 
                                                    input_conv_mode=None)
                print('@LargeMultimodalModels - init_llava16_model: ', self.llava_model_name)
            else:
                raise AssertionError('check the model: ', self.model_name)
            
        # for ferret
        elif self.model_name == 'ferret':
            self.ferret_model_path = ferret_model_path
        

    def describe_whole_images_with_boxes(self, image_path, bboxes, goal_label_cxcy, step_by_step=False, list_example_prompt=[], 
                                         prompt_id=18, prefix_prompt=None):
        print('\n')
        print('@describe_whole_images_with_boxes - image_path:', image_path)
        print('@describe_whole_images_with_boxes - bboxes:', bboxes)
        print('@describe_whole_images_with_boxes - goal_label_cxcy:', goal_label_cxcy)
        print('@describe_whole_images_with_boxes - prompt_id: ', prompt_id)

        if len(list_example_prompt) > 0:
            assert self.model_name == 'llava16_cli'

        res_answer = ''
        res_query = ''

        if self.model_name == 'dummy':
            res_answer = self.describe_all_bboxes_with_dummy()
        elif self.model_name in ['llava', 'ferret', 'llava16', 'llava16_cli', 'chatgpt']:
            res_query, res_answer = self.describe_all_bboxes_with_llava(image_path, bboxes, goal_label_cxcy, step_by_step, self.model_name, 
                                                                        list_example_prompt, prompt_id, prefix_prompt)
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

    
    def describe_all_bboxes_with_llava(self, image_path, bboxes, goal_label_cxcy, step_by_step=False, model_name=None, 
                                       list_example_prompt=[], prompt_id=18, prefix_prompt=None):
        set_seed(42)
        input_temperature = 0.6
        input_top_p = 0.9

        if step_by_step:
            # if prompt_id in [91118, 91148]:
            #     list_prompt, list_system = get_prompt_by_hbo(goal_label_cxcy, bboxes, trial_num=prompt_id, sep_system=True)
            # else:
            #     list_prompt, list_system = get_prompt(goal_label_cxcy, bboxes, trial_num=prompt_id, sep_system=True)
            list_prompt, list_system = self.prompt_library.get_prompt(goal_label_cxcy, bboxes, trial_num=prompt_id, sep_system=True)

            if prefix_prompt is not None:
                list_prompt = [' '.join(prefix_prompt + list_prompt)]

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
                
                list_answer = []
                messages = [
                    {"role": "system", "content": list_system[0]}
                ]

                for query in list_prompt:
                    print(f'@gpt - automatic user input: {query}')

                    user_content = [
                        {
                            "type": "text", 
                            "text": query
                        }
                    ]
                    if encoded_image is not None:
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}",
                                    "detail": "high"    # "low", "high", "auto"
                                }
                            } #, "mimetype": "image/jpeg"
                        )
                        encoded_image = None

                    messages.append(
                        {
                            "role": "user",
                            "content": user_content
                        }
                    )

                    completion = openai.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=messages,
                        max_tokens=1024,
                    )
                    answer = completion.choices[0].message.content

                    messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                    list_answer.append(answer)

                    # print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
            
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


import os
import pdb

if __name__ == '__main__':
    llava_model_base_path = None
    llava_model_path = '../llm_models/llava/llava-v1.6-34b'
    vlm_model_name = 'llava16_cli'

    lvm = LargeMultimodalModels(vlm_model_name, llava_model_base_path=llava_model_base_path, llava_model_path=llava_model_path)

    
    ppt_id = 45101

    prefix_prompt = None

    output_path_debug = '../output/HBO_B91118_MASK_10mPtr_CamInt_LLMDec_debug/debug'

    filename = ''
    ppt_c = 'D'
    ext = '.jpg'
    ppt_id_list_img_path = [os.path.join(output_path_debug, f'{filename}_{ppt_c}{ext}')]
    list_prompt = ['Explain the destination, represented by a orange point [0.4343, 0.4143] in 1 line. Example 1) The destination is an entrance of a building. Example 2) The destination is in the middle of the pedestrian road. ']
    
    ppt_bboxes = []

    ppt_id_list_img_path = [os.path.join(output_path_debug, f'MP_SEL_075205_D.jpg')]
    list_answer = run_llava16_model_cli(lvm.llava_tokenizer, lvm.llava_model, lvm.llava_image_processor, lvm.llava_context_len, lvm.llava_model_name, image_files=ppt_id_list_img_path, list_queries=list_prompt, input_conv_mode=lvm.llava_input_conv_mode, input_temperature=0.6, input_top_p=0.9, input_num_beams=1, input_max_new_tokens=512, input_debug=True, use_ex_image=False)
    

    
    pdb.set_trace()