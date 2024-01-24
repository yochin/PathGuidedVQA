import openai
import base64
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import pdb


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

class LargeMultimodalModels():
    def __init__(self, model_name=None, llava_model_path=None):
        self.possible_models = ['dummy', 'llava', 'chatgpt']

        self.model_name = model_name

        assert self.model_name in self.possible_models

        self.OPENAI_API_KEY = "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"
        self.llava_model_path = llava_model_path


    def describe_images_with_boxes(self, image_path, bboxes, goal_label_cxcy, order, num_total, merge=False, previous_descriptions=[]):
        res_answer = ''

        if self.model_name == 'dummy':
            res_answer = self.describe_all_bboxes_with_dummy()
        elif self.model_name == 'llava':
            res_answer = self.describe_all_bboxes_with_llava(self.llava_model_path, image_path, bboxes, goal_label_cxcy, order, num_total, merge, previous_descriptions)
        elif self.model_name == 'chatgpt':
            res_answer = self.describe_all_bboxes_with_chatgpt(image_path, bboxes, goal_label_cxcy)
        else:
            raise AssertionError(f'{self.model_name} is not supported!')
        
        return res_answer


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



    def describe_all_bboxes_with_llava(self, llava_model_path, image_path, bboxes, goal_label_cxcy, order, num_total, merge=False, previous_descriptions=[]):

        # 각 바운딩 박스에 대한 설명 구성
        bbox_descriptions = [f"{label} at ({x_min}, {y_min}, {x_max}, {y_max})" for label, (x_min, y_min, x_max, y_max), _ in bboxes]
        bbox_list_str = ", ".join(bbox_descriptions)
        goal_label, goal_cxcy = goal_label_cxcy
        dest_descriptions = f"{goal_label} at ({goal_cxcy[0]}, {goal_cxcy[1]})"

        if order == 1:
            str_order = 'first'
        elif order == 2:
            str_order = 'second'
        elif order == 3:
            str_order = 'third'
        else:
            raise AssertionError('Not implemented')

        if num_total == 1:
            raise AssertionError('Not implemented')
        elif num_total == 2:
            str_num_total = 'two'
        elif num_total == 3:
            str_num_total = 'three'
        else:
            raise AssertionError('Not implemented')
        
        if merge:
            previous_prompt = []
            
            if len(previous_descriptions) == 3:
                previous_prompt.append('The first description is following: ')
                previous_prompt.append(previous_descriptions[0])
                                
                previous_prompt.append('The second description is following: ')
                previous_prompt.append(previous_descriptions[1])

                previous_prompt.append('The third description is following: ')
                previous_prompt.append(previous_descriptions[2])

                previous_prompt = ' '.join(previous_prompt)


            else:
                raise AssertionError('Not implemented')

            prompt = (  f"[Context: The input image is whole image of the route to the destination. "
                        "The input image depicts the view from a pedestrian's position, " 
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
            prompt = (  f"[Context: The input image is the {str_order} of {str_num_total} images of the route to the destination. "
                        "The input image depicts the view from a pedestrian's position, " 
                        "taken at a point 80cm above the ground for pedestrian navigation purposes. " 
                        "In this image, the user's starting point is situated below the center of the image at (0.5, 1.0). "
                        "Consider the starting point as the ground where the user is standing.]\n" 
                        f"[Obstacle Name at (bounding box): [{bbox_list_str}].]\n"
                        f"[Destination Name at (point): [{dest_descriptions}].]\n"
                        "Describe the obstacles to the destination in a natural and simple way "
                        "for a visually impaired person as a navigation assistant in 3 sentences. "
                        "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. ")
        

        llm_args = type('Args', (), {
            "model_path": llava_model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(llava_model_path),
            "query": prompt,
            "conv_mode": None,
            "image_file": image_path,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()

        res_answer = eval_model(llm_args)

        # print('query: ', prompt)
        # print('answer: ', res_answer)

        return res_answer