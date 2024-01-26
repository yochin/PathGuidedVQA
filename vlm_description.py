import openai
import base64
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


OPENAI_API_KEY = "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"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def describe_all_bboxes_with_chatgpt(image_path, bboxes, goal_label_cxcy):
    # 이미지를 base64로 인코딩
    encoded_image = encode_image_to_base64(image_path)

    # 각 바운딩 박스에 대한 설명 구성
    bbox_descriptions = [f"{label} at ({x_min}, {y_min}, {x_max}, {y_max})" for label, (x_min, y_min, x_max, y_max), _ in bboxes]
    bbox_list_str = ", ".join(bbox_descriptions)
    goal_label, goal_cxcy = goal_label_cxcy
    dest_descriptions = f"{goal_label} at ({goal_cxcy[0]}, {goal_cxcy[1]})"

    # GPT-4에 대한 프롬프트 구성
    prompt = (  "[Context: The input image depicts the view from a pedestrian's position, " 
                "taken at a point 80cm above the ground for pedestrian navigation purposes. " 
                "In this image, the user's starting point is situated below the center of the image at (0.5, 1.0). "
                "Consider the starting point as the ground where the user is standing.]\n" 
                f"[Obstacle Name at (bounding box): [{bbox_list_str}].]\n"              
                f"[Destination Name at (point): [{dest_descriptions}].]\n"
                "Describe the obstacles to the destination in a natural and simple way "
                "for a visually impaired person as a navigation assistant in 3 sentences. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. "
                "First, answer it in English, then translate it Korean.")
   
    print("[PROMPT]: ", prompt)
    # OpenAI API 키 설정 (환경 변수에서 가져옴)
    openai.api_key = OPENAI_API_KEY
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

    print("[ANSWER]: ", answer)

    return answer



def describe_all_bboxes_with_llava(llava_model_base_path, llava_model_path, image_path, bboxes, goal_label_cxcy):

    # 각 바운딩 박스에 대한 설명 구성
    bbox_descriptions = [f"{label} at ({x_min}, {y_min}, {x_max}, {y_max})" for label, (x_min, y_min, x_max, y_max), _ in bboxes]
    bbox_list_str = ", ".join(bbox_descriptions)
    goal_label, goal_cxcy = goal_label_cxcy
    dest_descriptions = f"{goal_label} at ({goal_cxcy[0]}, {goal_cxcy[1]})"

    # GPT-4에 대한 프롬프트 구성
    prompt = (  "[Context: The input image depicts the view from a pedestrian's position, " 
                "taken at a point 80cm above the ground for pedestrian navigation purposes. " 
                "In this image, the user's starting point is situated below the center of the image at (0.5, 1.0). "
                "Consider the starting point as the ground where the user is standing.]\n" 
                f"[Obstacle Name at (bounding box): [{bbox_list_str}].]\n"              
                f"[Destination Name at (point): [{dest_descriptions}].]\n"
                "Describe the obstacles to the destination in a natural and simple way "
                "for a visually impaired person as a navigation assistant in 3 sentences. "
                "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. "
                "First, answer it in English, then translate it Korean.")

    print("[PROMPT] ====================================================")
    print(prompt)
    print("=============================================================")
    args = type('Args', (), {
        "model_path": llava_model_path,
        "model_base": llava_model_base_path,
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
    
    answer = eval_model(args)

    print("[ANSWER] ====================================================")
    print(answer)
    print("=============================================================")

    return eval_model(args)