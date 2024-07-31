import json
import base64
import io
import copy

import json
from PIL import Image
from openai import OpenAI

api_key = None
assert api_key is not None, "Please set your OpenAI API key in the api_key variable"

client = OpenAI(
    api_key=api_key,
)
system_prompt = None


def image_preprocess(image_path):
    image = Image.open(image_path)
    max_edge_length = 512
    width,height=image.size
    if max(width,height) > max_edge_length:
        scalar = max(width,height)/max_edge_length
        new_size = (int(width/scalar),int(height/scalar))
    else:
        new_size = (width,height)
    resized_image = image.resize(new_size)
    
    buffer = io.BytesIO()
    resized_image.save(buffer, format="JPEG")
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image


def load_json_from_output(output):
    start_index = output.find('{')
    end_index = output.rfind('}')
    if start_index == -1 or end_index == -1 or start_index >= end_index:
        print(output)
        return "Invalid input string"
    return json.loads(output[start_index:end_index + 1])


def construct_interleaved_input(text_list, image_list, image_ext="jpg"):
    images = []
    for path in image_list:
        images.append(image_preprocess(path))
    text_template = {"type": "text", "text": None}
    image_template = {"type": "image_url", 
                        "image_url": {"url": None}
                        }
    image_url_template = f"data:image/{image_ext};base64,"
    content = []
    #start from text
    cur_state = "text"
    text_index = 0
    image_index = 0
    for i in range(len(text_list)+len(images)):
        if cur_state == "text":
            temp = copy.deepcopy(text_template)
            temp["text"] = f"<Description_{text_index}> " + text_list[text_index] + f" </Description_{text_index}> "
            content.append(temp)
            text_index += 1
        else:
            content.append({"type": "text", "text": f"<Img_{image_index}>"})
            temp = copy.deepcopy(image_template)
            temp["image_url"]["url"] = image_url_template + images[image_index]
            content.append(temp)
            content.append({"type": "text", "text": f"</Img_{image_index}>"})
            image_index += 1
        if cur_state == "text" and image_index < len(images):
            cur_state = "image"
        else:
            cur_state = "text"

    return content


def modify_system_prompt(file_path):
    try:
        global system_prompt
        with open(file_path, 'r') as f:
            content = f.read()
        
        system_prompt = content
        return system_prompt
    
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def openai_api(content):
    global system_prompt
    is_stream = True
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content":system_prompt },
            {"role": "user", "content": content}
        ],
        stream=is_stream,
        temperature= 1.0,
    )
    if is_stream:
        out = ""
        for chunk in response:
            if chunk.choices:
                tmp = chunk.choices[0].delta.content or ""
            else:
                tmp = ""
            out+=tmp
            # print(tmp , end="")
        return out
    else:
        output = response.choices[0]
        return output


def get_image_path_list(step_info):
    # image_preprocess(path)
    image_path_list = []
    for tot_step in step_info:
        for step in tot_step:
            if step["type"] == "image":
                image_path = step["content"]
                image_path_list.append(image_path)
    return image_path_list


def construct_content(image_index, step_info):
    content = []
    for tot_step in step_info:
        for step in tot_step:
            if step["type"] == "text":
                content.append({"type": "text", "text": step["content"]})
            elif step["type"] == "image":
                content.append({"type": "text", "text": f" <Img_{image_index}>"})
                image_path = step["content"]
                base64_image = image_preprocess(image_path)
                content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                content.append({"type": "text", "text": f"</Img_{image_index}>"})
                image_index += 1
    return content, image_index


def remove_images_from_content(content):
    text_content = []
    for item in content:
        if item["type"] == "text":
            text_content.append(item)
    return text_content


def combine_input_output(input, output):
    input_token = [{"type": "text", "text": "INPUT:"}]
    output_token = [{"type": "text", "text": "OUTPUT:"}]
    content = input_token + input + output_token + output
    return content


def input_to_json(input_text):
    input_token = [[{"type": "text", "content": input_text}]]
    return input_token


def get_text_list(step_info):
    # image_preprocess(path)
    text_list = []
    for tot_step in step_info:
        temp_text = ""
        for step in tot_step:
            if step["type"] == "text":
                temp_text = temp_text + " " + step["content"]
        text_list.append(temp_text)
    return text_list


def construct_task1_content(image_sequence_step_info, test_sequence_step_info):
    text_list = get_text_list(test_sequence_step_info)
    content = []
    cur_text_index = 0
    for group in image_sequence_step_info:
        # add cur text description
        if cur_text_index >= len(text_list):
            break
        temp = {"type": "text", "text": text_list[cur_text_index]}
        cur_text_index += 1
        content.append(temp)
        for item in group:
            # add all the images
            if item["type"] == "image":
                base64_image = image_preprocess(item["content"])
                content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
    return content


def add_image_token(content):
    i = 0
    add_token_num = 0
    original_len = len(content)
    while i != add_token_num + original_len:
        cur = content[i]
        if cur["type"] == "text":
            i += 1
        else:
            # add tokens
            content.insert(i, {"type": "text", "text": f"<Img_{add_token_num//2}>"})
            i += 2
            content.insert(i, {"type": "text", "text": f"</Img_{add_token_num//2}>"})
            add_token_num += 2
            i += 1
    return content


def gpt4o_eval_task1(input_step_info, output_step_info):
    content = construct_task1_content(input_step_info, output_step_info)
    modify_system_prompt("./tot_metric/machine_evaluation_prompts/task_1.json")
    output = openai_api(content)
    completeness_and_text_fidelity = load_json_from_output(output)
    return completeness_and_text_fidelity


def gpt4o_eval_task2(input_step_info, output_step_info):
    content = construct_task1_content(output_step_info, input_step_info)
    content = add_image_token(content)

    modify_system_prompt("tot_metric/machine_evaluation_prompts/document_completeness_image_coherence_task2.json")
    output = openai_api(content)
    coherence_and_completeness = load_json_from_output(output)

    modify_system_prompt("./tot_metric/machine_evaluation_prompts/texts_to_image_descriptions_task2.json")
    model_text_output = remove_images_from_content(content)
    output = openai_api(content)
    texts_to_image_descriptions = load_json_from_output(output)

    modify_system_prompt("./tot_metric/machine_evaluation_prompts/descriptions_gt_images_consistency.json")
    descriptions = []

    for i in range(len(texts_to_image_descriptions["Tasks"]["Create an Interleaved Text-Image Document"])):
        descriptions.append(texts_to_image_descriptions["Tasks"]["Create an Interleaved Text-Image Document"][f"Content of Image {i}"])
    image_path = get_image_path_list(output_step_info)
    content = construct_interleaved_input(descriptions, image_path)
    output = openai_api(content)
    descriptions_gt_images_consistency = load_json_from_output(output)

    evaluation_results = {}
    evaluation_results["coherence_score"] = coherence_and_completeness["scores"]["Image_Coherence"]["overall_score"]
    # evaluation_results["completeness"] = {"Score": coherence_and_completeness["scores"]["Completeness"]["Score"], "Justification": coherence_and_completeness["scores"]["Completeness"]["Justification"]}
    evaluation_results["text fidelity to images"] = {"Score": descriptions_gt_images_consistency["Consistency"]["overall_score"], "Justification": descriptions_gt_images_consistency["Consistency"]["Justification"]}
    evaluation_results["image quality"] = {"Score": coherence_and_completeness["scores"]["Image_Quality"]["Score"], "Justification": coherence_and_completeness["scores"]["Image_Quality"]["Score"]}

    return evaluation_results, coherence_and_completeness, texts_to_image_descriptions, descriptions_gt_images_consistency


def gpt4o_eval_task4(input_text, step_info):
    if type(input_text) == str:
        model_input = input_to_json(input_text)
    else:
        model_input = input_text
    model_input, output_image_index = construct_content(0, model_input)
    model_output, final_image_number = construct_content(output_image_index, step_info)

    modify_system_prompt("./tot_metric/machine_evaluation_prompts/document_completeness_image_coherence.json")
    content = combine_input_output(model_input, model_output)
    output = openai_api(content)
    coherence_and_completeness = load_json_from_output(output)

    modify_system_prompt("tot_metric/machine_evaluation_prompts/texts_to_image_descriptions.json")
    model_text_output = remove_images_from_content(model_output)
    content = combine_input_output(model_input, model_text_output)
    output = openai_api(content)
    texts_to_image_descriptions = load_json_from_output(output)

    modify_system_prompt("tot_metric/machine_evaluation_prompts/descriptions_gt_images_consistency.json")
    descriptions = []
    for i in range(output_image_index, output_image_index+len(texts_to_image_descriptions["Tasks"]["Create an Interleaved Text-Image Document"])):
        descriptions.append("Description:" + texts_to_image_descriptions["Tasks"]["Create an Interleaved Text-Image Document"][f"Content of Image {i}"])
    image_path = get_image_path_list(step_info)
    content = construct_interleaved_input(descriptions, image_path)
    output = openai_api(content)
    descriptions_gt_images_consistency = load_json_from_output(output)

    evaluation_results = {}
    evaluation_results["coherence_score"] = coherence_and_completeness["scores"]["Image_Coherence"]["overall_score"]
    evaluation_results["completeness"] = {"Score": coherence_and_completeness["scores"]["Completeness"]["Score"], "Justification": coherence_and_completeness["scores"]["Completeness"]["Justification"]}
    evaluation_results["Image_Quality"] = {"Score": coherence_and_completeness["scores"]["Image_Quality"]["Score"], "Justification": coherence_and_completeness["scores"]["Image_Quality"]["Justification"]}
    evaluation_results["text fidelity to images"] = {"Score": descriptions_gt_images_consistency["Consistency"]["overall_score"], "Justification": descriptions_gt_images_consistency["Consistency"]["Justification"]}

    return evaluation_results, coherence_and_completeness, texts_to_image_descriptions, descriptions_gt_images_consistency