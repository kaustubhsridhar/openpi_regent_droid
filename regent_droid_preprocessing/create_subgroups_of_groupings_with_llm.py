import os 
import json
import argparse
from google import genai
from openai import OpenAI

prompt_1 = 'Take this dict of episode numbers to task descriptions and divide it into groups where the tasks are all similar or the same. In this case, "away from you" is the same as "backward" and "towards you" is the same as "forward".'

prompt_2 = 'Make sure that the output is a dict of dicts like {"subgroup_name": {"int": "task description", ...}, ...} and only output this dict of dicts and nothing else. Use the double quotes and not single quotes around the keys and values. Leave out any subgroups that do not have a strong similarity between its task descriptions like miscellaneous tasks, other tasks, and so on.'

def create_subgroups_of_groupings_with_llm(json_folder, llm_name, new_folder):
    all_json_files = [file for file in os.listdir(json_folder) if file.endswith('.json')]
    all_dicts = {file: json.load(open(f"{json_folder}/{file}")) for file in all_json_files}
    
    if "gemini" in llm_name:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        for file, loaded_dict in all_dicts.items():
            print(f'processing {file}...')
            input = prompt_1 + f'{loaded_dict}' + prompt_2
            # print(f'input: {input}')
            response = client.models.generate_content(
                        model=llm_name,
                        contents=input,
                    )
            response_text = response.text
            # print(f'response_text: {response_text}')

            # remove ```python and ``` if they exist
            if "```python" in response_text:
                response_text = response_text.replace("```python", "")
            if "```" in response_text:
                response_text = response_text.replace("```", "")

            with open(f"{new_folder}/{file}", "w") as f:
                f.write(response_text)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_folder", type=str, default="droid_groups/droid_new_N10_M5_minnumep50_scene_id_withlang_below95k")
    parser.add_argument("--llm_name", type=str, default="gemini-2.0-pro-exp-02-05")
    args = parser.parse_args()

    new_folder = f"{args.json_folder}_subgroups"
    os.makedirs(new_folder, exist_ok=True)
    create_subgroups_of_groupings_with_llm(args.json_folder, args.llm_name, new_folder)
