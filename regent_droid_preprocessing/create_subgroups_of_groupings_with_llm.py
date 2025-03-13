import os 
import json
import argparse
from google import genai
from openai import OpenAI
from datetime import datetime
from collections import defaultdict
prompt_1 = 'Take this dict of episode numbers to task descriptions and divide it into groups where the tasks are all similar or the same. In this case, "away from you" is the same as "backward" and "towards you" is the same as "forward".'

prompt_2 = 'Make sure that the output is a dict of dicts like {"subgroup_name": {"int": "task description", ...}, ...} and only output this dict of dicts and nothing else. Use the double quotes and not single quotes around the keys and values. Leave out any subgroups that do not have a strong similarity between its task descriptions like miscellaneous tasks, other tasks, and so on.'

def create_subgroups_of_groupings_with_llm(json_folder, llm_name, new_folder, count_subgroups_with_this_min_num_episodes):
    all_json_files = [file for file in os.listdir(json_folder) if file.endswith('.json')]
    all_dicts = {file: json.load(open(f"{json_folder}/{file}")) for file in all_json_files}
    
    if "gemini" in llm_name:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        for file, loaded_dict in all_dicts.items():
            if os.path.exists(f"{new_folder}/{file}"):
                continue

            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: processing {file}...')
            input = prompt_1 + f'{loaded_dict}' + prompt_2
            # print(f'input: {input}')
            response = client.models.generate_content(
                        model=llm_name,
                        contents=input,
                    )
            response_text = response.text
            # print(f'response_text: {response_text}')

            # remove ```python and ``` and json if they exist
            if "```python" in response_text:
                response_text = response_text.replace("```python", "")
            if "```json" in response_text:
                response_text = response_text.replace("json", "")
            if "```" in response_text:
                response_text = response_text.replace("```", "")

            with open(f"{new_folder}/{file}", "w") as f:
                f.write(response_text)
    
    elif "gpt" in llm_name:
        pass

    # iterate over the saved jsons, verify if it can be loaded
    superdict = defaultdict(dict)
    for file in os.listdir(new_folder):
        if file.endswith(".json"):
            # print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: processing {new_folder}/{file}...')
            new_dict = json.load(open(f"{new_folder}/{file}"))

            # next, see if there are any hallucinations by checking that for all subgroups, the keys and values are in the original dict
            hallucination = False
            for subgroup_name, subgroup_dict in new_dict.items():
                for key, value in subgroup_dict.items():
                    if key not in all_dicts[file]:
                        print(f'{key} in {new_folder}/{file} not found in {json_folder}/{file}')
                    
                    if value != all_dicts[file][key]:
                        hallucination = True
                        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: hallucination found in {new_folder}/{file} where we have {value} but the original has {all_dicts[file][key]}')
                        # overwrite the value in the new dict and resave it
                        new_dict[subgroup_name][key] = all_dicts[file][key]
            if hallucination:
                with open(f"{new_folder}/{file}", "w") as f:
                    json.dump(new_dict, f, indent=4)

            # count the number of subgroups with atleast count_subgroups_with_this_min_num_episodes episodes | also save to separate dict
            for subgroup_name, subgroup_dict in new_dict.items():
                if len(subgroup_dict) >= count_subgroups_with_this_min_num_episodes:
                    superdict[f'{file.replace(".json", "")}_{subgroup_name}'] = subgroup_dict

    # save the superdict to a json file
    with open(f"droid_groups/droid_new_superdict_of_subgroups_with_atleast_{count_subgroups_with_this_min_num_episodes}_episodes.json", "w") as f:
        json.dump(superdict, f, indent=4)
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: {len(superdict)} subgroups with atleast {count_subgroups_with_this_min_num_episodes} episodes found')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_folder", type=str, default="droid_groups/droid_new_N10_M5_minnumep50_scene_id_withlang_below95k")
    parser.add_argument("--llm_name", type=str, default="gemini-2.0-pro-exp-02-05")
    parser.add_argument("--count_subgroups_with_this_min_num_episodes", type=int, default=20)
    args = parser.parse_args()

    new_folder = f"{args.json_folder}_subgroups"
    os.makedirs(new_folder, exist_ok=True)
    create_subgroups_of_groupings_with_llm(args.json_folder, args.llm_name, new_folder, args.count_subgroups_with_this_min_num_episodes)
