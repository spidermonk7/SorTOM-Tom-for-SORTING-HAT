import os
# 设置 OPENAI_API_KEY 环境变量
os.environ["OPENAI_API_KEY"] = "sk-ltmViK8cXdguTU9DB8QX7uLxpfkrQEQqgp7YPG6Ts3sWPGpF"

from openai import OpenAI
import openai
import httpx
from tqdm import tqdm
import datetime
import numpy as np

# check api doc from
# https://flowus.cn/share/de98cb21-3c6d-4561-b6ac-648daa2bacda

possible_action_list = ['探索调查', '帮助NPC', '拒绝NPC', '背叛NPC', '战斗', '逃跑']

client = OpenAI(
    base_url='https://xiaoai.plus/v1',
    api_key=os.environ["OPENAI_API_KEY"]
)


def get_text_content(prompt: str):
    text_content = f'''
Translate the following sentences into english. Avoid fancy words and keep it simple.
{prompt}
'''
    return text_content


text_dir = '../dataset/story_tree'
en_text_dir = '../dataset/story_tree_en'
en_text_file_list = os.listdir(en_text_dir)

text_path_list = sorted(os.listdir(text_dir))
for text_path in tqdm(text_path_list, desc='translate text'):    
    if text_path in en_text_file_list:
        continue
    story_tree = np.load(os.path.join(text_dir, text_path), allow_pickle=True).item()
    story_tree_en = story_tree.copy()
    for k, v in story_tree.items():
        text_description = v['text_description']
        print(k, text_description)

        completion = client.chat.completions.create(
          model="gpt-4-turbo-2024-04-09",
          messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": get_text_content(text_description)},
          ]
        )

        response = completion.choices[0].message.content

        print(response)

        story_tree_en[k]['text_description'] = response
    np.save(os.path.join(en_text_dir, text_path), story_tree_en, allow_pickle=True)
print('done')
