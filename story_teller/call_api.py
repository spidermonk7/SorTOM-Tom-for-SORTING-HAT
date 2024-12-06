import os
# 设置 OPENAI_API_KEY 环境变量
os.environ["OPENAI_API_KEY"] = "sk-ltmViK8cXdguTU9DB8QX7uLxpfkrQEQqgp7YPG6Ts3sWPGpF"

from openai import OpenAI
import openai
import httpx
from tqdm import tqdm
import datetime

# check api doc from
# https://flowus.cn/share/de98cb21-3c6d-4561-b6ac-648daa2bacda

possible_action_list = ['探索调查', '帮助NPC', '拒绝NPC', '背叛NPC', '战斗', '逃跑']

client = OpenAI(
    base_url='https://xiaoai.plus/v1',
    api_key=os.environ["OPENAI_API_KEY"]
)


def get_time_stamp():
    time_stamp = f'{datetime.datetime.now().day}, {datetime.datetime.now().hour}:{datetime.datetime.now().minute}:{datetime.datetime.now().second}'
    return time_stamp

log_file_path = 'log.txt'

def call_gpt(prompt: str):
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f'[{get_time_stamp()}]\n{prompt}\n')

    completion = client.chat.completions.create(
        model="gpt-4-turbo-2024-04-09",
        messages=[
        {"role": "system", "content": "你是一个辅助DND文字冒险游戏创作的AI助手"},
        {"role": "user", "content": prompt},
        ]
    )

    response = completion.choices[0].message.content
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f'[{get_time_stamp()}]\n{response}\n')
        

    return response


if __name__ == '__main__':
    response = call_gpt("hello")
    print(response)
    print('done')