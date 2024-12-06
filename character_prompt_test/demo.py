import gymnasium as gym
from chat_demo import generate_answer
from .utils import *
import json
import os
import random

game_list = {
    "taxi": "Taxi-v3",
    "lake": "FrozenLake-v1",
    "jack": "Blackjack-v1",
}

def run(game, character, debug=False, data_num=3):
    # 初始化文件路径
    file_path = f"data/{game}_{character}.json"
    # 如果文件已存在，加载已有数据
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
    else:
        data = {}
    
    while len(data) < data_num:
        num = len(data)
        print(f"The length of data is {len(data)}")
        env = gym.make(game_list[game])

        obs = env.reset()[0]
        action = generate_answer(str(obs), character, game)
        data[str(num)] = {"state": str(obs), "action": action}
        
        # **立即保存数据**
        with open(file_path, "w") as file:
            json.dump(data, file)
        
        terminated = False
        truncated = False

        while not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(parse_answer(action))
            if debug:
                print("新状态:", obs, "奖励:", reward, "动作:", action)
            # **立即保存数据**
            with open(file_path, "w") as file:
                json.dump(data, file)
            # 为下一轮生成新动作
            action = generate_answer(str(obs), character, game)
            data[str(num)] = {"state": str(obs), "action": action}



def run_v2(game, character, debug=False, data_num=3):
    # 初始化文件路径
    file_path = f"data/{game}_{character}_v2.json"
    # 如果文件已存在，加载已有数据
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
    else:
        data = {}
 
    # all combinations of self_sum and dealer_sum
    for self_sum in tqdm(range(3, 14), desc=f"Handling {character}"):
        for dealer_sum in range(1, 6):
            num = len(data)
            print(f"The length of data is {len(data)}")
            obs = (self_sum, dealer_sum)
            action = generate_answer(str(obs), character, game)
            data[str(num)] = {"state": str(obs), "action": action}
    
        # **立即保存数据**
        with open(file_path, "w") as file:
            json.dump(data, file)
        
    
def run_v3(game, character, debug=False, data_num=3):
    # 初始化文件路径
    file_path = f"data/{game}_{character}_v3.json"
    # 如果文件已存在，加载已有数据
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
    else:
        data = {}
 
    # all combinations of self_sum and dealer_sum
    for self_sum in tqdm(range(3, 14), desc=f"Handling {character}"):
        for dealer_sum in range(1, 6):
            num = len(data)
            print(f"The length of data is {len(data)}")
            obs = (self_sum, dealer_sum)
            action = generate_answer(str(obs), character, game)
            data[str(num)] = {"state": str(obs), "action": action}
    
        # **立即保存数据**
        with open(file_path, "w") as file:
            json.dump(data, file)
     
    
if __name__ == '__main__':
    for character in ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]:
        for game in ["jack"]:
            run_v3(game, character, debug=False, data_num=200)
            print("Finished", game, character)
            print('-'*50)
