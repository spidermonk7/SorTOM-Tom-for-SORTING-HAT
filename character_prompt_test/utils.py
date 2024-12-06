import numpy
import os
import json
from tqdm import tqdm
import torch


colleges_map = {
    "Gryffindor": 0,
    "Slytherin": 1,
    "Hufflepuff": 2,
    "Ravenclaw": 3
}

action_dict = {"Explore": 0, "Help": 0, "Refuse": 0, "Betray": 0, "Fight": 0, "Escape": 0}
CNmap = {"探索调查": "Explore", "帮助NPC": "Help", "拒绝NPC": "Refuse", "背叛NPC": "Betray", "战斗": "Fight", "逃跑": "Escape"}



# The function to parse the answer to action
def parse_answer(answer, game='dnd'):
    answer = answer.rstrip()
    if game == 'dnd':
        try:
            return int(answer[-2])
        except:
            raise ValueError(f"The answer {answer} is not valid")
    
    elif game == 'jack':
        if answer[-1] == '.':
            answer = answer[:-1]
        try:
            return int(answer[-1])
        except:
            raise ValueError("The answer is not valid")
    else:
        raise ValueError(f"The game {game} is not supported")

# The function to parse the observation
def parse_obs(obs, game):
    if game == 'jack':
        obs = eval(obs)
        my_sum = obs[0]
        dealer_sum = obs[1]
        prompt = f"Player's sum is {my_sum}, and the dealer's face-on card is {dealer_sum}."
        return prompt
    else:
        raise ValueError(f"The game {game} is not supported")
    

def parse_obs_dnd(obs, history, valid_action, his_len = 3):
    if len(history) >= his_len:
        # get the last his_len history in the history dict
        history = dict(list(history.items())[-his_len:])
    
    history_prompt = "**The following are game history:** \n"

    for id, value in history.items():
        state, action, reason, valid = value
        step_id = len(history) - id
        history_prompt += f"{step_id} steps before, you observed:\n {state} You took the action:\n {action}\n"
    
    obs_prompt = f"**Now you observed the current state:**\n {str(obs)}\n"
    valid_action_prompt = "The cuurent valid actions are: \n"
    for id, action in enumerate(valid_action):
        valid_action_prompt += f"{id}: {CNmap[action]}\n"

    return history_prompt + obs_prompt + valid_action_prompt




def load_char_data(game, character):
    file_path = f"data/{game}_{character}_v3.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
    else:
        raise ValueError(f"The file {file_path} does not exist")
    return data






def build_dataset(game, characters = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']):
    dataset = {}
    """
    The structure of dataset should be like:
    (1) state
    (2) character
    (3) reason
    (4) action
    """
    cnt = 0
    for character in characters:
        data = load_char_data(game, character)
        for id in tqdm(data):
            item = data[id]
            state = (eval(item["state"])[0], eval(item["state"])[1])
            reason = item["action"]
            action = parse_answer(reason)
            dataset[cnt] = {"state": state, "character": colleges_map[character], "reason": reason, "action": action}
            cnt += 1
    # Save the dataset
    save_path = f"data/{game}_all_characters_v3.json"
    with open(save_path, "w") as file:
        json.dump(dataset, file)
 
    return dataset


if __name__ == '__main__':
    dataset = build_dataset("jack", characters = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw'])
    print("Finished building dataset")
    print('-'*50)
    print("The first 10 items in the dataset:")
    print(dataset[0])