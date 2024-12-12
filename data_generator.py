from story_teller.story import StoryNode, StoryTree
from character_prompt_test.chat_demo import generate_answer
from character_prompt_test.utils import colleges_map, parse_answer, parse_obs_dnd
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import openai
from utils import *
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader


# 加载 BERT tokenizer 和模型
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


action_dict = {"Explore": 0, "Help": 0, "Refuse": 0, "Betray": 0, "Fight": 0, "Escape": 0}
CNmap = {"探索调查": "Explore", "帮助NPC": "Help", "拒绝NPC": "Refuse", "背叛NPC": "Betray", "战斗": "Fight", "逃跑": "Escape"}

openai.api_key = "sk-MSSwI7MgizQFSyUE64359c5000D64b518cCc7c00F30e0321"
openai.base_url = "https://api.gpt.ge/v1/"
# openai.base_url = "https://api.v3.cm/v1/"
openai.default_headers = {"x-foo": "true"}


def run(character="Gryffindor", game="dnd", max_step=7, record_window=3, print_out = False):
    story_tree = StoryTree()
    story_tree.load_story_tree()
    history = {}
    obs, valid_action = story_tree.reset()

    for i in range(max_step):
        llm_input = parse_obs_dnd(obs, history, valid_action, record_window)
        action = generate_answer(llm_input, character, game)
        reason = action.split("[Action]")[0]
        if print_out:
            print(f"-----------------round {i}-----------------")
            print(f"模型输出:", action)
        action = parse_answer(action, game)
        history[i] = {"current_obs":obs,  "action":valid_action[action], "reason":reason}
        if print_out:
            print("新状态:", obs)
            print("动作:", valid_action[action])
        obs, valid_action = story_tree.step(action)

    # dump the history 
    with open(f"dataset/final_data/{game}0_{character}.json", "w") as file:
        json.dump(history, file)
        
def run_batch(character="Gryffindor", game="dnd", max_step=7, record_window=3, print_out = True):
    for id in tqdm(range(0,19), desc=f"{character}: Batches"):
        base_path = f"./dataset/story_tree_en/story_tree__batch_id_{id}.npy"
        story_tree = StoryTree()
        story_tree.load_story_tree(base_path)
        history = {}
        obs, valid_action = story_tree.reset()
        for i in range(max_step):
            llm_input = parse_obs_dnd(obs, history, valid_action, record_window)
            action = generate_answer(llm_input, character, game)
            reason = action.split("[Action]")[0]
            if print_out:
                print(f"-----------------round {i}-----------------")
                print(f"合理动作:", valid_action)
                print(f"模型输出:", action)
               
            action = parse_answer(action, game)
            history[i] = {"current_obs":obs,  "action":CNmap[valid_action[action]], "reason":reason, "valid_action":[CNmap[act] for act in valid_action]}
            if print_out:
                print("新状态:", obs)
                print("动作:", CNmap[valid_action[action]])
            obs, valid_action = story_tree.step(action)

        # dump the history 
        with open(f"dataset/final_data_en/{game}_batch{id}_{character}.json", "w") as file:
            json.dump(history, file)        

# def load_data(game, character):
#     dataset = []
#     for id in range(19):
#         file_path = f"./dataset/final_data_en/{game}_batch{id}_{character}.json"
#         if os.path.exists(file_path):
#             with open(file_path, "r") as file:
#                 data = json.load(file)
#                 dataset.append(data)
#         else:
#             raise ValueError(f"The file {file_path} does not exist")
#     return dataset

### Modified by Yichen
def load_data(game, character, is_test=False):
    dataset = []
    if is_test:
        for id in range(16, 19):
            file_path = f"./dataset/final_data_en/{game}_batch{id}_{character}.json"
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    data = json.load(file)
                    dataset.append(data)
            else:
                raise ValueError(f"The file {file_path} does not exist")
        return dataset
    else:
        for id in range(16):
            file_path = f"./dataset/final_data_en/{game}_batch{id}_{character}.json"
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    data = json.load(file)
                    dataset.append(data)
            else:
                raise ValueError(f"The file {file_path} does not exist")
        return dataset

def action_distribution_ana():
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    # add grid
    colors = {'Gryffindor': '#FF6F61',  # 柔和红
              'Hufflepuff': '#FFD700',  # 柔和黄
              'Ravenclaw': '#6495ED',  # 柔和蓝
              'Slytherin': '#3CB371'}  # 柔和绿

    for idx, character in enumerate(["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]):
        action_dict = {"Explore": 0, "Help": 0, "Refuse": 0, "Betray": 0, "Fight": 0, "Escape": 0}
        CNmap = {"探索调查": "Explore", "帮助NPC": "Help", "拒绝NPC": "Refuse", "背叛NPC": "Betray", "战斗": "Fight", "逃跑": "Escape"}

        dataset = load_data("dnd", character)  # 确保这个函数能正确加载数据
        for data in dataset:
            for id, value in data.items():
                action = value["action"]
             
                if action in action_dict:
                    action_dict[action] += 1

        # normalize the action_dict
        total = sum(action_dict.values())
        for key in action_dict:
            action_dict[key] /= total

        ax[int(idx / 2), idx % 2].bar(action_dict.keys(), action_dict.values(), color=colors[character], width=0.5)
        ax[int(idx / 2), idx % 2].set_title(character)
        ax[int(idx / 2), idx % 2].set_ylim(0, 1)
        ax[int(idx / 2), idx % 2].grid(True)

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig("action_distribution.png")
    plt.show()

    return action_dict



def get_SA_txt(character = "Ravenclaw"):
    state_action_pair = {}


    if character != "Hogwarts":
        data = load_data(game="dnd", character=character)
        for batch_id, batch_data in enumerate(data):
            print(f"Handling data batch {batch_id} of character {character}")
            for id in batch_data:
                s_a = {"state": batch_data[id]['current_obs'], "action":batch_data[id]['action']}
                state_action_pair[f"{character}_batch{batch_id}_step{id}"] = s_a

    else:
        for character in ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]:
            data = load_data(game="dnd", character=character)
            for batch_id, batch_data in enumerate(data):
                print(f"Handling data batch {batch_id} of character {character}")
                # choose a window size of W, got three keys: trajectory with windowsize W, current_obs, action
                for id in batch_data:
                    s_a = {"state": batch_data[id]['current_obs'], "action":batch_data[id]['action']}
                    state_action_pair[f"{character}_batch{batch_id}_step{id}"] = s_a

    check_path(f"./dataset/character_wise/")
    with open(f"./dataset/character_wise/{character}_all.json", "w") as file:
            json.dump(state_action_pair, file)

def got_SA_embeded():
    # 获取数据集
    for character in ["Slytherin", "Gryffindor", "Hufflepuff", "Ravenclaw", "Hogwarts"]:
        # 加载数据
        path = f"./dataset/character_wise/{character}_all.json"
        with open(path, 'r') as file:
            data = json.load(file)

        # 存储嵌入的列表
        embeddings = []
        labels = []  # 标签列表，用于后续训练
        label_dic = {"Explore": 0, "Help": 1, "Refuse": 2, "Betray": 3, "Fight": 4, "Escape": 5}

        # 生成 BERT 嵌入
        for key in tqdm(data, desc="Generating BERT embeddings:"):
            # 获取数据
            sample = data[key]
            state = sample['state']
            action = sample['action']
            
            # 获取标签
            label = label_dic[action]
            labels.append(label)

            # 使用 BERT Tokenizer进行编码
            inputs = tokenizer(state, return_tensors='pt', max_length=230, truncation=True, padding="max_length")

            # 获取 BERT 的输出
            with torch.no_grad():
                embedded_state = model(**inputs)
                embedded_state = embedded_state.last_hidden_state.squeeze(0)  # (sequence_length, hidden_size)

            embeddings.append(embedded_state.numpy())  # 转为 NumPy 数组并保存

        # 将嵌入数据保存为 .npy 文件
        embedding_path = f"./dataset/character_wise/embedded/{character}_all_embedded.npy"
        np.save(embedding_path, np.array(embeddings))  # 保存为 NumPy 文件

        # 保存标签
        label_path = f"./dataset/character_wise/embedded/{character}_labels.npy"
        np.save(label_path, np.array(labels))  # 保存标签数据

### Modified by Yichen
def get_trajectory_txt(character="Ravenclaw", window_size=6, is_test=False):
    state_action_pair = {}

    if character != "Hogwarts":
        data = load_data(game="dnd", character=character, is_test=is_test)
        for batch_id, batch_data in enumerate(data):
            print(f"Handling data batch {batch_id} of character {character}")
            for id in range(window_size, len(batch_data)):
                s_a = {"trajectory": [(batch_data[str(i)]['current_obs'], batch_data[str(i)]['action']) for i in range(id - window_size, id)], "action":batch_data[str(id)]['action'], "state":batch_data[str(id)]['current_obs']}
                state_action_pair[f"{character}_batch{batch_id}_step{id}"] = s_a

    else:
        for characters in ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]:
            data = load_data(game="dnd", character=characters, is_test=is_test)
            for batch_id, batch_data in enumerate(data):
                print(f"Handling data batch {batch_id} of character {characters}")
                for id in range(window_size, len(batch_data)):
                    s_a = {"trajectory": [(batch_data[str(i)]['current_obs'], batch_data[str(i)]['action']) for i in range(id - window_size, id)], "action":batch_data[str(id)]['action'], "state":batch_data[str(id)]['current_obs']}
                    state_action_pair[f"{characters}_batch{batch_id}_step{id}"] = s_a

    if is_test:
        check_path(f"./dataset/Trajectory/window_{window_size}/")
        with open(f"./dataset/Trajectory/window_{window_size}/{character}-test.json", "w") as file:
                json.dump(state_action_pair, file)
    else:
        check_path(f"./dataset/Trajectory/window_{window_size}/")
        with open(f"./dataset/Trajectory/window_{window_size}/{character}.json", "w") as file:
                json.dump(state_action_pair, file)
                
### Modified by Yichen           
def got_trajectory_embeded(window_size = 2, character = "Slytherin", is_test=False):

    if is_test:
        path = f"./dataset/Trajectory/window_{window_size}/{character}-test.json"
    else:
        path = f"./dataset/Trajectory/window_{window_size}/{character}.json"
        
    with open(path, 'r') as file:
        data = json.load(file)

    # 存储嵌入的列表
    embeddings = []

    embeddings_trajectories = []
    labels = []  # 标签列表，用于后续训练
    label_dic = {"Explore": 0, "Help": 1, "Refuse": 2, "Betray": 3, "Fight": 4, "Escape": 5}
    
    # 生成 BERT 嵌入
    for key in tqdm(data, desc="Generating BERT embeddings:"):
        # 获取数据
        sample = data[key]
        state = sample['state']
        action = sample['action']
        trajectory = sample['trajectory']
        
        with torch.no_grad():
            embeddings_trajectory = []
            for trj_state, trj_action in trajectory:
                trj_inputs = tokenizer(trj_state, return_tensors='pt', max_length=200, truncation=True, padding="max_length")
                trj_action = tokenizer(trj_action, return_tensors='pt')

                trj_embedded_state = model(**trj_inputs)
                trj_embedded_state = trj_embedded_state.last_hidden_state.squeeze(0)

                trj_embedded_action = model(**trj_action)
                trj_embedded_action = trj_embedded_action.last_hidden_state.squeeze(0)

                trj_embedded = torch.cat((trj_embedded_state, trj_embedded_action), dim=0)

                embeddings_trajectory.append(trj_embedded.numpy())
        
        embeddings_trajectories.append(embeddings_trajectory)

        # 获取标签
        label = label_dic[action]
        labels.append(label)

        # 使用 BERT Tokenizer进行编码
        inputs = tokenizer(state, return_tensors='pt', max_length=200, truncation=True, padding="max_length")

        # 获取 BERT 的输出
        with torch.no_grad():
            embedded_state = model(**inputs)
            embedded_state = embedded_state.last_hidden_state.squeeze(0)  # (sequence_length, hidden_size)

        embeddings.append(embedded_state.numpy())  # 转为 NumPy 数组并保存

    check_path(f"./dataset/Trajectory/embedded/window_{window_size}/")
    # 将嵌入数据保存为 .npy 文件

    if is_test:
        embedding_path = f"./dataset/Trajectory/embedded/window_{window_size}/{character}_all_embedded-test.npy"
        np.save(embedding_path, np.array(embeddings))  # 保存为 NumPy 文件

        embedding_trajectory_path = f"./dataset/Trajectory/embedded/window_{window_size}/{character}_all_trajectory_embedded-test.npy"
        np.save(embedding_trajectory_path, np.array(embeddings_trajectories))  # 保存为 NumPy 文件

        # 保存标签
        label_path = f"./dataset/Trajectory/embedded/window_{window_size}/{character}_labels-test.npy"
        np.save(label_path, np.array(labels))  # 保存标签数据
    else:
        embedding_path = f"./dataset/Trajectory/embedded/window_{window_size}/{character}_all_embedded.npy"
        np.save(embedding_path, np.array(embeddings))  # 保存为 NumPy 文件

        embedding_trajectory_path = f"./dataset/Trajectory/embedded/window_{window_size}/{character}_all_trajectory_embedded.npy"
        np.save(embedding_trajectory_path, np.array(embeddings_trajectories))  # 保存为 NumPy 文件

        # 保存标签
        label_path = f"./dataset/Trajectory/embedded/window_{window_size}/{character}_labels.npy"
        np.save(label_path, np.array(labels))  # 保存标签数据





class TrajectoryDataset(Dataset):
    def __init__(self, character="Slytherin", alpha=[0.9], model_name='bert-base-uncased', max_length=230, window_size = 2):
        self.character = character
        self.alpha = alpha
        self.max_length = max_length
        self.model_name = model_name
        
        # 加载预先保存的 BERT 嵌入数据
        embeddings_path = f"./dataset/Trajectory/embedded/window_{window_size}/{self.character}_all_embedded.npy"
        trajectory_embeddings_path = f"./dataset/Trajectory/embedded/window_{window_size}/{self.character}_all_trajectory_embedded.npy"
        self.embeddings = np.load(embeddings_path)  # 加载嵌入数据
        self.traj_embeddings = np.load(trajectory_embeddings_path)  # 加载嵌入数据

        # 加载标签
        labels_path = f"./dataset/Trajectory/embedded/window_{window_size}/{self.character}_labels.npy"
        self.labels = np.load(labels_path)  # 加载标签数据

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedded_state = torch.tensor(self.embeddings[idx])  # 加载预处理的嵌入
        trajction_embedded = torch.tensor(self.traj_embeddings[idx])
        label = torch.tensor(self.labels[idx])  # 获取标签
        return trajction_embedded, embedded_state, label



if __name__ == '__main__':
    for window_size in range(2, 7):
        for character in {"Slytherin", "Gryffindor", "Hufflepuff", "Ravenclaw", "Hogwarts"}:
            get_trajectory_txt(character, window_size)
            got_trajectory_embeded(window_size, character)
            get_trajectory_txt(character, window_size, is_test=True)
            got_trajectory_embeded(window_size, character, is_test=True)