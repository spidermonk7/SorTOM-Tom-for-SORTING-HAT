from story_teller.story import StoryTree
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
model = BertModel.from_pretrained(model_name).to(device)

action_dict = {"Explore": 0, "Help": 0, "Refuse": 0, "Betray": 0, "Fight": 0, "Escape": 0}
CNmap = {"探索调查": "Explore", "帮助NPC": "Help", "拒绝NPC": "Refuse", "背叛NPC": "Betray", "战斗": "Fight", "逃跑": "Escape"}

openai.api_key = "sk-VpHQwCIkYGslHgM5990c07C057184fC2AcB9Ef81543dC2E6"
openai.base_url = "https://api.v3.cm/v1/"
openai.default_headers = {"x-foo": "true"}
  


def run_batch(character="Gryffindor", game="dnd", max_step=6, record_window=3, print_out = True, batch_only_folder="batch_only"):
    for id in tqdm(range(5,6), desc=f"{character}: Batches"):
        # if id == 14: continue
        base_path = f"./dataset/story_tree_en/story_tree__batch_id_{id}.npy"
        story_tree = StoryTree()
        history = {}
        if batch_only_folder == "batch_only":
            obs, valid_action = story_tree.reset()
        else:
            obs, valid_action = story_tree.reset(path=base_path)
            print(f"Reset the tree with path {base_path}")

        for i in range(max_step):
            if print_out:
                print(f"-----------------round {i}-----------------")
                print(f"模型观测:", obs)
                print(f"合理动作:", valid_action)
                
            llm_input = parse_obs_dnd(obs, history, valid_action, record_window)
            # print(f"模型输入:", llm_input)
            action = generate_answer(llm_input, character, game)
            reason = action.split("[Action]")[0]
            # print(f"模型输出:", action)
            
               
            action = parse_answer(action, game)
            try:
                history[i] = {"current_obs":obs,  "action":CNmap[valid_action[action]], "reason":reason, "valid_action":[CNmap[act] for act in valid_action]}
            except:
                print("The action is not valid")
                print(f"the action is {action}")
                print(f"the valid actions are: {valid_action}")
            if print_out:
                print("新状态:", obs)
                print("动作:", CNmap[valid_action[action]])
            obs, valid_action = story_tree.step(action)

        # dump the history 
        check_path(f"dataset/final_data_en/{character}/{batch_only_folder}")
        with open(f"dataset/final_data_en/{character}/{batch_only_folder}/{game}_batch{id}_{character}.json", "w") as file:
            json.dump(history, file)        

### Modified by Yichen
def load_data(game, character, is_test=False, batch_only_folder = "batch_only"):
    # random select 3 batches as test set
    dataset = []
    if is_test:
        for id in range(16, 19):
            file_path = f"./dataset/final_data_en/{character}/{batch_only_folder}/{game}_batch{id}_{character}.json"
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    data = json.load(file)
                    dataset.append(data)
            else:
                raise ValueError(f"The file {file_path} does not exist")
        return dataset
    else:
        for id in range(0, 16):
            if id == 5: continue
            file_path = f"./dataset/final_data_en/{character}/{batch_only_folder}/{game}_batch{id}_{character}.json"
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    data = json.load(file)
                    dataset.append(data)
            else:
                raise ValueError(f"The file {file_path} does not exist")
        return dataset

### Modified by Yichen
def get_trajectory_txt(character="Ravenclaw", window_size=6, is_test=False, batch_only_folder = "batch_only"):
    state_action_pair = {}
    assert batch_only_folder in ["batch_only", "all_batches"]
    if character != "Hogwarts":
        data = load_data(game="dnd", character=character, is_test=is_test, batch_only_folder = batch_only_folder)
        for batch_id, batch_data in enumerate(data):
            print(f"Handling data batch {batch_id} of character {character}")
            for id in range(window_size, len(batch_data)):
                s_a = {"trajectory": [(batch_data[str(i)]['current_obs'], batch_data[str(i)]['action']) for i in range(id - window_size, id)], "action":batch_data[str(id)]['action'], "state":batch_data[str(id)]['current_obs']}
                state_action_pair[f"{character}_batch{batch_id}_step{id}"] = s_a

    else:
        for characters in ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]:
            data = load_data(game="dnd", character=characters, is_test=is_test, batch_only_folder=batch_only_folder)
            for batch_id, batch_data in enumerate(data):
                print(f"Handling data batch {batch_id} of character {characters}")
                for id in range(window_size, len(batch_data)):
                    s_a = {"trajectory": [(batch_data[str(i)]['current_obs'], batch_data[str(i)]['action']) for i in range(id - window_size, id)], "action":batch_data[str(id)]['action'], "state":batch_data[str(id)]['current_obs']}
                    state_action_pair[f"{characters}_batch{batch_id}_step{id}"] = s_a

    check_path(f"./dataset/Trajectory/{batch_only_folder}/window_{window_size}/{character}")
    if is_test:
        with open(f"./dataset/Trajectory/{batch_only_folder}/window_{window_size}/{character}/{character}-test.json", "w") as file:
                json.dump(state_action_pair, file)
    else:
        with open(f"./dataset/Trajectory/{batch_only_folder}/window_{window_size}/{character}/{character}-train.json", "w") as file:
                json.dump(state_action_pair, file)
                
### Modified by Yichen           
def got_trajectory_embeded(window_size = 2, character = "Slytherin", is_test=False, batch_only_folder = "batch_only"):

    if is_test:
        path = f"./dataset/Trajectory/{batch_only_folder}/window_{window_size}/{character}/{character}-test.json"
    else:
        path = f"./dataset/Trajectory/{batch_only_folder}/window_{window_size}/{character}/{character}-train.json"
        
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
                # set device
                trj_inputs['input_ids'] = trj_inputs['input_ids'].to(device)
                trj_inputs['attention_mask'] = trj_inputs['attention_mask'].to(device)

                trj_action['input_ids'] = trj_action['input_ids'].to(device)
                trj_action['attention_mask'] = trj_action['attention_mask'].to(device)

                trj_embedded_state = model(trj_inputs['input_ids'], trj_inputs['attention_mask'])
                trj_embedded_state = trj_embedded_state.pooler_output
                trj_embedded_action = model(trj_action['input_ids'], trj_action['attention_mask'])
                trj_embedded_action = trj_embedded_action.pooler_output
                trj_embedded = torch.cat((trj_embedded_state, trj_embedded_action), dim=0)
                embeddings_trajectory.append(trj_embedded.cpu().numpy())
        
        embeddings_trajectories.append(embeddings_trajectory)

        # 获取标签
        label = label_dic[action]
        labels.append(label)

        # 使用 BERT Tokenizer进行编码
        inputs = tokenizer(state, return_tensors='pt', max_length=200, truncation=True, padding="max_length")
        # set device
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        # 获取 BERT 的输出
        with torch.no_grad():
            embedded_state = model(inputs['input_ids'], inputs['attention_mask'])
            embedded_state = embedded_state.pooler_output.squeeze(0)  # (sequence_length, hidden_size)

        embeddings.append(embedded_state.cpu().numpy())  # 转为 NumPy 数组并保存

    check_path(f"./dataset/Trajectory/embedded/{batch_only_folder}/window_{window_size}/{character}")
    # 将嵌入数据保存为 .npy 文件

    embeddings_trajectories = np.array(embeddings_trajectories)
    embeddings_trajectories = embeddings_trajectories.reshape(embeddings_trajectories.shape[0], embeddings_trajectories.shape[1]*embeddings_trajectories.shape[2], -1)
    embeddings = np.array(embeddings)
    print(f"embeddings.shape = {embeddings.shape}")
    print(f"embeddings_trajectories.shape = {embeddings_trajectories.shape}")


    if is_test:
        embedding_path = f"./dataset/Trajectory/embedded/{batch_only_folder}/window_{window_size}/{character}/{character}_all_embedded-test.npy"
        np.save(embedding_path, embeddings)  # 保存为 NumPy 文件

        embedding_trajectory_path = f"./dataset/Trajectory/embedded/{batch_only_folder}/window_{window_size}/{character}/{character}_all_trajectory_embedded-test.npy"
        np.save(embedding_trajectory_path, embeddings_trajectories)  # 保存为 NumPy 文件

        # 保存标签
        label_path = f"./dataset/Trajectory/embedded/{batch_only_folder}/window_{window_size}/{character}/{character}_labels-test.npy"
        np.save(label_path, np.array(labels))  # 保存标签数据
    else:
        embedding_path = f"./dataset/Trajectory/embedded/{batch_only_folder}/window_{window_size}/{character}/{character}_all_embedded-train.npy"
        np.save(embedding_path, embeddings)  # 保存为 NumPy 文件

        embedding_trajectory_path = f"./dataset/Trajectory/embedded/{batch_only_folder}/window_{window_size}/{character}/{character}_all_trajectory_embedded-train.npy"
        np.save(embedding_trajectory_path, embeddings_trajectories)  # 保存为 NumPy 文件

        # 保存标签
        label_path = f"./dataset/Trajectory/embedded/{batch_only_folder}/window_{window_size}/{character}/{character}_labels-train.npy"
        np.save(label_path, np.array(labels))  # 保存标签数据


### Modified by Yichen
class TrajectoryDataset(Dataset):
    def __init__(self, character="Slytherin", alpha=[0.9], model_name='bert-base-uncased', max_length=230, window_size = 2, is_test=False, batch_only_folder = "batch_only", zero_trajectory = False):
        self.character = character
        self.alpha = alpha
        self.max_length = max_length
        self.model_name = model_name
        
        # 加载预先保存的 BERT 嵌入数据
        if is_test:
            embeddings_path = f"./dataset/Trajectory/embedded/{batch_only_folder}/window_{window_size}/{self.character}/{self.character}_all_embedded-test.npy"
            trajectory_embeddings_path = f"./dataset/Trajectory/embedded/{batch_only_folder}/window_{window_size}/{self.character}/{self.character}_all_trajectory_embedded-test.npy"
            self.embeddings = np.load(embeddings_path)  # 加载嵌入数据
            self.traj_embeddings = np.load(trajectory_embeddings_path)  # 加载嵌入数据

            # 加载标签
            labels_path = f"./dataset/Trajectory/embedded/{batch_only_folder}/window_{window_size}/{self.character}/{self.character}_labels-test.npy"
            self.labels = np.load(labels_path)  # 加载标签数据

        else:
            embeddings_path = f"./dataset/Trajectory/embedded/{batch_only_folder}/window_{window_size}/{self.character}/{self.character}_all_embedded-train.npy"
            trajectory_embeddings_path = f"./dataset/Trajectory/embedded/{batch_only_folder}/window_{window_size}/{self.character}/{self.character}_all_trajectory_embedded-train.npy"
            self.embeddings = np.load(embeddings_path)  # 加载嵌入数据
            self.traj_embeddings = np.load(trajectory_embeddings_path)  # 加载嵌入数据

            # 加载标签
            labels_path = f"./dataset/Trajectory/embedded/{batch_only_folder}/window_{window_size}/{self.character}/{self.character}_labels-train.npy"
            self.labels = np.load(labels_path)  # 加载标签数据

        if zero_trajectory:
            self.traj_embeddings = np.zeros_like(self.traj_embeddings)

        # get the distribution of labels 
        self.distribution = torch.zeros(6, device=device)
        for label in self.labels:
            self.distribution[label] += 1

        non_zero_values = self.distribution[self.distribution > 0]

        # 2. 找到非零元素中的最小值
        min_non_zero = non_zero_values.min() if non_zero_values.numel() > 0 else 0  # 防止没有非零值时出错

        # 3. 将最小非零值加到分布上
        self.distribution += min_non_zero
        self.distribution = self.distribution / self.distribution.sum()



    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedded_state = torch.tensor(self.embeddings[idx])  # 加载预处理的嵌入
        trajction_embedded = torch.tensor(self.traj_embeddings[idx])
        label = torch.tensor(self.labels[idx])  # 获取标签
        return trajction_embedded, embedded_state, label



if __name__ == '__main__':
    # action_distribution_ana()
    # 把batch id抽离出来做分析。 
    batch_only_folder = "all_batches"
    # for window_size in [2, 3, 4]:
    #     for character in ['joey']:
    #         get_trajectory_txt(character, window_size, batch_only_folder=batch_only_folder)
    #         got_trajectory_embeded(window_size, character, batch_only_folder=batch_only_folder)
    #         get_trajectory_txt(character, window_size, is_test=True, batch_only_folder=batch_only_folder)
    #         got_trajectory_embeded(window_size, character, is_test=True, batch_only_folder=batch_only_folder)