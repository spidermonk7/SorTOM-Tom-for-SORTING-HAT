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
from data_generator import get_trajectory_txt, got_trajectory_embeded, TrajectoryDataset
import random
import torch
from transformers import BertTokenizer, BertModel
from model import SortingHat
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

from torch.utils.data import Dataset, DataLoader
from data_generator import *
import torch
import torch.nn as nn
from utils import * 
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import SortingHat, FocalLoss
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F


CNmap = {"探索调查": "Explore", "帮助NPC": "Help", "拒绝NPC": "Refuse", "背叛NPC": "Betray", "战斗": "Fight", "逃跑": "Escape"}


def collect_testee_trajectory(character="joey", game="dnd", max_step=6, record_window=3, print_out = True, batch_only_folder = "all_batches"):
    for id in tqdm(range(0, 16), desc=f"{character}: Batches"):
        base_path = f"./dataset/story_tree_en/story_tree__batch_id_{id}.npy"
        story_tree = StoryTree()
        if id == 5:
            continue
        # story_tree.load_story_tree(path=base_path)
        history = {}
        obs, valid_action = story_tree.reset(base_path)
        for i in range(max_step):
            print(f"Step is : {i}")
            print(f"当前状态:", obs)
            print(f"合理动作:", valid_action)
            action = int(input(f"Please input the action for the character {character}:\n"))
            history[i] = {"current_obs":obs,  "action":CNmap[valid_action[action]], "valid_action":[CNmap[act] for act in valid_action]}
            obs, valid_action = story_tree.step(action)
        
        save_path = f"dataset/final_data_en/{character}/{batch_only_folder}"
        check_path(save_path)
        # dump the history 
        with open(f"dataset/final_data_en/{character}/{batch_only_folder}/{game}_batch{id}_{character}.json", "w") as file:
            json.dump(history, file)        






def sorting_hat(window_sizes = [2, 3, 4], tester_dataset = None, batch_only_folder = "all_batches", test_character = "joey"):
    # 假设你有 model, datasets, colors 等已定义
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, len(window_sizes), figsize=(18, 6), sharey=True)

    for id, window_size in enumerate(window_sizes):
        model = SortingHat()
        model.load_state_dict(torch.load(f"./models/{batch_only_folder}/SortingHat_Hogwarts_{window_size}.pt", map_location=torch.device('cpu'), weights_only=True))
        print(f"model loaded...")

        Gryffindor_dataset = TrajectoryDataset(character='Gryffindor', alpha=0.9, window_size=window_size, is_test=False, batch_only_folder=batch_only_folder)
        Hufflepuff_dataset = TrajectoryDataset(character='Hufflepuff', alpha=0.9, window_size=window_size, is_test=False, batch_only_folder=batch_only_folder)
        Ravenclaw_dataset = TrajectoryDataset(character='Ravenclaw', alpha=0.9, window_size=window_size, is_test=False, batch_only_folder=batch_only_folder)
        Slytherin_dataset = TrajectoryDataset(character='Slytherin', alpha=0.9, window_size=window_size, is_test=False, batch_only_folder=batch_only_folder)
        tester_dataset = TrajectoryDataset(character=test_character, alpha=0.9, window_size=window_size, is_test=False, batch_only_folder=batch_only_folder)
    
        dic_embeddings = {"Gryffindor": [], "Hufflepuff": [], "Ravenclaw": [], "Slytherin": [], "Tester": []}
        tester_dataset.character = "Tester"

        # 提取 embeddings
        for dataset in [Gryffindor_dataset, Hufflepuff_dataset, Ravenclaw_dataset, Slytherin_dataset, tester_dataset]:
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            with torch.no_grad():
                for trajectory, data, label in loader:
                    output = model(trajectory, data)
                    character_embed = model.chara
                    dic_embeddings[dataset.character].append(character_embed.squeeze(0).numpy())

        embeddings = []
        labels = []
        for key in dic_embeddings:
            embeddings.extend(dic_embeddings[key])
            labels += [key] * len(dic_embeddings[key])

        embeddings = np.array(embeddings)
        embeddings = embeddings.reshape(embeddings.shape[0], -1)

        distributions = {"Gryffindor": None, "Hufflepuff": None, "Ravenclaw": None, "Slytherin": None, "Tester": None}

        for idx, character in enumerate(dic_embeddings.keys()):
            char_embeddings = np.array(dic_embeddings[character]).reshape(-1, embeddings.shape[1])
            # 对该角色数据进行KDE拟合
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
            distributions[character] = kde.fit(char_embeddings)
        
        
        kl_tester_gryffindor = kl_divergence(distributions["Tester"], distributions["Gryffindor"])
        kl_tester_hufflepuff = kl_divergence(distributions["Tester"], distributions["Hufflepuff"])
        kl_tester_ravenclaw = kl_divergence(distributions["Tester"], distributions["Ravenclaw"])
        kl_tester_slytherin = kl_divergence(distributions["Tester"], distributions["Slytherin"])


        sum_p = np.exp( -kl_tester_gryffindor) + np.exp( -kl_tester_hufflepuff) + np.exp( -kl_tester_ravenclaw) + np.exp( -kl_tester_slytherin)

        P_gryffindor = np.exp( -kl_tester_gryffindor)/sum_p
        P_hufflepuff = np.exp( -kl_tester_hufflepuff)/sum_p
        P_ravenclaw = np.exp( -kl_tester_ravenclaw)/sum_p
        P_slytherin = np.exp( -kl_tester_slytherin)/sum_p

        probabilities = [P_gryffindor, P_hufflepuff, P_ravenclaw, P_slytherin]
        labels = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
        colors = ["#FF6F61", "#6DC066", "#F7C242", "#4472C4"]  # 自定义配色

        # 绘制柱状图
        bars = ax[id].bar(labels, probabilities, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

        # 设置标题和轴标签
        ax[id].set_title(f"Window size: {window_size}", fontsize=14, fontweight='bold')
        ax[id].set_ylim(0, 1)
        ax[id].tick_params(axis='x', labelrotation=0, labelsize=10)
        ax[id].set_xlabel("Houses", fontsize=12)
        if id == 0:
            ax[id].set_ylabel("Probability", fontsize=12)
    
       
        plt.grid(axis='y', linestyle='--', linewidth=1, alpha=0.7)


        # 添加数据标签
        for bar, prob in zip(bars, probabilities):
            ax[id].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{prob:.2f}", ha='center', fontsize=10)
        
    plt.tight_layout()

    fig.set_alpha(0)
    check_path(f"./results/{batch_only_folder}/sorting_res/") 
    plt.savefig(f"./results/{batch_only_folder}/sorting_res/Hogwarts_KL_divergence.png", transparent=True)
    plt.show()
    plt.close()




if __name__ == "__main__":
    character = input("Please input the user name:\n")
    # character = "joey"
    if not os.path.exists(f"./dataset/final_data_en/{character}"):
        print(f"Your ALBUS is not found, please record it for further analysis...")
        print(f"===============================")
        print(f"The GELLERT Games will start soon...")
        time.sleep(3)
        collect_testee_trajectory(max_step=6)
    else:
        print(f"Welecome back, {character}! We found your ALBUS, let's start analyse it~")

    # get the trajectory of the character
    batch_only_folder = "all_batches"
    for window_size in [2, 3, 4]:
        if not os.path.exists(f"./dataset/Trajectory/all_batches/window_{window_size}/{character}"):
            get_trajectory_txt(character, window_size, batch_only_folder=batch_only_folder)
        if not os.path.exists(f"./dataset/Trajectory/embedded/all_batches/window_{window_size}/{character}"):
            got_trajectory_embeded(window_size, character, batch_only_folder=batch_only_folder)

    sorting_hat(batch_only_folder=batch_only_folder, test_character=character)

