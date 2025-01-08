import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

from utils import *
from data_generator import *
from model import SortingHat, FocalLoss


softmax_layer = nn.Softmax(dim=1)


def action_distribution_ana():
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 4, figsize=(32, 8))

    ax = ax.flatten()
    # add grid
    colors = {'Gryffindor': '#FF6F61',  # 柔和红
              'Hufflepuff': '#FFD700',  # 柔和黄
              'Ravenclaw': '#6495ED',  # 柔和蓝
              'Slytherin': '#3CB371'}  # 柔和绿

    for idx, character in enumerate(["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]):
        action_dict = {"Explore": 0, "Help": 0, "Refuse": 0, "Betray": 0, "Fight": 0, "Escape": 0}
        CNmap = {"探索调查": "Explore", "帮助NPC": "Help", "拒绝NPC": "Refuse", "背叛NPC": "Betray", "战斗": "Fight", "逃跑": "Escape"}

        dataset = load_data("dnd", character, batch_only_folder="all_batches")  # 确保这个函数能正确加载数据
        for data in dataset:
            for id, value in data.items():
                action = value["action"]
                if action in action_dict:
                    if action == "Betray" and character == "Gryffindor":
                        print(f"Betray in Gryffindor")
                        print(value['current_obs'])
                        print(value['valid_action'])
                        print(value['action'])
                        print(value['reason'])
                        
                    action_dict[action] += 1

        test_sets = load_data("dnd", character, is_test=True, batch_only_folder="all_batches")
        for data in test_sets:
            for id, value in data.items():
                action = value["action"]
                if action in action_dict:
                    action_dict[action] += 1

         # normalize the action_dict
        total = sum(action_dict.values())
        for key in action_dict:
            action_dict[key] /= total

        ax[idx].bar(action_dict.keys(), action_dict.values(), color=colors[character], width=0.5)
        ax[idx].set_title(character, fontsize=30)
        ax[idx].tick_params(axis='x', labelsize=25)
        ax[idx].set_ylim(0, 0.6)
        ax[idx].grid(True)

        # only show y label for the first plot
        if idx != 0:
            ax[idx].set_yticklabels([])


    # 将整个画布背景设置为透明
    fig.patch.set_alpha(0)


    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig("action_distribution.png")
    plt.show()

    return action_dict


### Modified by Yichen
def experiment2_Chara_Ment_Ss_A_prediction(window_size = 3, character = "Hogwarts", result_path = None, batch_only_folder = "all_batches"):
    assert window_size in [2, 3, 4] , "window_size should be one of 2, 3, 4"

    test_dataset = TrajectoryDataset(character=character, alpha=0.9, window_size=window_size, is_test=True, batch_only_folder=batch_only_folder)
    train_dataset = TrajectoryDataset(character=character, alpha=0.9, window_size=window_size, is_test=False, batch_only_folder=batch_only_folder)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    sorthat_model = SortingHat().to(device)
    optimizer = torch.optim.AdamW(params=sorthat_model.parameters(), lr=0.00001)

    loss_weight = 1/(train_dataset.distribution)
    loss_weight/=loss_weight.sum()

    print(f"Now we are training our model on {character} with window size {window_size}")
    print(f"length of train_loader is {len(train_loader)}")
    print(f"length of test_loader is {len(test_loader)}")
    print(f"the label distribution is {train_dataset.distribution}")
    print(f"the label weights are {loss_weight}")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    loss_function = FocalLoss(alpha=loss_weight, gamma=2.0, reduction='sum')

    epochs = 40
    test_losses = []
    train_losses = []
    accs = []
    print("Start training...")
    max_acc = 0
    for epoch in tqdm(range(epochs)):
        losses = []
        for train_trajectory, train_data, train_label in train_loader:
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            train_trajectory = train_trajectory.to(device)
            optimizer.zero_grad()
            output = sorthat_model(train_trajectory, train_data)
            loss = loss_function(output, train_label)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        train_losses.append(np.mean(np.array(losses)))


        with torch.no_grad():
            correct = 0
            total_samples = len(test_loader.dataset)  # 测试集总样本数

            for test_trajectory, test_data, test_label in test_loader:
                test_data = test_data.to(device)
                test_label = test_label.to(device)
                test_trajectory = test_trajectory.to(device)
                output = sorthat_model(test_trajectory, test_data)  # 模型输出
                loss = loss_function(output, test_label)

                pred = torch.argmax(output, dim=1)
                correct += (pred == test_label).sum().item()

            # 计算准确率
            acc = correct / total_samples

            # 保存最佳模型
            if acc >= max_acc:
                max_acc = acc    
                torch.save(sorthat_model.state_dict(), f"./models/{batch_only_folder}/SortingHat_{character}_{window_size}.pt")

            test_losses.append(loss.item())  # 保存最后一个批次的测试损失
            accs.append(acc)

        scheduler.step()

    if result_path is not None:
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        ax[0].plot(train_losses, label='Train Loss')
        ax[0].set_title('Train Loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        ax[1].plot(test_losses, label='Test Loss')
        ax[1].set_title('Test Loss')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        ax[1].legend()

        ax[2].plot(accs, label='Accuracy')
        ax[2].set_title('Accuracy')
        ax[2].set_xlabel('Epochs')
        ax[2].set_ylabel('Accuracy')
        ax[2].legend()
        # set title
        fig.suptitle(f"Training SortingHat on {character} with window size {window_size}", fontsize=16)

        plt.tight_layout()
        # add grid
        plt.grid(True)

        check_path(f"{result_path}")
        plt.savefig(f"{result_path}/SortingHat_{character}_"+str(window_size)+".png")
        
    # save the model/only the parameters
    check_path(f"./models/{batch_only_folder}")
    # np.save(f"./results/{batch_only_folder}/{character}_accs.npy", np.array(accs))


def experiment_different_model_data_test(model_character = "Hogwarts", data_character = "Hogwarts", window_size = 3, batch_only_folder = "all_batches"):
    test_dataset = TrajectoryDataset(character=data_character, alpha=0.9, window_size=window_size, is_test=True, batch_only_folder=batch_only_folder)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    sorthat_model = SortingHat()
   
    sorthat_model.load_state_dict(torch.load(f"./models/{batch_only_folder}/SortingHat_{model_character}_{window_size}.pt", map_location=torch.device('cpu'), weights_only=True))

    loss_function = nn.CrossEntropyLoss()
    with torch.no_grad():
        correct = 0
        for test_trajectory, test_data, test_label in test_loader:
            output = sorthat_model(test_trajectory, test_data)
            loss = loss_function(output, test_label)
            
            output = softmax_layer(output)
            pred = torch.argmax(output)
            if pred == test_label:
                correct += 1
        acc = correct / len(test_loader)
        print(f"Accuracy of model trained on {model_character} and tested on {data_character} is {acc}")
    return acc


def check_results_of_different_windowsize(model_character = "Hogwarts", window_sizes = [2, 3, 4], batch_only_folder = "all_batches", zero_trajectory = False):
    accs = []
    for window_size in window_sizes:
        model = SortingHat()
        model.load_state_dict(torch.load(f"./models/{batch_only_folder}/SortingHat_{model_character}_{window_size}.pt", map_location=torch.device('cpu'), weights_only=True))
        print(f"model loaded...")

        test_dataset = TrajectoryDataset(character=model_character, alpha=0.9, window_size=window_size, is_test=True, batch_only_folder=batch_only_folder, zero_trajectory=zero_trajectory)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        loss_function = nn.CrossEntropyLoss()
        with torch.no_grad():
            correct = 0
            for test_trajectory, test_data, test_label in test_loader:
                output = model(test_trajectory, test_data)
                loss = loss_function(output, test_label)
                
                pred = torch.argmax(output)
                if pred == test_label:
                    correct += 1
            acc = correct / len(test_loader)
            accs.append(acc)
    # save the results
    check_path(f"./results/{batch_only_folder}")
    if zero_trajectory:
        np.save(f"./results/{batch_only_folder}/{model_character}_accs_zero_trajectory.npy", np.array(accs))
    else:
        np.save(f"./results/{batch_only_folder}/{model_character}_accs.npy", np.array(accs))


def character_ana(window_size=3):
    model = SortingHat()
   
    model.load_state_dict(torch.load(f"./models/all_batches/SortingHat_Hogwarts_{window_size}.pt",map_location=torch.device('cpu'), weights_only=True))
    print(f"model loaded...")

    Gryffindor_dataset = TrajectoryDataset(character='Gryffindor', alpha=0.9, window_size=window_size, is_test=False, batch_only_folder="all_batches")
    Hufflepuff_dataset = TrajectoryDataset(character='Hufflepuff', alpha=0.9, window_size=window_size, is_test=False, batch_only_folder="all_batches")
    Ravenclaw_dataset = TrajectoryDataset(character='Ravenclaw', alpha=0.9, window_size=window_size, is_test=False, batch_only_folder="all_batches")
    Slytherin_dataset = TrajectoryDataset(character='Slytherin', alpha=0.9, window_size=window_size, is_test=False, batch_only_folder="all_batches")
    print(f"data loaded...")

    dic_embeddings = {"Gryffindor": [], "Slytherin": [], "Hufflepuff": [], "Ravenclaw": []}

    # 提取 embeddings
    for dataset in [Gryffindor_dataset, Hufflepuff_dataset, Ravenclaw_dataset, Slytherin_dataset]:
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
    pca = PCA(n_components=2)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    pca_embeddings = pca.fit_transform(embeddings)
   

    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes = axes.flatten()
    # 定义坐标范围
    x_min, x_max = -3, 3
    y_min, y_max = -2, 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.vstack([xx.ravel(), yy.ravel()]).T

    for idx, character in enumerate(dic_embeddings.keys()):
        char_embeddings = np.array(dic_embeddings[character]).reshape(-1, embeddings.shape[1])
        pca_char_embeddings = pca.transform(char_embeddings)

        # 对该角色数据进行KDE拟合
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
        kde.fit(pca_char_embeddings)
        # 对固定坐标范围的网格点计算密度值
        log_dens = kde.score_samples(grid)
        dens = np.exp(log_dens).reshape(xx.shape)

        base_color = colors[character]
        custom_cmap = create_custom_cmap(base_color)
        ax = axes[idx]

        # 绘制多级等高线图
        max_density = dens.max()
        levels = np.linspace(max_density/80, max_density, 8)
        contour_lines = ax.contour(xx, yy, dens, levels=levels, colors=base_color, alpha=0.5)
        ax.contourf(xx, yy, dens, levels=levels, cmap=custom_cmap, alpha=0.8)
        if idx == 0:
            ax.set_ylabel("PCA-E2")
        ax.set_xlabel("PCA-e1")
        ax.set_facecolor('#f0f0f0')
        ax.grid(color='white', linestyle='-', linewidth=1)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

        # 固定坐标范围
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    # plt.set_xlabels("E1")
    plt.tight_layout()
    check_path(f"./results/all_batches/character_embeddings/PCA_2")
    plt.savefig(f"./results/all_batches/character_embeddings/PCA_2/Hogwarts_embedding_heatmap_window{window_size}_subplots_fixed_range.png")
    plt.show()


def plot_curves():
    plt.style.use('ggplot')  # 使用内置样式
    # 数据和配色
    x_labels = ["2", "3", "4"]
    characters = ["Hogwarts", "Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    # color_list = ["black", "red", "gold", "blue", "green"]
    
    # 将x_labels转换为数值坐标
    x = np.arange(len(x_labels))
    width = 0.15  # 每个柱子的宽度，可根据需要调节

    # fig = plt.figure(figsize=(10, 6))
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    # 标题和坐标轴标签

    # 绘制每个角色的柱状图
    for i, character in enumerate(characters):
        axes[i].set_title(f"{character}", fontsize=14, fontweight='bold')
        accs_zero = np.load(f"./results/all_batches/{character}_accs_zero_trajectory.npy")
        accs = np.load(f"./results/all_batches/{character}_accs.npy")
        
        # plot zero case with a lighter purple
        axes[i].plot(x, accs_zero, label=f"M off", color='pink', linewidth=2, marker='o', markersize=6)
        axes[i].plot(x, accs, label=f"M on", color=colors[character], linewidth=2, marker='o', markersize=6)
        axes[i].text(x[-1] + 0.1, accs[-1], s = '', fontsize=12, color=colors[character])
        axes[i].xlabel = "Window Size"
        axes[i].ylabel = "Test Accuracy"
    
        axes[i].set_xticks(x)  # 设置x轴刻度
        axes[i].set_xticklabels(x_labels, fontsize=12)  # 设置x轴刻度标签

        axes[i].legend(fontsize=12, loc='upper left')


    # plt.ylim(0, 0.6)  # 设置y轴范围
    # # 显示图例
    # plt.legend(fontsize=12, loc='upper left')
    plt.style.use('ggplot')  # 使用内置样式
    # 添加网格线
    plt.grid(axis='y', linestyle='--', linewidth=1, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"./results/accs.png", dpi=300)
    plt.show()





# 主函数
if __name__ == "__main__":
    pass
