import torch
import torch.nn as nn
import json
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import numpy as np


""""
This is a file for building the naive models, which:
(1) Takes the state as input: 2d input
(2) Outputs the action: 1d output
"""


# set the random seed
torch.manual_seed(1)

"""
Character AnA Model: Try to valid that the characters can be distinguished by the actions

The model should be like:
(1) Input: 2d state and 1d action
(2) Output: the character
"""



class CharacterAnAModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=6):
        super(CharacterAnAModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.hidden_out = None
        
    def forward(self, x):
        x = self.fc1(x)
        self.hidden_out = x
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
def run_character_ana_model():
    model = CharacterAnAModel(3, 4)  # 输出为4个类别，适配CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50], gamma=0.5)

    # 加载并打乱数据集
    dataset = json.load(open("data/jack_all_characters_v3.json", "r"))
    keys = list(dataset.keys())
    random.shuffle(keys)
    dataset = {key: dataset[key] for key in keys}

    # 划分训练集和测试集
    train_dataset = {}
    test_dataset = {}
    for id in dataset:
        if int(id) % 10 == 0:
            test_dataset[id] = dataset[id]
        else:
            train_dataset[id] = dataset[id]

    accs = []
    test_losses = []
    for epoch in range(100):
        # 训练
        model.train()
        for id in train_dataset:
            item = train_dataset[id]
            state = torch.tensor(item["state"], dtype=torch.float32)
            action = torch.tensor([item["action"]], dtype=torch.float32)  # 改为 long 类型
            character = torch.tensor(item["character"], dtype=torch.long)  # character 用于拼接
            optimizer.zero_grad()
            # print(f"shape of state: {state.shape}, shape of action: {action.shape}")
            input_x = torch.cat([state, action], dim=0)
            output = model(input_x)
            loss = criterion(output, character)  # action 扩展 batch 维度
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

        # 测试
        if epoch % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                losses = []
                for id in test_dataset:
                    item = test_dataset[id]
                    state = torch.tensor(item["state"], dtype=torch.float32)
                    action = torch.tensor([item["action"]], dtype=torch.float32)  # 改为 long 类型
                    character = torch.tensor(item["character"], dtype=torch.long)

                    input_x = torch.cat([state, action], dim=0)
                    output = model(input_x)
                    losses.append(criterion(output, character).item())

                    # 预测标签
                    predicted = torch.argmax(output)
                    correct += (predicted == action).sum().item()
                    total += 1

                # 记录测试集损失和准确率
                avg_loss = sum(losses) / len(losses)
                test_losses.append(avg_loss)
                accuracy = correct / total
                accs.append(accuracy)
                print(f"Epoch {epoch + 1} - Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # 调整学习率
        # scheduler.step()

    # 保存模型
    torch.save(model.state_dict(), "model/character_ana_model_v3.pth")

    # 绘制测试集损失和准确率
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(test_losses, label="Test Losses")
    plt.title("Test Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accs, label="Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import seaborn as sns
import torch
import json
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from matplotlib.patches import Polygon
import seaborn as sns
import torch
import json
from tqdm import tqdm
import numpy as np

def ana_model():
    model_path = "model/character_ana_model_v3.pth"
    model = CharacterAnAModel(3, 4)
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
    dataset = json.load(open("data/jack_all_characters_v3.json", "r"))

    # 存储不同 character 的隐藏状态
    hidden_states = {'Gryffindor': [], 'Hufflepuff': [], 'Ravenclaw': [], 'Slytherin': []}
    id_character_map = {0: 'Gryffindor', 1: 'Slytherin', 2: 'Hufflepuff', 3: 'Ravenclaw'}

    # 处理数据
    for id in tqdm(dataset):
        item = dataset[id]
        state = torch.tensor(item["state"], dtype=torch.float32)
        action = torch.tensor([item["action"]], dtype=torch.float32)
        character = item["character"]
        input_x = torch.cat([state, action], dim=0)
        output = model(input_x)
        hidden_states[id_character_map[character]].append(model.hidden_out.detach().numpy())

    # 设置背景样式
    sns.set_style('darkgrid')

    # 创建绘图
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # 设置坐标轴在中间
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # 调整颜色
    colors = {'Gryffindor': '#FF6F61',  # 柔和红
              'Hufflepuff': '#FFD700',  # 柔和黄
              'Ravenclaw': '#6495ED',  # 柔和蓝
              'Slytherin': '#3CB371'}  # 柔和绿

    # 绘制每个 character 的点和分布区域
    for character in hidden_states:
        hidden_states[character] = np.array(hidden_states[character])
        
        # 使用 PCA 将隐藏状态降到二维
        pca = PCA(n_components=2)
        pca_hidden_states = pca.fit_transform(hidden_states[character])
        # 绘制散点
        plt.scatter(pca_hidden_states[:, 0], pca_hidden_states[:, 1], alpha=0.7, color=colors[character])

        # 绘制分布区域（平滑的边界曲线）
        try:
            hull = ConvexHull(pca_hidden_states)
            hull_points = pca_hidden_states[hull.vertices]

            # 用样条插值平滑边界
            tck, u = splprep([hull_points[:, 0], hull_points[:, 1]], s=0, per=True)
            spline = splev(np.linspace(0, 1, 500), tck)
            # 填充区域
            plt.fill(spline[0], spline[1], color=colors[character], alpha=0.2, zorder=1)
            # 绘制边界曲线
            plt.plot(spline[0], spline[1], color=colors[character], linestyle='--', linewidth=1.5, zorder=2)

        except Exception as e:
            print(f"Unable to compute ConvexHull or spline for {character}: {e}")

    # 设置标题和网格
    plt.title("Hidden States PCA Visualization", fontsize=16)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.show()
    # 显示图形
    plt.savefig("figures/hidden_states_pca.png")

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import colormaps

def data_ana():
    # 加载数据
    data = json.load(open("data/jack_all_characters_v3.json", "r"))

    # 颜色映射，使用内置的 colormap
    colormaps_list = ['Reds', 'Greens', 'YlOrBr', 'Blues']  # 红、绿、黄、蓝
    id_character_map = {0: 'Gryffindor', 1: 'Slytherin', 2: 'Hufflepuff', 3: 'Ravenclaw'}

    # 初始化每个 character 的数据存储
    character_data = {i: {'state': [], 'action': []} for i in range(4)}

    # 提取满足条件的数据 (action=1 且 state 在指定范围内)
    for id in data:
        item = data[id]
        character = item["character"]
        state = item["state"]
        action = item["action"]

        # 过滤条件
        if action == 1 and 3 <= state[0] <= 24 and 1 <= state[1] <= 10:
            character_data[character]['state'].append(state)

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()  # 展平方便索引

    # 绘制每个 character 的密度分布
    for character_id in range(4):
        ax = axes[character_id]
        states = np.array(character_data[character_id]['state'])

        if len(states) > 0:  # 检查是否有符合条件的点
            # 提取 state 的两个维度
            x = states[:, 0]
            y = states[:, 1]

            # 计算密度估计
            kde = gaussian_kde([x, y])
            xmin, xmax = 0, 16
            ymin, ymax = 0, 16
            X, Y = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

            # 获取对应的 colormap
            cmap = colormaps[colormaps_list[character_id]]

            # 绘制密度热力图和等高线
            ax.contourf(X, Y, Z, levels=10, cmap=cmap, alpha=0.5)
            ax.contour(X, Y, Z, levels=10, colors=cmap(np.linspace(0.2, 0.8, 10)), linewidths=1.0)

        # 设置子图标题和坐标轴
        ax.set_title(f"{id_character_map[character_id]}", fontsize=14)
        ax.set_xlim(0., 16)
        ax.set_ylim(0., 8)
        ax.set_xlabel("State Dimension 1")
        ax.set_ylabel("State Dimension 2")
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # 设置全局标题
    fig.suptitle("State-Action Density Heatmap for Action=1", fontsize=16)
    plt.savefig("figures/state_action_density_heatmap.png")
    # 显示图形
    plt.show()


def scatter_action_1_subplots():
    # 加载数据
    data = json.load(open("data/jack_all_characters_v3.json", "r"))

    # 颜色映射，红、绿、黄、蓝
    colors = ['#FF6F61', '#3CB371', '#FFD700', '#6495ED']
    id_character_map = {0: 'Gryffindor', 1: 'Slytherin', 2: 'Hufflepuff', 3: 'Ravenclaw'}

    # 初始化每个 character 的数据存储
    character_data = {i: {'state': [], 'action': []} for i in range(4)}

    # 提取满足条件的数据 (action=1 且 state 在指定范围内)
    for id in data:
        item = data[id]
        character = item["character"]
        state = item["state"]
        action = item["action"]

        # 过滤条件
        if action == 1 and 3 <= state[0] <= 24 and 1 <= state[1] <= 10:
            character_data[character]['state'].append(state)

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()  # 将子图展平方便索引

    # 绘制每个 character 的散点图
    for character_id in range(4):
        ax = axes[character_id]
        states = np.array(character_data[character_id]['state'])

        if len(states) > 0:  # 检查是否有符合条件的点
            # 提取 state 的两个维度
            x = states[:, 0]
            y = states[:, 1]

            # 绘制散点图
            ax.scatter(
                x, y, color=colors[character_id], alpha=0.7, edgecolor='k', s=50
            )

        # 设置子图标题和坐标轴
        ax.set_title(f"{id_character_map[character_id]}", fontsize=14)
        ax.set_xlim(0., 16)  # 指定 x 轴范围
        ax.set_ylim(0., 8)   # 指定 y 轴范围
        ax.set_xlabel("State Dimension 1")
        ax.set_ylabel("State Dimension 2")
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # 设置全局标题
    fig.suptitle("Scatter Plot of Action=1 for All Characters", fontsize=16)

    # 显示图形
    plt.show()



if __name__ == '__main__':
    # data_ana()
    # scatter_action_1_subplots()
    # run_character_ana_model()
    ana_model()
    # data_ana()