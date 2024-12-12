from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from model import StateEmbedding, SortingHat, CharaNet, MentNet, SimpleMLP, SimpleRNN, SimpleLSTM, SimpleGRU

from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from utils import *




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is {device}")

label_dic = {"Explore": 0, "Help": 1, "Refuse": 2, "Betray": 3, "Fight": 4, "Escape": 5}


class TextDataset(Dataset):
    def __init__(self, character="Slytherin", alpha=[0.9], model_name='bert-base-uncased', max_length=230):
        self.character = character
        self.alpha = alpha
        self.max_length = max_length
        self.model_name = model_name
        
        # 加载预先保存的 BERT 嵌入数据
        embeddings_path = f"./dataset/character_wise/embedded/{self.character}_all_embedded.npy"
        self.embeddings = np.load(embeddings_path)  # 加载嵌入数据

        # 加载标签
        labels_path = f"./dataset/character_wise/embedded/{self.character}_labels.npy"
        self.labels = np.load(labels_path)  # 加载标签数据

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedded_state = torch.tensor(self.embeddings[idx])  # 加载预处理的嵌入
        label = torch.tensor(self.labels[idx])  # 获取标签
        return embedded_state, label


### Modified by Yichen

class TrajectoryDataset(Dataset):
    def __init__(self, character="Slytherin", alpha=[0.9], model_name='bert-base-uncased', max_length=230, window_size = 2, is_test=False):
        self.character = character
        self.alpha = alpha
        self.max_length = max_length
        self.model_name = model_name
        
        # 加载预先保存的 BERT 嵌入数据
        if is_test:
            embeddings_path = f"./dataset/Trajectory/embedded/window_{window_size}/{self.character}_all_embedded-test.npy"
            trajectory_embeddings_path = f"./dataset/Trajectory/embedded/window_{window_size}/{self.character}_all_trajectory_embedded-test.npy"
            self.embeddings = np.load(embeddings_path)  # 加载嵌入数据
            self.traj_embeddings = np.load(trajectory_embeddings_path)  # 加载嵌入数据

            # 加载标签
            labels_path = f"./dataset/Trajectory/embedded/window_{window_size}/{self.character}_labels-test.npy"
            self.labels = np.load(labels_path)  # 加载标签数据

        else:
            embeddings_path = f"./dataset/Trajectory/embedded/window_{window_size}/{self.character}_all_embedded-test.npy"
            trajectory_embeddings_path = f"./dataset/Trajectory/embedded/window_{window_size}/{self.character}_all_trajectory_embedded-test.npy"
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




def experiment1_Simple_S_A_prediction(model_name = "SimpleMLP", input_size=230, output_size=6, hidden_size=128):
    # 训练函数
    def train(model, train_loader, optimizer, loss_function, epoch):
        model.train()
        running_loss = 0.0
        for train_data, train_label in tqdm(train_loader):
            train_data = train_data.permute(1, 0, 2)
            optimizer.zero_grad()
            output = model(train_data)

            loss = loss_function(output, train_label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}: Train Loss is {running_loss/len(train_loader)}")
    
        return running_loss/len(train_loader)


    # 验证函数
    def validate(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for test_data, test_label in test_loader:
                test_data = test_data.permute(1, 0, 2)
                output = model(test_data)
                _, predicted = torch.max(output, 1)
                total += test_label.size(0)
                correct += (predicted == test_label).sum().item()
                loss = loss_function(output, test_label)

        return correct / total,  loss.item()


    assert model_name in ["SimpleMLP", "SimpleRNN", "SimpleLSTM", "SimpleGRU"], "model should be one of SimpleMLP, SimpleRNN, SimpleLSTM, SimpleGRU"

    # switch the str model name to the class
    model = eval(model_name)(input_size=input_size, output_size=output_size, hidden_size=hidden_size)

    text_dataset = TextDataset(character='Hogwarts', alpha=0.9)
    print(text_dataset.shape)
    # split text dataset into train and test
    train_size = int(0.9 * len(text_dataset))
    test_size = len(text_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(text_dataset, [train_size, test_size])
    print(f"length of train set: {len(train_dataset)}, test set: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

     # 设置优化器和损失函数
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    loss_function = nn.CrossEntropyLoss()
    
    # 训练和验证
    test_losses = []
    train_losses = []
    accs = []
    epochs = 20
    print("Start training...")
    for epoch in range(epochs):
        # 训练
        train_loss = train(model, train_loader, optimizer, loss_function, epoch=epoch)
        train_losses.append(train_loss)
        # 验证
        if epoch % 2 == 0:
            accuracy, test_loss = validate(model, test_loader)
            test_losses.append(test_loss)
            accs.append(accuracy)
            print(f"Epoch {epoch+1}: Validation Accuracy is {accuracy}")

    
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

    plt.tight_layout()
    # add grid 
    plt.grid(True)
    plt.show()


### Modified by Yichen
def experiment2_Chara_Ment_Ss_A_prediction(window_size = 3):
    assert window_size in [2, 3, 4, 5, 6] , "window_size should be one of 2, 3, 4"

    test_dataset = TrajectoryDataset(character='Hogwarts', alpha=0.9, window_size=window_size, is_test=True)
    train_dataset = TrajectoryDataset(character='Hogwarts', alpha=0.9, window_size=window_size, is_test=False)


    # split train and test
    # dataset = TrajectoryDataset(character='Hogwarts', alpha=0.9, window_size=window_size, is_test=False)
    # train_size = int(0.9*len(dataset))
    # test_size = len(dataset)-train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    sorthat_model = SortingHat(window_size = window_size, chara_embedding_size = 200, hidden_size=128, output_size=6, len_state_embedding = 200)

    optimizer = torch.optim.Adam(params=sorthat_model.parameters(), lr=0.00001)
    loss_function = nn.CrossEntropyLoss()

    epochs = 100
    test_losses = []
    train_losses = []
    accs = []
    print("Start training...")
    for epoch in tqdm(range(epochs)):
        losses = []
        for train_trajectory, train_data, train_label in train_loader:
            optimizer.zero_grad()
            output = sorthat_model(train_trajectory, train_data)
            loss = loss_function(output, train_label)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        train_losses.append(np.mean(np.array(losses)))


        with torch.no_grad():
            correct = 0
            for test_trajectory, test_data, test_label in test_loader:
                output = sorthat_model(test_trajectory, test_data)
                loss = loss_function(output, test_label)
                
                pred = torch.argmax(output)
                if pred == test_label:
                    correct += 1
            test_losses.append(loss.item())

            acc = correct / len(test_loader)
            accs.append(acc)
        #     print(f"Epoch {epoch+1}: Validation Accuracy is {acc}")
        # print(f"Epoch {epoch+1}: Train Loss is {train_losses[-1]}")


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

    plt.tight_layout()
    # add grid
    plt.grid(True)


    check_path("./results")
    plt.savefig("./results/SortingHat_"+str(window_size)+".png")
    


    # save the model/only the parameters
    check_path("./models")
    torch.save(sorthat_model.state_dict(), f"./models/SortingHat_{window_size}.pt")
  

# 定义基础颜色
colors = {
    'Gryffindor': '#FF6F61', 
    'Hufflepuff': '#FFD700', 
    'Ravenclaw': '#6495ED', 
    'Slytherin': '#3CB371'
}

def create_custom_cmap(base_color):
    """
    根据基础颜色创建渐变 colormap，从深色到白色
    """
    from matplotlib.colors import to_rgba
    base_rgb = to_rgba(base_color)  # 转换为 RGBA 格式
    white_rgb = (1, 1, 1, 1)  # 白色
    return LinearSegmentedColormap.from_list(f"{base_color}_cmap", [white_rgb, base_rgb])


def character_ana(window_size=3):
    # 加载预训练模型
    model = SortingHat(window_size=window_size, chara_embedding_size=200, hidden_size=128, output_size=6, len_state_embedding=200)
    model.load_state_dict(torch.load(f"./models/SortingHat_{window_size}.pt"))
    print(f"model loaded...")
    
    # 加载数据集
    Gryffindor_dataset = TrajectoryDataset(character='Gryffindor', alpha=0.9, window_size=window_size)
    Hufflepuff_dataset = TrajectoryDataset(character='Hufflepuff', alpha=0.9, window_size=window_size)
    Ravenclaw_dataset = TrajectoryDataset(character='Ravenclaw', alpha=0.9, window_size=window_size)
    Slytherin_dataset = TrajectoryDataset(character='Slytherin', alpha=0.9, window_size=window_size)
    print(f"data loaded...")
    
    dic_embeddings = {"Gryffindor": [], "Hufflepuff": [], "Ravenclaw": [], "Slytherin": []}

    # 提取每个数据集的 embeddings
    for dataset in [Gryffindor_dataset, Hufflepuff_dataset, Ravenclaw_dataset, Slytherin_dataset]:
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for trajectory, data, label in tqdm(loader, desc=f"character {dataset.character}"):
                output = model(trajectory, data)
                character_embed = model.chara
                dic_embeddings[dataset.character].append(character_embed.squeeze(0).numpy())
    
    # 使用 PCA 将 embeddings 降维到 2D
    embeddings = []
    labels = []
    for key in dic_embeddings:
        embeddings.extend(dic_embeddings[key])
        labels += [key] * len(dic_embeddings[key])
    
    embeddings = np.array(embeddings)
    print(f"shape of embeddings is {embeddings.shape}")
    
    # PCA降维
    pca = PCA(n_components=2)
    embeddings = embeddings.reshape(-1, 200)
    print(f"shape of embeddings is {embeddings.shape}")
    pca_embeddings = pca.fit_transform(embeddings)
    print(f"length of pca_embeddings is {len(pca_embeddings)}")

    # 创建图形和子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 创建四个子图
    axes = axes.flatten()

    # 定义每个学院的颜色和索引
    for idx, character in enumerate(dic_embeddings.keys()):
        # 获取当前学院的 PCA 降维结果
        char_embeddings = np.array(dic_embeddings[character]).reshape(-1, 200)
        pca_char_embeddings = pca.transform(char_embeddings)
        
        # 核密度估计
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
        kde.fit(pca_char_embeddings)

        # 生成网格用于评估密度
        x_min, x_max = pca_char_embeddings[:, 0].min() - 1, pca_char_embeddings[:, 0].max() + 1
        y_min, y_max = pca_char_embeddings[:, 1].min() - 1, pca_char_embeddings[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid = np.vstack([xx.ravel(), yy.ravel()]).T
        
        # 计算网格点的密度值
        log_dens = kde.score_samples(grid)
        dens = np.exp(log_dens).reshape(xx.shape)

        # 获取基础颜色并创建渐变色图
        base_color = colors[character]
        custom_cmap = create_custom_cmap(base_color)

        # 绘制热力图
        ax = axes[idx]  # 当前子图
        ax.contourf(xx, yy, dens, 20, cmap=custom_cmap, alpha=0.8)  # 使用自定义的渐变色图显示密度
        ax.scatter(pca_char_embeddings[:, 0], pca_char_embeddings[:, 1], color=base_color, label=character, s=10)


        # 添加标题、坐标轴标签和网格
        ax.set_title(f"{character} Embedding Heatmap")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.grid(True)
        ax.legend(loc='upper right')

   

    # 保存热力图
    plt.tight_layout()  # 调整布局，避免重叠

    check_path("./results")

    plt.savefig(f"./results/Hogwarts_embedding_heatmap_window{window_size}_subplots.png")
    plt.show()  # 显示图形

# 主函数
if __name__ == "__main__":
    # print(f"====Experiment 1: Simple State and Action Prediction====")
    # experiment1_Simple_S_A_prediction(model_name = "SimpleMLP", input_size=230, output_size=6, hidden_size=128)
    print(f"====Experiment 2: Character, Ment and State Action Prediction====")
    for i in range(2, 7):
        print("########### Window Size = "+str(i)+" ##############")
        experiment2_Chara_Ment_Ss_A_prediction(window_size=i)
        character_ana(window_size=i)


