import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

np.random.seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")



colors = {'Gryffindor': '#FF6F61',  # 柔和红
            'Hufflepuff': '#FFD700',  # 柔和黄
            'Ravenclaw': '#6495ED',  # 柔和蓝
            'Slytherin': '#3CB371',
            'Tester': '#8FBC8F',
            'Hogwarts': 'purple'
            }  # 柔和绿

def check_path(path = "./dataset"):
    if not os.path.exists(path):
        os.makedirs(path)


def kl_divergence(kde_p, kde_q, n_samples=10000):
    # 从 p(x) 的KDE中采样
    samples = kde_p.sample(n_samples)
    # 计算 p(x_j)
    log_p = kde_p.score_samples(samples)  # log_p = log p(x_j)
    # 计算 q(x_j)
    log_q = kde_q.score_samples(samples)  # log_q = log q(x_j)
    
    # 计算 KL：mean over samples of log(p(x)/q(x)) = mean(log_p - log_q)
    kl = np.mean(log_p - log_q)
    return kl


def overlap_coefficient_2d(kde_p, kde_q, 
                           x_min=-2.5, x_max=2.5, 
                           y_min=-1.5, y_max=1.5, 
                           grid_size=200):
    """
    使用数值积分近似法计算两个2D KDE分布之间的Overlap Coefficient。
    overlap = ∫ min(p(x), q(x)) dx
    
    参数：
    kde_p, kde_q: 已使用 KernelDensity 拟合好的KDE模型 (2D数据)
    x_min, x_max, y_min, y_max: 积分边界范围
    grid_size: 网格划分数目，每维200表示 200x200 的网格点
    
    返回值：
    overlap_coeff: 一个介于[0,1]的数值，表示两个分布的重叠程度。
    """
    # 建立网格
    x_lin = np.linspace(x_min, x_max, grid_size)
    y_lin = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_lin, y_lin)
    
    # 将网格点打平，以便一次性批量计算 score_samples
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # 计算 p(x) 和 q(x)
    log_p = kde_p.score_samples(grid_points)  # log p(x)
    log_q = kde_q.score_samples(grid_points)  # log q(x)
    p_vals = np.exp(log_p)
    q_vals = np.exp(log_q)
    
    # 对每个点取 min(p, q)
    min_vals = np.minimum(p_vals, q_vals)
    
    # 网格单元面积
    dx = (x_max - x_min) / (grid_size - 1)
    dy = (y_max - y_min) / (grid_size - 1)
    cell_area = dx * dy
    
    # 对所有点进行求和并乘以 cell_area 近似积分
    overlap_coeff = np.sum(min_vals) * cell_area
    
    return overlap_coeff



def create_custom_cmap(base_color):
    # 可以根据base_color创建渐变色图，这里仅做示意，可自行改进
    return LinearSegmentedColormap.from_list("custom_cmap", ["white", base_color])

def plot_ellipsoid(ax, center, cov, n_std=1.0, color='blue', alpha=0.1):
    # 绘制椭球面的辅助函数，根据协方差与均值拟合椭球
    # 本例参考了之前的代码，可根据需要微调分辨率
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    eigvals, eigvecs = np.linalg.eigh(cov)
    radii = n_std * np.sqrt(eigvals)
    T = (eigvecs * radii).T
    ellipsoid = np.dot(np.stack([x.flatten(), y.flatten(), z.flatten()]).T, T) + center
    X = ellipsoid[:,0].reshape(x.shape)
    Y = ellipsoid[:,1].reshape(x.shape)
    Z = ellipsoid[:,2].reshape(x.shape)
    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=alpha, color=color, linewidth=0, shade=True)



if __name__ == '__main__':

    plt.style.use('ggplot')

    colors = {
        'Gryffindor': '#FF6F61', # 柔和红
        'Hufflepuff': '#FFD700', # 柔和黄
        'Ravenclaw':  '#6495ED', # 柔和蓝
        'Slytherin':  '#3CB371', # 柔和绿
        'Hogwarts':   'purple'
    }

    test_sets = ["G", "S", "H", "R"]
    window_sizes = [2, 3, 4, 5]

    # 训练集行顺序及对应原数据行索引
    train_datasets = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]
    row_order = [1, 4, 2, 3]  # Gryffindor=1, Slytherin=4, Hufflepuff=2, Ravenclaw=3, Hogwarts=0

    # 原始数据（从之前的数据中摘取）
    # 行顺序：0-Hogwarts,1-Gryffindor,2-Hufflepuff,3-Ravenclaw,4-Slytherin
    # 列顺序：测试集 [Gryffindor, Slytherin, Hufflepuff, Ravenclaw]

    "[Hog, Gry, Huff, Rav, Sly]"

    data_w2_orig = np.array([
        [0.11,    0.25, 0.25,    0.25],
        [0.25,    0.25, 0.0,    0.0   ],
        [0.1667,    0.0833, 0.4167,    0.3333],
        [0.1667,    0.0833, 0.3333,    0.4167],
        [0.5,    0.3333,    0.0833,    0.1667   ]
    ])
    data_w3_orig = np.array([
        [0.111,    0.6667, 0.333,    0.333],
        [0.111,    0.111, 0.222,    0.111],
        [0.222,    0.111,   0.444,    0.444],
        [0.222,    0.333,    0.222,   0.444],
        [0.333,    0.555,   0.0,    0.3333]
    ])
    data_w4_orig = np.array([
        [0.1667,    0.5, 0.1667,    0.5],
        [0.3333,    0.0, 0.3333,    0.3333],
        [0.6667,    0.0, 0.5,    0.3333],
        [0.1667, 0.1667, 0.0, 0.3333],
        [0.1667, 0.5, 0.0,    0.3333   ]
    ])
    data_w5_orig = np.array([
        [0.6667,    0.6667, 0.0,    1.0],
        [0.3333,    0.3333, 0.0,    0.3333   ],
        [0.0,    0.0, 0.6667,    0.0],
        [0.6667,    0.3333, 0.0,    1.0],
        [0.3333,    0.3333, 0.0,    1.0   ]
    ])
    data_dict = {
        2: data_w2_orig,
        3: data_w3_orig,
        4: data_w4_orig,
        5: data_w5_orig,
    }

   
    for row_idx, train_set_name in enumerate(train_datasets):
        fig, axes = plt.subplots(1, 5, figsize=(20, 3), sharey=True, sharex=True)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        for col_idx, wsize in enumerate(window_sizes):
            ax = axes[col_idx]
            data = data_dict[wsize]
            
            hogwarts_y = data[0]  # Hogwarts 行
            train_y = data[row_order[row_idx]]  # 当前训练集行

            # 绘制当前训练集结果
            ax.plot(test_sets, train_y, marker='o', color=colors[train_set_name], label=train_set_name)

            # 绘制Hogwarts结果
            ax.plot(test_sets, hogwarts_y, marker='o', color=colors['Hogwarts'], label='Hogwarts')
       
            ax.grid(True)

            # 只在第一列显示Y轴标签
            if col_idx == 0:
                ax.set_ylabel("Accuracy", fontsize=12)
            # X轴标签水平显示（不旋转）
            ax.set_xticklabels(test_sets, rotation=0)
        # plt.suptitle(f"Window Size = {wsize}", fontsize=16)
        plt.savefig(f"./results/{train_set_name}_lineplots_with_Hogwarts.png", dpi=300)
        plt.tight_layout()
        plt.close()
    # 在左上角的第一个子图显示图例

    