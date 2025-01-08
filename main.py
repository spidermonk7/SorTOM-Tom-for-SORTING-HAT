from experiments  import *
from argparse import ArgumentParser
from data_generator import get_trajectory_txt, got_trajectory_embeded


# DONE
def train_all(window_sizes = [2, 3, 4], characters = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw", "Hogwarts"], device = device):
    # Check if data is ready
    prepear_data(window_sizes=window_sizes, characters=characters)
    print(f"Datasets are ready!!! Now start training the model....")
    for window_size in window_sizes:
        for character in characters:
            experiment2_Chara_Ment_Ss_A_prediction(window_size=window_size, character=character)

# DONE   
def Figure3_action_distribution():
    action_distribution_ana()

def prepear_data(window_sizes = [2, 3, 4], characters = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw", "Hogwarts"]):
    print(f"Start!!! Now generating Datasets....")
    for window_size in window_sizes:
        for character in characters:
            if not os.path.exists(f"./dataset/Trajectory/all_batches/window_{window_size}/{character}"):
                get_trajectory_txt(character, window_size, batch_only_folder="all_batches")
                get_trajectory_txt(character, window_size, is_test=True, batch_only_folder="all_batches")
            if not os.path.exists(f"./dataset/Trajectory/embedded/all_batches/window_{window_size}/{character}"):
                got_trajectory_embeded(window_size, character, batch_only_folder="all_batches")
                got_trajectory_embeded(window_size, character, is_test=True, batch_only_folder="all_batches")

    print(f"We are good to go!!!")

# 
def Figure4_mental_helps():
    character_list = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw", "Hogwarts"]
    window_sizes = [2, 3, 4]
    
    for character in character_list:
        for window_size in window_sizes:
            if not os.path.exists(f"./models/all_batches/SortingHat_{character}_{window_size}.pt"):
                print(f"Model for {character} with window size {window_size} is not ready, now start training the model....")
                train_all(window_sizes=[window_size], characters=[character], device=device)
    
    print(f"all model are prepeared!!!")      

    # Check if the model is ready: Both zero trajectory and non-zero trajectory
    for character in ["Ravenclaw", "Hufflepuff", "Gryffindor", "Slytherin", "Hogwarts"]:
        check_results_of_different_windowsize(model_character=character, window_sizes=[2, 3, 4], batch_only_folder="all_batches", zero_trajectory=False)
        check_results_of_different_windowsize(model_character=character, window_sizes=[2, 3, 4], batch_only_folder="all_batches", zero_trajectory=True)
    plot_curves()


def Figure5_meta_learning():
    data_dict = {
        2: np.zeros((5, 4)),
        3: np.zeros((5, 4)),
    }
    for window_size in [2, 3]:
        print(f"window size is {window_size}")
        for id, model_character in enumerate(["Hogwarts", "Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]):
            for jd, data_character in enumerate(["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]):
                
                acc = experiment_different_model_data_test(model_character=model_character, data_character=data_character, window_size=window_size, batch_only_folder="all_batches")
                data_dict[window_size][id][jd] = acc

    
    plt.style.use('ggplot')
    test_sets = ["G", "S", "H", "R"]
    window_sizes = [2, 3]

    # 训练集行顺序及对应原数据行索引
    train_datasets = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]
    row_order = [1, 4, 2, 3]  # Gryffindor=1, Slytherin=4, Hufflepuff=2, Ravenclaw=3, Hogwarts=0

   
    for row_idx, train_set_name in enumerate(train_datasets):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True, sharex=True)
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
            ax.set_xticks(ticks = [0, 1, 2, 3], minor=False)
            ax.set_xticklabels(test_sets, rotation=0)
        # plt.suptitle(f"Window Size = {wsize}", fontsize=16)
        check_path(f"./results/all_batches/meta_learning")
        fig.patch.set_alpha(0)
        plt.tight_layout()
        plt.savefig(f"./results/all_batches/meta_learning/{train_set_name}_lineplots_with_Hogwarts.png", dpi=300)
        plt.show()
        plt.close()

  

def Figure6_character_embedding():
    for window_size in [2, 3, 4]:
        character_ana(window_size=window_size)
    



if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--exp_id", type=int, default=0)
    args = args.parse_args()


    print(f"Before starting the experiment, please make sure the data is ready!!!")
    prepear_data()


    if args.exp_id == 1:
        Figure3_action_distribution()

    elif args.exp_id == 2:
        Figure4_mental_helps()

    elif args.exp_id == 3:
        Figure5_meta_learning()

    elif args.exp_id == 4:
        Figure6_character_embedding()

    elif args.exp_id == 0:
        train_all()
    else:
        print("ERROR! The experiment id should be in [0, 1, 2, 3, 4], please follow the instruction in the README file.")

