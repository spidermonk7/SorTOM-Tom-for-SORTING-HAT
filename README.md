# SorTOM-Tom-for-SORTING-HAT
![6651733464215_ pic](https://github.com/user-attachments/assets/f81a75d9-9b7b-49ec-882d-6aa9cc8ec49d)


## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Experiments](#experiments)
- [Finding Your House](#finding-your-house)


## Project Overview

This repository implements a series of experiments to study the sorting hat mechanism, where the task is to predict a character's house at Hogwarts based on trajectories and mental representations over time. The experiments involve the use of machine learning models and embeddings to train and test sorting hat predictions, visualize the action distributions, analyze meta-learning capabilities, and more. 

For more details, please check our [report](SortingHat_Report.pdf). 


1. **Action Distribution Analysis**: Analysis of the distribution of actions in the dataset.
2. **Demonstrating the effectiveness of mental state embedding.**: Examining the role of mental states in the sorting process, with results for different window sizes.
3. **Meta-Learning**: Investigating the ability of the model to generalize from different training sets.
4. **Character Embeddings**: Visualizing and analyzing character embeddings for better understanding of the model's behavior.

## Installation

To run the experiments, you will need to set up the environment and install the required dependencies. Here's how you can do it:

1. Clone the repository:
    ```bash
    git clone https://github.com/spidermonk7/SorTOM-Tom-for-SORTING-HAT.git
    cd SorTOM-Tom-for-SORTING-HAT
    ```

2. Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure that the necessary datasets are available. The data generation process is handled by the code, but the data must be prepared before running the experiments.


## Data & Model Preparation

Before running the experiments, make sure the necessary datasets are ready. This is handled by the `prepear_data()` function, which will generate the required data files for training and testing. You can also manually ensure the data is available in the correct directories.

Run the following command to prepare the data:

```bash
python main.py --exp_id -1
```

This will trigger data generation.


## Experiments

**Attention!**
Because the randomness of training process, since we offered our trained model, to got exactly the same result as we did, we strongly recoomend you to run experiments id 1-4 before you try to re-train the models by running [exp_id 0]

You can run different experiments by specifying the `--exp_id` argument when executing the `main.py` script. The available experiments are:


1. **Train the model on all window sizes and characters**  
    Exp ID: `0`  
    This will train the model for all window sizes (2, 3, 4) and characters (Gryffindor, Slytherin, Hufflepuff, Ravenclaw, Hogwarts).

    ```bash
    python main.py --exp_id 0
    ```

2. **Action Distribution Analysis (Figure 3)**  
    Exp ID: `1`  
    This experiment analyzes the distribution of actions in the dataset.

    ```bash
    python main.py --exp_id 1
    ```
    ![action_distribution](https://github.com/user-attachments/assets/e9070189-6bb3-49bd-825e-461b37a66ab7)

    

3. **Mental Health Analysis (Figure 4)**  
    Exp ID: `2`  
    This experiment checks the impact of mental states on model performance for different window sizes and characters.

    ```bash
    python main.py --exp_id 2
    ```
    ![accs](https://github.com/user-attachments/assets/4cdd831d-85a5-4056-b9f7-862d4543f2fa)

    

4. **Meta-Learning Analysis (Figure 5)**  
    Exp ID: `3`  
    This experiment investigates the meta-learning performance of the model, specifically how well it generalizes to unseen data.

    ```bash
    python main.py --exp_id 3
    ```
    ![meta_learning](https://github.com/user-attachments/assets/f23f2413-a13d-4be9-8ee6-bdddd17e183e)


5. **Character Embedding Analysis (Figure 6)**  
    Exp ID: `4`  
    This experiment analyzes character embeddings to better understand the model's learned representations.

    ```bash
    python main.py --exp_id 4
    ```
    ![mental_embedding_distribution](https://github.com/user-attachments/assets/3d7210ac-e7ef-4098-b4d8-539331200568)


   
## Finding Your House

We present a complete program for you to use our Sorting Hat model. Simply run the following command:

```bash
python sorting_hat.py
```
Then, follow the instructions and play for about 15 rounds. In each round, you will take 6 steps. Afterward, we will perform an analysis based on your own state-action sequence (ALBUS).

In just a few seconds, you'll see your sorting result, which will look like this:
![Hogwarts_KL_divergence](https://github.com/user-attachments/assets/ddc84ee4-baed-427a-9a44-82d695a4edf5)

This chart shows the probability of you belonging to each house based on models with different window sizes.


## License

This project is licensed under the HOGWARTS License ~
