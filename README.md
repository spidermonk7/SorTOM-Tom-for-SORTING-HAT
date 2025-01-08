# SorTOM-Tom-for-SORTING-HAT
![6651733464215_ pic](https://github.com/user-attachments/assets/f81a75d9-9b7b-49ec-882d-6aa9cc8ec49d)


This repository implements a series of experiments to study the sorting hat mechanism, where the task is to predict a character's house at Hogwarts based on trajectories and mental representations over time. The experiments involve the use of machine learning models and embeddings to train and test sorting hat predictions, visualize the action distributions, analyze meta-learning capabilities, and more.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Experiments](#experiments)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [License](#license)

## Project Overview

The project is the Term Project of PKU-CORE:24-autumn of team Co-rejectors. It is designed to predict the behaviour of agents of characters (e.g., Gryffindor, Slytherin, Hufflepuff, Ravenclaw) based on their state-action sequence(namely, ALBUS) on text based D&D games(namely, GELLERT). The model is trained on different window sizes and evaluated on various tasks:

1. **Action Distribution Analysis**: Analysis of the distribution of actions in the dataset.
2. **Demonstrating the effectiveness of mental state embedding.**: Examining the role of mental states in the sorting process, with results for different window sizes.
3. **Meta-Learning**: Investigating the ability of the model to generalize from different training sets.
4. **Character Embeddings**: Visualizing and analyzing character embeddings for better understanding of the model's behavior.

## Installation

To run the experiments, you will need to set up the environment and install the required dependencies. Here's how you can do it:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/hogwarts-sorting-hat.git
    cd hogwarts-sorting-hat
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

## Experiments

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

3. **Mental Health Analysis (Figure 4)**  
    Exp ID: `2`  
    This experiment checks the impact of mental states on model performance for different window sizes and characters.

    ```bash
    python main.py --exp_id 2
    ```

4. **Meta-Learning Analysis (Figure 5)**  
    Exp ID: `3`  
    This experiment investigates the meta-learning performance of the model, specifically how well it generalizes to unseen data.

    ```bash
    python main.py --exp_id 3
    ```

5. **Character Embedding Analysis (Figure 6)**  
    Exp ID: `4`  
    This experiment analyzes character embeddings to better understand the model's learned representations.

    ```bash
    python main.py --exp_id 4
    ```

## Data Preparation

Before running the experiments, make sure the necessary datasets are ready. This is handled by the `prepear_data()` function, which will generate the required data files for training and testing. You can also manually ensure the data is available in the correct directories.

Run the following command to prepare the data:

```bash
python main.py --exp_id 0
```

This will trigger data generation and model training.

## Usage

- To train the model for all window sizes and characters, use:

    ```bash
    python main.py --exp_id 0
    ```

- To run the action distribution analysis, use:

    ```bash
    python main.py --exp_id 1
    ```

- To check the mental health analysis results, use:

    ```bash
    python main.py --exp_id 2
    ```

- To explore the meta-learning performance, use:

    ```bash
    python main.py --exp_id 3
    ```

- To analyze character embeddings, use:

    ```bash
    python main.py --exp_id 4
    ```

## License

This project is licensed under the HOGWARTS License ~
