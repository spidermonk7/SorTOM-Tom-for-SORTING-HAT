# SorTOM-Tom-for-SORTING-HAT
![6651733464215_ pic](https://github.com/user-attachments/assets/f81a75d9-9b7b-49ec-882d-6aa9cc8ec49d)




# Quick Start

`python experiment.py`

# File Structure with explanation 

## Cricial Files

### experiments.py
Including codes for training Sorting-Hat models, and a possible analysis experiments of character embedding. 

### data_generator.py
All functions that could be used for loading data. 
**F.Y.I**: If you've already embedded trajectory data with BERT, you can just load it with ***TrajectoryDataset*** 


### model.py

Including models: SimpleMLP, SimpleRNN, SimpleLSTM, SimpleGRU
Crucial models: StateEmbedding, CharaNet, MentNet, SortingHat



### utils.py
Possible utils. 





## Folder: Character_prompt_test
This involve the code for generating data and adjusting prompts for different characters. **You don't need to read it during current stage.** 

## Folder: datasets
This involve all the things that can be called as "data" in our project till current stage, in which if:


**You want to fix the window-overlapping problem** —— Go and check the folder **final_data_en** for the game recordings of 4 character players on 19 different game batches. 

**You want to build up the final Sorting Hat** -- Focus on the **Trajectory** folders for splited(though with overlapping) text dataset(stored in JSON), you should first embed them with bert, than trying SortingHat models to get their Character Embeddings. 
***Tips***:  
(1) Check data_generator.py: ***get_trajectory_embeded*** method for help, it's convinient.  

(2) Use model.chara after calling forward method to got the recorded character state.     

(3) You are strongly recommanded to build up a comprehensive SORTING-HAT based on bose charasteristic and Mental State(hidden state of MentNet, just try to take it into consideration). 



## Folder: story_teller
This involve the code for generating DND games and adjusting prompts for different scenarios. **You don't need to read it during current stage.**


## Folder: models
Here stores the .pt file of trained SortingHat model, it is also the default path for model saving. 
Initially, we present 5 models: SortingHat_{i}.pt, where i indicates the window size. 


