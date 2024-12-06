import numpy as np

story_tree = np.load('story_tree.npy', allow_pickle=True).item()

print(story_tree)



# for i in range(0, 30):
#     with open('init_node_prompt.txt', 'r', encoding='utf-8') as f:
#         init_prompt = f.readlines()
#     init_prompt = ''.join(init_prompt)
#     with open(f'../dataset/background/{i}.txt', 'r', encoding='utf-8') as f:
#         background_prompt = f.readlines()
#     background_prompt = ''.join(background_prompt)

#     init_prompt = init_prompt.replace("####background####", background_prompt)

#     with open(f'../dataset/init_prompt/{i}.txt', 'w', encoding='utf-8') as f:
#         f.write(init_prompt)
