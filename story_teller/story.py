import numpy as np
import json
import os
from .call_api import call_gpt


possible_action_list = ['探索调查', '帮助NPC', '拒绝NPC', '背叛NPC', '战斗', '逃跑']


class StoryNode:
    '''
    Each node contains a text description of the story scene
    and a list of possible actions
    '''

    def __init__(self, node_id, text_description, action_list, father_id, father_choice, is_leaf, **kwargs):
        self.text_description = text_description
        self.action_list = action_list  # no more than 2 actions
        self.node_id = node_id
        self.children_id_list = []
        self.father_id = father_id
        self.father_choice = father_choice  # the choice made by the father node
        self.is_leaf = is_leaf

    def get_info_dict(self):
        return {
            'node_id': self.node_id,
            'text_description': self.text_description,
            'action_list': self.action_list,
            'children_id_list': self.children_id_list,
            'father_id': self.father_id,
            'father_choice': self.father_choice,
            'is_leaf': self.is_leaf
        }

    def load_info_dict(self, info_dict):
        self.node_id = info_dict['node_id']
        self.text_description = info_dict['text_description']
        self.action_list = info_dict['action_list']
        self.children_id_list = info_dict['children_id_list']
        self.father_id = info_dict['father_id']
        self.father_choice = info_dict['father_choice']
        self.is_leaf = info_dict['is_leaf']
    

class StoryTree:
    '''
    Tree structure for the story

    '''

    def __init__(self, batch_id=0):
        self.batch_id = batch_id

    def construct_story_tree(self):
        '''
        Construct the story tree by calling chatgpt api
        '''
        with open(f'../dataset/init_prompt/{self.batch_id}.txt', 'r', encoding='utf-8') as f:
            init_prompt = f.readlines()
        init_text_prompt = ''.join(init_prompt)
        init_text_prompt = init_text_prompt.split('####')
        for i in range(len(init_text_prompt)):
            if '剧情描述：' in init_text_prompt[i]:
                init_text_description = init_text_prompt[i].replace('剧情描述：', '')
            if '玩家选择：' in init_text_prompt[i]:
                init_action_list = [action for action in possible_action_list if action in init_text_prompt[i]]
        init_node_dict = {
            'node_id': 0,
            'text_description': init_text_description,
            'action_list': init_action_list,
            'father_id': -1,
            'father_choice': 'None',
            'is_leaf': True
        }
        story_node_list = [StoryNode(**init_node_dict)]
        top_id = 0
        tail_id = 1

        while top_id < tail_id:
            print(top_id)
            if top_id >= 32:
                break
            # pop the first element in the list
            current_story_node = story_node_list[top_id]
            story_node_list[top_id].is_leaf = False
            top_id += 1

            for idx, action in enumerate(current_story_node.action_list):
                # construct the prompt
                with open('middle_node_prompt.txt', 'r', encoding='utf-8') as f:
                    prompt = f.readlines()
                prompt = ''.join(prompt)

                with open(f'../dataset/background/{self.batch_id}.txt', 'r', encoding='utf-8') as f:
                    background_prompt = f.readlines()
                background_prompt = ''.join(background_prompt)

                prompt = prompt.replace("####background####", background_prompt)
                
                prompt = prompt.replace("####game_record####", self.get_story_history(current_story_node, story_node_list, action))

                # get the response from chatgpt
                response = call_gpt(prompt)

                response_dict = {
                    'node_id': tail_id,
                    'father_id': current_story_node.node_id,
                    'father_choice': action,
                    'is_leaf': True
                }
                response = response.split('####')
                for i in range(len(response)):
                    if '剧情描述：' in response[i]:
                        response_dict['text_description'] = response[i].replace('剧情描述：', '')
                    if '玩家选择：' in response[i]:
                        response_dict['action_list'] = [action for action in possible_action_list if action in response[i]]
                
                # construct the children nodes
                new_story_node = StoryNode(**response_dict)
                story_node_list[top_id-1].children_id_list.append(len(story_node_list))
                story_node_list.append(new_story_node)
                tail_id += 1
            
            # breakpoint()

        self.save_story_tree(story_node_list)

    def get_story_history(self, current_story_node, story_node_list, action):
        '''
        Get the story history
        '''
        history_node = []
        while current_story_node.father_id != -1:
            history_node.append(current_story_node)
            current_story_node = story_node_list[current_story_node.father_id]
        history_node.append(current_story_node)
        history_node.reverse()
        history_text = ''
        for idx, node in enumerate(history_node):
            if node.father_id != -1:
                history_text += f'玩家选择了<{node.father_choice}>\n'
            history_text += f'第{idx+1}幕：{node.text_description}\n'
            if idx == len(history_node) - 1:
                history_text += f'玩家选择了: <{action}>\n'
            
        return history_text


    def save_story_tree(self, story_node_list):
        '''
        save the story tree to a file
        '''

        file_dict = {}
        for story_node in story_node_list:
            file_dict[story_node.node_id] = story_node.get_info_dict()
        np.save(os.path.join('./dataset/story_tree_en', f'story_tree__batch_id_{self.batch_id}_new.npy'), file_dict, allow_pickle=True)

    def load_story_tree(self, path = './dataset/story_tree_en/story_tree__batch_id_0.npy'):
        '''
        load the story tree from a file
        '''
        self.story_node_list = []

        file_dict = np.load(path, allow_pickle=True).item()

        for node_id, info_dict in file_dict.items():
            assert node_id == info_dict['node_id']
            new_story_node = StoryNode(**info_dict)
            new_story_node.load_info_dict(info_dict)
            self.story_node_list.append(new_story_node)

    # def play_story(self):
    #     '''
    #     Play the story
    #     '''

    #     self.load_story_tree()

    #     self.current_node_id = 0

    #     while True:
    #         current_story_node = self.story_node_list[self.current_node_id]
    #         print(current_story_node.text_description)
    #         print('Please choose an action:')
    #         for i, action in enumerate(current_story_node.action_list):
    #             print(f'{i}: {action}')
    #         action_id = int(input())
    #         self.current_node_id = current_story_node.children_id_list[action_id]

    def reset(self, path = './dataset/story_tree_en/story_tree__batch_id_0.npy'):
        '''
        Reset the story
        '''
        self.load_story_tree(path=path)
        self.current_node_id = 0

        
        current_story_node = self.story_node_list[self.current_node_id]
        return current_story_node.text_description, current_story_node.action_list

    def step(self, action_id):
        '''
        Take a step in the story
        '''
        current_story_node = self.story_node_list[self.current_node_id]
        if not current_story_node.is_leaf:
            self.current_node_id = current_story_node.children_id_list[action_id]
            return self.story_node_list[self.current_node_id].text_description, self.story_node_list[self.current_node_id].action_list
        else:
            return None, None



def run_story_tree(step_depth = 10):
    story_tree = StoryTree()
    obs, valid_action = story_tree.reset()
    print(f"Now the game shall begin with an observation: {obs}")
    while step_depth > 0:
        print(obs)
        print('Please choose an action:')
        for i, action in enumerate(valid_action):
            print(f'{i}: {action}')
        action_id = int(input())
        obs, valid_action = story_tree.step(action_id)
        step_depth -= 1
        
        if obs is None and valid_action is None:
            print('The story has ended!')
            break



def batch_construct_story_tree():
    for batch_id in range(1, 30):
        try:
            story_tree = StoryTree(batch_id=batch_id)
            story_tree.construct_story_tree()
            print(f'Batch {batch_id} finished!')

        except Exception as e:
            print(f'Batch {batch_id} failed with error: {e}')
            continue



if __name__ == '__main__':
    
    batch_construct_story_tree()

    # run_story_tree()
