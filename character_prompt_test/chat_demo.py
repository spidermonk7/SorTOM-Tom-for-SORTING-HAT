import os
import openai
from .utils import *

openai.api_key = "sk-MSSwI7MgizQFSyUE64359c5000D64b518cCc7c00F30e0321"
openai.base_url = "https://api.gpt.ge/v1/"
# openai.base_url = "https://api.v3.cm/v1/"
openai.default_headers = {"x-foo": "true"}


def generate_answer(prompt, character, game, model = 'gpt-4o-2024-08-06', max_tokens = 100, debug = False):
    prompt_path = "character_prompt_test/prompts/characters/" + character + ".md"
    assert character in ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"], "The character is not supported"    
    with open(prompt_path, "r") as file:
        system_prompt = file.read() 

    game_prompt_path = "character_prompt_test/prompts/rules/" + game + ".md"
    assert game in ["taxi", "lake", "jack", "dnd"], "The game is not supported"
    with open(game_prompt_path, "r") as file:
        game_prompt = file.read()

    system_prompt = system_prompt + game_prompt
    if game == "dnd":
        prompt = "Now you got the input state: \n" + prompt + "\n"
    else:
        prompt = "Now you got the input state: \n" + parse_obs(prompt, game=game) + "\n"

    completion = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content":system_prompt,              
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return completion.choices[0].message.content





if __name__ == '__main__':
    # character = "Hufflepuff"
    # game = "jack"

    # obs = (23, 10)
    # action = generate_answer(str(obs), character, game)
    # print(action)
    game_path = "story_teller/story_tree.npy"
    translate_game(game_path)