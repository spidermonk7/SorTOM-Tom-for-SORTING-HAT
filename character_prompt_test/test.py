import json


def check_data(path):
    with open(path, "r") as file:
        data = json.load(file)
    for id in data:
        item = data[id]
        state = item["state"]
        action = item["action"]
        print("ID:", id)
        print("State:", state)
        print("Action:", action)
        print('-'*50)

if __name__ == '__main__':
    path = "data/jack_Hufflepuff_v2.json"
    check_data(path)