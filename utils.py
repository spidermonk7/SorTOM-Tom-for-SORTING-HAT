import os


def check_path(path = "./dataset"):
    if not os.path.exists(path):
        os.makedirs(path)
