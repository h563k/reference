import os


def get_path_list():
    path_list = []
    for x in os.listdir("origin_data"):
        file_path = os.path.join("origin_data", x, "参考文献")
        path_list.append(file_path)
    return path_list
