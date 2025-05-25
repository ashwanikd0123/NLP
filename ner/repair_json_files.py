import json
import os
from common_utils import *
from tqdm import tqdm

indent_count = 4

def get_repaired_data(file_path):
    data_list = []
    with open(file_path, "r") as file:
        data_list.extend([json.loads(cur) for cur in file.readlines()])
    return data_list

if __name__ == "__main__":
    file_names = train_file_names
    file_names.append(test_file_name)
    file_names.append(valid_file_name)

    pbar = tqdm(file_names)
    for f_name in pbar:
        pbar.set_description("repairing: " + f_name)
        read_file_path = os.path.join(dir_path, f_name)
        write_file_path = os.path.join(repair_dir_path, f_name)
        data = get_repaired_data(read_file_path)
        with open(write_file_path, "w") as file:
            json.dump(data, file, indent = indent_count)

    print("finished...")