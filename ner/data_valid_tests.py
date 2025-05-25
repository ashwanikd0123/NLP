import json
import os
from common_utils import *
import unittest

def test_file_load(file_path):
    with open(os.path.join(file_path), "r") as file:
        try:
            json.load(file)
        except Exception as e:
            print(e)
            return False
    return True

class DataTests(unittest.TestCase):
    def test_lables(self):
        self.assertTrue(test_file_load(os.path.join(repair_dir_path, label_file_name)))

    def test_train(self):
        for f_name in train_file_names:
            self.assertTrue(test_file_load(os.path.join(repair_dir_path, f_name)))

    def test_valid(self):
        self.assertTrue(test_file_load(os.path.join(repair_dir_path, valid_file_name)))

if __name__ == "__main__":
    unittest.main()
