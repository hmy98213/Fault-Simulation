import os
import shutil

def get_inst(path_from, path_to):
    folder = os.listdir(path_from)
    for file_name in folder:
        if(file_name.endswith("0_0.qasm")):
            original = path_from + file_name
            target = path_to + file_name
            shutil.move(original, target)

if __name__ == "__main__":
    path_from = "Benchmarks/inst_ori/"
    path_to = "Benchmarks/inst_test/"
    get_inst(path_from, path_to)