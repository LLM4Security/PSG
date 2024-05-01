import os
from tqdm import tqdm

def get_java_file_paths(directory_path):
    java_file_paths = []

    # 递归遍历目录
    for root, dirs, files in tqdm(os.walk(directory_path)):
        for file in files:
            if file.endswith(".java"):
                # 如果是Java文件，获取文件的完整路径
                java_file_path = os.path.abspath(os.path.join(root, file))
                # 将文件路径添加到列表中
                java_file_paths.append(java_file_path)

    return java_file_paths

def write_to_file(output_file, data):
    with open(output_file, 'w') as file:
        for item in data:
            file.write("%s\n" % item)

if __name__ == "__main__":
    # 指定目录路径
    # 反编译结果目录
    directory_path = ""

    # 获取目录下所有Java文件的完整路径
    java_file_paths = get_java_file_paths(directory_path)

    # 指定输出文件路径
    output_file_path = "./java_path.txt"

    # 将结果写入文件
    write_to_file(output_file_path, java_file_paths)

    print(f"结果已写入到文件: {output_file_path}")
