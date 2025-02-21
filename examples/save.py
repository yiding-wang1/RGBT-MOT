import os
import shutil

def copy_csv_files(root_a, root_b):
    # 遍历根目录 a 及其子目录
    for root, dirs, files in os.walk(root_a):
        for file in files:
            if file.endswith('.csv'):
                # 获取文件在根目录 a 中的完整路径
                file_path_a = os.path.join(root, file)
                # 获取文件相对于根目录 a 的相对路径
                relative_path = os.path.relpath(root, root_a)
                # 构建文件在根目录 b 中的目标路径
                target_dir = os.path.join(root_b, relative_path)
                # 如果目标路径不存在，则创建该路径
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                # 构建文件在根目录 b 中的完整目标路径
                target_file_path = os.path.join(target_dir, file)
                # 复制文件到目标路径
                shutil.copy2(file_path_a, target_file_path)
                print(f"已复制 {file_path_a} 到 {target_file_path}")

# 定义根目录 a 和根目录 b 的路径
root_a = '../../../dataset/csv_216_1/csv_216'
root_b = 'E:/lasher/LasHeR_Unalined_960_0615/seleted'

# 调用函数进行文件复制
copy_csv_files(root_a, root_b)
