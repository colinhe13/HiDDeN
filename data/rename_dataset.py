import os


def rename_and_count_files(folder_path):
    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 统计文件数量
    file_count = len(files)

    # 对文件按数字顺序重命名
    for i, file_name in enumerate(sorted(files)):
        original_path = os.path.join(folder_path, file_name)
        # new_name = f"train_image{i + 1}.jpg"
        new_name = f"val_image{i + 1}.jpg"
        new_path = os.path.join(folder_path, new_name)
        os.rename(original_path, new_path)

    return file_count


# 指定文件夹路径
# folder_path = 'boss_h/data_size512/500_50/train/train_class'
folder_path = 'boss_h/data_size512/500_50/val/val_class'

# 调用函数
file_count = rename_and_count_files(folder_path)

print(f"Total files: {file_count}")
print("Files renamed successfully.")
