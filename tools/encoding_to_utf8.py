import os
import chardet

def encoding_to_utf8(folder_path):
    """
    将指定文件夹中所有CSV文件的编码转换为UTF-8。

    参数:
        folder_path (str): 包含CSV文件的文件夹路径。

    过程:
        1. 遍历给定文件夹中的所有文件。
        2. 仅处理扩展名为 .csv 的文件。
        3. 使用 `chardet` 库检测每个CSV文件的原始编码。
        4. 使用检测到的编码读取文件内容，忽略编码错误。
        5. 将文件内容重新保存为UTF-8编码。
        6. 打印转换状态和原始编码的消息。

    注意:
        需要 `chardet` 库用于编码检测。
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # 检测文件的原始编码
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                original_encoding = result['encoding']
            # 读取文件内容并转换为UTF-8编码
            with open(file_path, 'r', encoding=original_encoding, errors='ignore') as infile:
                content = infile.read()
            # 保存文件为UTF-8编码
            with open(file_path, 'w', encoding='utf-8', newline='') as outfile:
                outfile.write(content)
            print(f"已将 {filename} 从 {original_encoding} 转换为 UTF-8")
