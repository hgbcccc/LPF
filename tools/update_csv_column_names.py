import csv
import json
import os

def update_csv_column_names(csv_file, mapping, output_file, suffixes=None, prefixes=None):
    """
    更新CSV文件中的列名称，并将结果保存到新的CSV文件中。

    参数:
        csv_file (str): 输入CSV文件的路径。
        mapping (dict): 列名映射字典，格式为 {旧列名: 新列名}。
        output_file (str): 输出CSV文件的路径。
        suffixes (list, optional): 要去除的后缀列表，默认为空列表。
        prefixes (list, optional): 要去除的前缀列表，默认为空列表。
    """

    def strip_suffix_prefix(column_name, suffixes, prefixes):
        """去除指定的前缀和后缀"""
        for suffix in suffixes:
            if column_name.endswith(suffix):
                column_name = column_name[:-len(suffix)]
        for prefix in prefixes:
            if column_name.startswith(prefix):
                column_name = column_name[len(prefix):]
        return column_name

    suffixes = suffixes or []
    prefixes = prefixes or []

    # 尝试使用UTF-8和GBK读取CSV文件
    encoding = None
    for enc in ['utf-8', 'gbk']:
        try:
            with open(csv_file, 'r', encoding=enc) as infile:
                reader = csv.DictReader(infile)
                fieldnames = reader.fieldnames
                encoding = enc
                break
        except (UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Failed to read {csv_file} with encoding {enc}: {e}")
            continue

    if not encoding:
        raise ValueError("Unable to read the CSV file with UTF-8 or GBK encoding.")

    with open(csv_file, 'r', encoding=encoding) as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        new_fieldnames = []

        for field in fieldnames:
            aaa = ''
            if field.endswith('_dis'):
                aaa = '_dis'
            stripped_name = strip_suffix_prefix(field, suffixes, prefixes)
            new_fieldname = mapping.get(stripped_name, field) + aaa
            new_fieldnames.append(new_fieldname)

            print(f"原特征名称: {field}, 修改为新的特征名称: {new_fieldname}")

        writer = csv.DictWriter(outfile, fieldnames=new_fieldnames)
        writer.writeheader()
        infile.seek(0)  # Reset file pointer to start
        next(reader)  # Skip header row
        for row in reader:
            writer.writerow({new_fieldnames[i]: row[fieldnames[i]] for i in range(len(fieldnames))})
            
            
            
            
            
            
            
            
            
            
            
            
# # 读取映射字典
# def load_mapping(mapping_file):
#     with open(mapping_file, 'r', encoding='utf-8') as file:
#         return json.load(file)

# # 去除指定的前缀和后缀
# def strip_suffix_prefix(column_name, suffixes, prefixes):
#     for suffix in suffixes:
#         if column_name.endswith(suffix):
#             column_name = column_name[:-len(suffix)]
#     for prefix in prefixes:
#         if column_name.startswith(prefix):
#             column_name = column_name[len(prefix):]
#     return column_name

# # 更新CSV文件的列名称
# def update_csv_column_names(csv_file, mapping, output_file, suffixes=None, prefixes=None):
#     suffixes = suffixes or []
#     prefixes = prefixes or []
    
#     #reverse_mapping = {v: k for k, v in mapping.items()}
#     reverse_mapping = mapping

#     with open(csv_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
#         reader = csv.DictReader(infile)
#         fieldnames = reader.fieldnames
#         new_fieldnames = []


        
#         for field in fieldnames:
#             aaa=''
#             if field.endswith('_dis'):aaa='_dis'
#             stripped_name = strip_suffix_prefix(field, suffixes, prefixes)
#             new_fieldname = reverse_mapping.get(stripped_name, field)+aaa
#             new_fieldnames.append(new_fieldname)
            
#             print(stripped_name,new_fieldname)

#         writer = csv.DictWriter(outfile, fieldnames=new_fieldnames)
#         writer.writeheader()
#         for row in reader:
#             new_row = {new_fieldname: row[old_fieldname] for old_fieldname, new_fieldname in zip(fieldnames, new_fieldnames)}
#             writer.writerow(new_row)



# def update_csv_column_names(csv_file, mapping, output_file, suffixes=None, prefixes=None):
#     """
#     更新CSV文件中的列名称，并将结果保存到新的CSV文件中。

#     参数:
#         csv_file (str): 输入CSV文件的路径。
#         mapping (dict): 列名映射字典，格式为 {旧列名: 新列名}。
#         output_file (str): 输出CSV文件的路径。
#         suffixes (list, optional): 要去除的后缀列表，默认为空列表。
#         prefixes (list, optional): 要去除的前缀列表，默认为空列表。
#     """

#     def strip_suffix_prefix(column_name, suffixes, prefixes):
#         """去除指定的前缀和后缀"""
#         for suffix in suffixes:
#             if column_name.endswith(suffix):
#                 column_name = column_name[:-len(suffix)]
#         for prefix in prefixes:
#             if column_name.startswith(prefix):
#                 column_name = column_name[len(prefix):]
#         return column_name

#     suffixes = suffixes or []
#     prefixes = prefixes or []

#     # 尝试使用UTF-8和GBK读取CSV文件
#     encoding = None
#     for enc in ['utf-8', 'gbk']:
#         try:
#             with open(csv_file, 'r', encoding=enc) as infile:
#                 reader = csv.DictReader(infile)
#                 fieldnames = reader.fieldnames
#                 encoding = enc
#                 break
#         except (UnicodeDecodeError, FileNotFoundError) as e:
#             print(f"Failed to read {csv_file} with encoding {enc}: {e}")
#             continue

#     if not encoding:
#         raise ValueError("Unable to read the CSV file with UTF-8 or GBK encoding.")

#     reverse_mapping = mapping

#     with open(csv_file, 'r', encoding=encoding) as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
#         reader = csv.DictReader(infile)
#         fieldnames = reader.fieldnames
#         new_fieldnames = []

#         for field in fieldnames:
#             aaa = ''
#             if field.endswith('_dis'):
#                 aaa = '_dis'
#             stripped_name = strip_suffix_prefix(field, suffixes, prefixes)
#             new_fieldname = reverse_mapping.get(stripped_name, field) + aaa
#             new_fieldnames.append(new_fieldname)

#             print(f"原特征名称: {field},修改为新的特征名称, New: {new_fieldname}")

#         writer = csv.DictWriter(outfile, fieldnames=new_fieldnames)
#         writer.writeheader()
#         infile.seek(0)  # Reset file pointer to start
#         next(reader)  # Skip header row
#         for row in reader:
#             writer.writerow({new_fieldnames[i]: row[fieldnames[i]] for i in range(len(fieldnames))})


#     #update_csv_column_names(csv_file, mapping, output_file, suffixes=None, prefixes=None)
