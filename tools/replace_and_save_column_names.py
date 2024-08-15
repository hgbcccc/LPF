import os
import csv
import json
import re

def replace_and_save_column_names(input_filename, output_filename, json_path):
    '''
    替换CSV文件中的列名并将处理后的数据保存到新的CSV文件中。

    参数:
        input_filename (str): 输入CSV文件的路径。
        output_filename (str): 输出CSV文件的路径。
        json_path (str): 包含列名映射规则的JSON文件的路径。

    过程:
        1. 读取映射规则JSON文件，将其内容加载到 `mapping` 变量中。
        2. 读取输入的CSV文件，并提取列名。
        3. 使用正则表达式匹配和处理不同格式的列名（例如包含六位、四位、两位数字的列名）。
        4. 根据映射规则替换列名，并记录未匹配的列名。
        5. 将替换后的列名写入新的CSV文件中，保持数据的其他内容不变。
'''

    
    with open(json_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
        
    
    with open(input_filename, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        column_names = next(reader)

        # 使用正则表达式匹配各种格式的列名
        pattern_six_digits = re.compile(r'poi_(\d{6})_dis')
        pattern_two_digits = re.compile(r'poi_(\d{2})_dis')
        pattern_four_digits = re.compile(r'poi_(\d{4})_dis')
        pattern_generic_dis = re.compile(r'(.*)_dis$')

        new_column_names = []
        non_matching_columns = {
            'two_digits': [],
            'four_digits': [],
            'other': []
        }
        modified_six_digits = []
        modified_four_digits = []
        modified_two_digits = []
        modified_generic_dis = []
        renamed_others = []

        rename_mapping = {
            "poi_11_hmd": "餐饮POI核密度",
            "poi_12_hmd": "住、宿POI核密度",
            "poi_13_hmd": "批发、零售POI核密度",
            "poi_14_hmd": "汽车销售及服务POI核密度",
            "poi_15_hmd": "金融、保险POI核密度",
            "poi_16_hmd": "教育、文化POI核密度",
            "poi_17_hmd": "卫生、社保POI核密度",
            "poi_18_hmd": "运动、休闲POI核密度",
            "poi_19_hmd": "公共设施POI核密度",
            "poi_20_hmd": "商业设施、商务服务POI核密度",
            "poi_21_hmd": "居民服务POI核密度",
            "poi_22_hmd": "公司企业POI核密度",
            "poi_23_hmd": "交通运输、仓储POI核密度",
            "poi_24_hmd": "科研及技术服务POI核密度",
            "poi_25_hmd": "农林牧渔业POI核密度",
            "poi_26_hmd": "地名地址POI核密度",

        }

        for col in column_names:
            match_six = pattern_six_digits.match(col)
            match_two = pattern_two_digits.match(col)
            match_four = pattern_four_digits.match(col)
            match_generic = pattern_generic_dis.match(col)
            
            if match_six:
                key = match_six.group(1)
                if key in mapping:
                    new_col_name = f"距{mapping[key]}的距离_{key}"
                    new_column_names.append(new_col_name)
                    modified_six_digits.append(new_col_name)
                else:
                    new_column_names.append(col)
                    non_matching_columns['other'].append(col)
            elif match_two:
                key = match_two.group(1)
                if key in mapping:
                    new_col_name = f"距{mapping[key]}的距离_{key}"
                    new_column_names.append(new_col_name)
                    modified_two_digits.append(new_col_name)
                else:
                    new_column_names.append(col)
                    non_matching_columns['other'].append(col)
            elif match_four:
                key = match_four.group(1)
                if key in mapping:
                    new_col_name = f"距{mapping[key]}的距离_{key}"
                    new_column_names.append(new_col_name)
                    modified_four_digits.append(new_col_name)
                else:
                    new_column_names.append(col)
                    non_matching_columns['other'].append(col)
            elif match_generic:
                new_col_name = f"距{match_generic.group(1)}的距离"
                new_column_names.append(new_col_name)
                modified_generic_dis.append(new_col_name)
            elif col in rename_mapping:
                new_col_name = rename_mapping[col]
                new_column_names.append(new_col_name)
                renamed_others.append(new_col_name)
            else:
                new_column_names.append(col)
                non_matching_columns['other'].append(col)
        
        print(f"列名替换完成。以下列名已被修改：")
        print(f"六位数字列名: {modified_six_digits}")
        print(f"四位数字列名: {modified_four_digits}")
        print(f"两位数字列名: {modified_two_digits}")
        print(f"通用格式列名: {modified_generic_dis}")
        print(f"其他未匹配的列名: {non_matching_columns['other']}")
        
        # 写回修改后的CSV文件
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(new_column_names)
            for row in reader:
                writer.writerow(row)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
# 加载JSON文件


# def replace_and_save_column_names(input_filename, output_filename, json_path):
    
#     with open(json_path, 'r', encoding='utf-8') as f:
#         mapping = json.load(f)
    
    
#     with open(input_filename, 'r', newline='', encoding='utf-8') as csvfile:
#         reader = csv.reader(csvfile)
#         column_names = next(reader)

#         # 使用正则表达式匹配各种格式的列名
#         pattern_six_digits = re.compile(r'poi_(\d{6})_dis')
#         pattern_two_digits = re.compile(r'poi_(\d{2})_dis')
#         pattern_four_digits = re.compile(r'poi_(\d{4})_dis')
#         pattern_generic_dis = re.compile(r'(.*)_dis$')

#         new_column_names = []
#         non_matching_columns = {
#             'two_digits': [],
#             'four_digits': [],
#             'other': []
#         }
#         modified_six_digits = []
#         modified_four_digits = []
#         modified_two_digits = []
#         modified_generic_dis = []
#         renamed_others = []

#         rename_mapping = {
#             "poi_11_hmd": "餐饮POI核密度",
#             "poi_12_hmd": "住、宿POI核密度",
#             "poi_13_hmd": "批发、零售POI核密度",
#             "poi_14_hmd": "汽车销售及服务POI核密度",
#             "poi_15_hmd": "金融、保险POI核密度",
#             "poi_16_hmd": "教育、文化POI核密度",
#             "poi_17_hmd": "卫生、社保POI核密度",
#             "poi_18_hmd": "运动、休闲POI核密度",
#             "poi_19_hmd": "公共设施POI核密度",
#             "poi_20_hmd": "商业设施、商务服务POI核密度",
#             "poi_21_hmd": "居民服务POI核密度",
#             "poi_22_hmd": "公司企业POI核密度",
#             "poi_23_hmd": "交通运输、仓储POI核密度",
#             "poi_24_hmd": "科研及技术服务POI核密度",
#             "poi_25_hmd": "农林牧渔业POI核密度",
#             "poi_26_hmd": "地名地址POI核密度",

#         }

#         for col in column_names:
#             match_six = pattern_six_digits.match(col)
#             match_two = pattern_two_digits.match(col)
#             match_four = pattern_four_digits.match(col)
#             match_generic = pattern_generic_dis.match(col)
            
#             if match_six:
#                 key = match_six.group(1)
#                 if key in mapping:
#                     new_col_name = f"距{mapping[key]}的距离"
#                     new_column_names.append(new_col_name)
#                     modified_six_digits.append(new_col_name)
#                 else:
#                     new_column_names.append(col)
#                     non_matching_columns['other'].append(col)
#             elif match_two:
#                 key = match_two.group(1)
#                 if key in mapping:
#                     new_col_name = f"距{mapping[key]}的距离"
#                     new_column_names.append(new_col_name)
#                     modified_two_digits.append(new_col_name)
#                 else:
#                     new_column_names.append(col)
#                     non_matching_columns['other'].append(col)
#             elif match_four:
#                 key = match_four.group(1)
#                 if key in mapping:
#                     new_col_name = f"距{mapping[key]}的距离"
#                     new_column_names.append(new_col_name)
#                     modified_four_digits.append(new_col_name)
#                 else:
#                     new_column_names.append(col)
#                     non_matching_columns['other'].append(col)
#             elif match_generic:
#                 new_col_name = f"距{match_generic.group(1)}的距离"
#                 new_column_names.append(new_col_name)
#                 modified_generic_dis.append(new_col_name)
#             elif col in rename_mapping:
#                 new_col_name = rename_mapping[col]
#                 new_column_names.append(new_col_name)
#                 renamed_others.append(new_col_name)
#             else:
#                 new_column_names.append(col)
#                 non_matching_columns['other'].append(col)
        
#         # # 打印修改后的列名称
#         # print("修改后所有的列名称：")
#         # print(new_column_names)
#         # print("\n\n\n")
        
#         # 打印poi_六位数字_dis的修改列名称
#         # print("poi_六位数字_dis的修改列名称：")
#         # print(modified_six_digits)
#         # print("\n\n\n")
        
#         # # 打印poi_四位数字_dis的修改列名称
#         # print("poi_四位数字_dis的修改列名称：")
#         # print(modified_four_digits)
#         # print("\n\n\n")

#         # # 打印poi_两位数字_dis的修改列名称
#         # print("poi_两位数字_dis的修改列名称：")
#         # print(modified_two_digits)
#         # print("\n\n\n")

#         # # 打印形如xxx_dis的修改列名称
#         # print("形如xxx_dis的修改列名称：")
#         # print(modified_generic_dis)
#         # print("\n\n\n")
        
#         # # 打印不匹配的列名称
#         # print("其他形式的列名称：")
#         # print(non_matching_columns['other'])
#         # print("\n\n\n")

#         # # 打印重命名的其他列名称
#         # print("重命名的其他列名称：")
#         # print(renamed_others)
#         # print("\n\n\n")
#         print(f"列名替换完成。以下列名已被修改：")
#         print(f"六位数字列名: {modified_six_digits}")
#         print(f"四位数字列名: {modified_four_digits}")
#         print(f"两位数字列名: {modified_two_digits}")
#         print(f"通用格式列名: {modified_generic_dis}")
#         print(f"其他未匹配的列名: {non_matching_columns['other']}")
        
#         # 写回修改后的CSV文件
#         with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(new_column_names)
#             for row in reader:
#                 writer.writerow(row)
