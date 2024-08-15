import os
import csv


def replace_column_names(input_filename, output_filename, rename_mapping):
    # 需要替换的列名称映射
    rename_mapping = {
        "350k伏线路_hmd": "350k伏线路核密度",
        "RD-05-次干路（底衬）_hmd": "RD-05-次干路（底衬）核密度",
        "高压燃气管道路_hmd": "高压燃气管道路核密度",
        "RD-02-快速路_hmd": "RD-02-快速路核密度",
        "RD-03-2-一般主干路_hmd": "RD-03-2-一般主干路核密度",
        "现状电网线路_hmd": "现状电网线路核密度",
        "RD-01-高速路_hmd":"RD-01-高速路核密度",
        "轨道S4号线_hmd":"轨道S4号线核密度",
        "轨道2号线_hmd":"轨道2号线核密度",
        "供水路线_hmd":"供水路线核密度",
        "轨道S3号线_hmd":"轨道S3号线核密度",
        "轨道4号线_hmd":"轨道4号线核密度",
        "轨道环线_hmd":"道环线核密度",
        "轨道3号线_hmd":"轨道3号线核密度",
        "RD-04-公路_hmd":"RD-04-公路核密度",
        "轨道S2号线_hmd":"轨道S2号线核密度",
        "现状110k伏埋地电缆_hmd":"现状110k伏埋地电缆核密度",
        "轨道5号线_hmd":"轨道5号线核密度",
        "输气管道路线_hmd":"输气管道路线核密度",
        "RD-03-1-干线主干路_hmd":"RD-03-1-干线主干路核密度",
        "轨道1号线_hmd":"轨道1号线核密度",
        "现状220k伏埋地电缆_hmd":"现状220k伏埋地电缆核密度",
        '轨道S1号线_hmd':'轨道S1号线核密度',
        "距国家级的距离": "距国家级道路的距离",
        "距自治区级的距离": "距自治区级道路的距离",
        "距县级的距离": "距县级道路的距离",
        "距省级的距离": "距省级道路的距离",
        "距市级的距离": "距市级道路的距离"
    
    }

    # '':'核密度',
    # '':'核密度',
    # '':'核密度',
    # '':'核密度',
    # '':'核密度',
    # '':'核密度',
    # '':'核密度',
    # '':'核密度',
    # '':'核密度',
    with open(input_filename, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        column_names = next(reader)
        
        # 替换列名称
        new_column_names = [rename_mapping.get(col, col) for col in column_names]
        
        # 写回修改后的CSV文件
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(new_column_names)
            for row in reader:
                if row:  # Skip empty rows
                    writer.writerow(row)
    print('所有列名称已经修改完毕')                
    
