# import json

# # 加载处理后的实体文本映射
# with open('entity_texts_processed.json', 'r', encoding='utf-8') as f:
#     entity_mapping = json.load(f)

# # 创建从original_id到key的反向映射
# reverse_mapping = {v['original_id']: k for k, v in entity_mapping.items()}

# # 处理原始文件
# input_file = 'datasets/amazon-book/entity_list_raw.txt'
# output_file = 'datasets/amazon-book/entity_list.txt'

# with open(input_file, 'r', encoding='utf-8') as fin, \
#      open(output_file, 'w', encoding='utf-8') as fout:
    
#     # 写入第一行（保持不变）
#     first_line = fin.readline()
#     fout.write(first_line)
    
#     # 处理剩余行
#     for line in fin:
#         parts = line.strip().split()
#         if len(parts) == 2:
#             org_id, remap_id = parts
#             # 检查remap_id是否在映射中
#             if remap_id in reverse_mapping:
#                 new_remap_id = reverse_mapping[remap_id]
#                 fout.write(f"{org_id} {new_remap_id}\n")
#             # 如果不在映射中，则跳过（相当于删除该行）

import json

# 1. 读取 remap_ids.json 并反转映射关系（值 -> 键）
with open('datasets/amazon-book/remap_ids.json', 'r') as f:
    remap_data = json.load(f)

# 反转映射：{"值": "键"}
reverse_remap = {v: k for k, v in remap_data.items()}

# 2. 处理 entity_list_raw.txt
input_file = 'datasets/amazon-book/entity_list_raw.txt'
output_file = 'datasets/amazon-book/entity_list.txt'

with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    # 写入表头
    header = fin.readline()
    fout.write(header)
    
    # 处理每一行
    for line in fin:
        parts = line.strip().rsplit(maxsplit=1)
        if len(parts) < 2:
            continue  # 跳过不完整的行
        
        org_id, remap_id = parts[0], parts[1]
        
        # 查找 remap_id 是否在反转映射中
        if remap_id in reverse_remap:
            new_remap_id = reverse_remap[remap_id]
            fout.write(f"{org_id} {new_remap_id}\n")
        else:
            print(f"删除行：{line.strip()}（未找到 remap_id {remap_id} 的映射）")

print("处理完成！结果已保存到", output_file)