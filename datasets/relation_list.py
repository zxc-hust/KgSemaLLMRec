import json

# 输入和输出文件路径
input_file = 'datasets/amazon-book/relation_list_raw.txt'
output_file = 'datasets/amazon-book/relation_list.txt'
mapping_file = 'datasets/amazon-book/relation_remap.json'

# 需要删除的 remap_id（转换为字符串，因为文件中的 remap_id 是字符串）
remap_ids_to_delete = {'2', '6', '32', '37', '31', '35'}

# 1. 读取文件并过滤行
with open(input_file, 'r') as fin:
    lines = fin.readlines()

header = lines[0]  # 保留表头
data_lines = [line.strip().split(maxsplit=1) for line in lines[1:]]  # 分割为 [org_id, remap_id]

# 过滤掉要删除的 remap_id
filtered_lines = []
remap_mapping = {}  # 记录新旧 remap_id 的映射：{"old_remap_id": new_remap_id}

new_remap_id = 0
for parts in data_lines:
    if len(parts) != 2:
        continue  # 跳过无效行
    org_id, old_remap_id = parts[0], parts[1]
    if old_remap_id in remap_ids_to_delete:
        continue  # 跳过要删除的行
    
    # 记录映射关系
    remap_mapping[old_remap_id] = str(new_remap_id)
    filtered_lines.append(f"{org_id} {new_remap_id}\n")
    new_remap_id += 1

# 2. 写入新文件
with open(output_file, 'w') as fout:
    fout.write(header)  # 写入表头
    fout.writelines(filtered_lines)  # 写入处理后的行

# 3. 保存映射关系到 JSON 文件
with open(mapping_file, 'w') as fmap:
    json.dump(remap_mapping, fmap, indent=2)

print(f"处理完成！结果保存至 {output_file}")
print(f"映射关系保存至 {mapping_file}")