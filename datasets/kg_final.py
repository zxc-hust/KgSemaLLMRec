import json

# 1. 读取 remap_ids.json 并反转映射（值 -> 键）
with open('remap_ids.json', 'r') as f:
    remap_data = json.load(f)
reverse_remap = {v: k for k, v in remap_data.items()}  # 反转映射：用于替换头尾节点

# 2. 读取 relation_remap.json 映射关系（用于替换关系ID）
with open('datasets/amazon-book/relation_remap.json', 'r') as f:
    relation_remap = json.load(f)  # 格式: {"old_relation_id": "new_relation_id"}

# 3. 处理 kg_final.txt
input_file = 'datasets/amazon-book/kg_final_raw.txt'
output_file = 'datasets/amazon-book/kg_final.txt'

with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue  # 跳过空行

        # 分割三列
        parts = line.split()
        if len(parts) != 3:
            print(f"跳过无效行（列数错误）: {line}")
            continue

        head_id, relation_id, tail_id = parts[0], parts[1], parts[2]

        # 检查 head_id 和 tail_id 是否能映射
        if head_id in reverse_remap and tail_id in reverse_remap:
            new_head = reverse_remap[head_id]
            new_tail = reverse_remap[tail_id]
            
            # 检查 relation_id 是否能映射
            new_relation = relation_remap[relation_id]
            fout.write(f"{new_head} {new_relation} {new_tail}\n")

        else:
            # 打印被删除的行（可选）
            if head_id not in reverse_remap:
                print(f"删除行（head_id {head_id} 无映射）: {line}")
            if tail_id not in reverse_remap:
                print(f"删除行（tail_id {tail_id} 无映射）: {line}")

print(f"处理完成！结果保存至 {output_file}")