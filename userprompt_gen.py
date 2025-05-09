import json
from collections import defaultdict
import ast
from tqdm import tqdm
import pickle
import os
import json
# 1. 建立remap_id到org_id的映射
remap_to_org = {}
required_org_ids = set()
with open('/data/zou/KGAT-pytorch-master/datasets/amazon-book/item_list.txt', 'r') as f:
    next(f)  # 跳过标题行
    for line in f:
        parts = line.strip().split()
        org_id = parts[0]
        remap_id = int(parts[1])
        remap_to_org[remap_id] = org_id
        required_org_ids.add(org_id) 

# 2. 建立org_id到title的映射
if os.path.exists('/data/zou/KGAT-pytorch-master/datasets/amazon-book/org_to_title.pkl'):
    with open('/data/zou/KGAT-pytorch-master/datasets/amazon-book/org_to_title.pkl', 'rb') as f:
        org_to_title = pickle.load(f)
else:
    org_to_title = {}
    file_path = '/data/zou/KGAT-pytorch-master/meta_Books.json'
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="处理meta_Books.json"):
            data = ast.literal_eval(line.strip())
            asin = data.get('asin')
            # org_to_title[data['asin']] = data['title']
            if asin in required_org_ids:
                title = data.get('title', '')  # 如果没有title则设为空字符串
                org_to_title[asin] = title
    with open('/data/zou/KGAT-pytorch-master/datasets/amazon-book/org_to_title.pkl', 'wb') as f:
        pickle.dump(org_to_title, f)
# 3. 处理train.txt文件并生成模板
user_templates = {}
with open('/data/zou/KGAT-pytorch-master/datasets/amazon-book/train.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        user_id = parts[0]
        item_ids = list(map(int, parts[1:]))
        
        # 获取所有标题
        titles = []
        for item_id in item_ids:
            org_id = remap_to_org.get(item_id)
            if org_id:
                title = org_to_title.get(org_id, f"未知标题(org_id: {org_id})")
                titles.append(title)
        
        # 生成模板
        books_str = "，".join([f"{title}，" for title in titles])
        template = f"This user has read {books_str} Please summarize the user's interests."
        user_templates[user_id] = template

output_file = "/data/zou/KGAT-pytorch-master/datasets/amazon-book/user_templates.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(user_templates, f, ensure_ascii=False, indent=4)

print(f"用户模板已成功保存到 {output_file}")