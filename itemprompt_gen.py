import ast
from tqdm import tqdm
import json

# 读取 item_list.txt 构建初始 item_dict
item_dict = {}
search_asin = set()

with open('datasets/amazon-book/item_list.txt', 'r') as f:
    next(f)  # 跳过标题行

    # for i, line in enumerate(f):
    #     if i >= 10:  # 测试时只读取前10行（正式使用时删除这个条件）
    #         break
    for line in f:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        org_id, remap_id, freebase_id = parts
        # 初始化 item_dict 条目，保留 remap_id 和 freebase_id
        item_dict[org_id] = {
            'remap_id': remap_id,
            'freebase_id': freebase_id
        }
        search_asin.add(org_id)  # 记录需要搜索的 ASIN

# 先统计文件总行数以显示准确进度
print("正在统计 meta_Books.json 行数...")
# 遍历 meta_Books.json，将匹配到的 JSON 数据合并到 item_dict 中
with open('meta_Books.json', 'r') as f:
    total_lines = sum(1 for _ in f)
with open('meta_Books.json', 'r') as f:
    # for line in f:
    for line in tqdm(f, total=total_lines, desc="处理 meta_Books.json"):
        try:
            data = ast.literal_eval(line.strip())
            asin = data.get('asin', '')
            if asin in item_dict:
                # 生成模板字符串
                title = data.get('title', 'N/A').replace('\n', ' ')       # 处理换行符
                description = data.get('description', 'N/A').replace('\n', ' ') if data.get('description') else 'N/A'
                
                # 处理 categories（展平嵌套列表）
                categories = ', '.join([x for sublist in data.get('categories', []) for x in sublist]) or 'N/A'
                
                # 处理 price（添加美元符号）
                price = f"${data['price']}" if 'price' in data else 'N/A'
                
                # 处理 salesRank（提取 Books 排名）
                sales_rank = data.get('salesRank', {}).get('Books', 'N/A')
                
                # # 构建 prompt
                # prompt = (
                #     f"Title: {title}\n"
                #     f"Description: {description}\n"
                #     f"Categories: {categories}\n"
                #     f"Price: {price}\n"
                #     f"Sales Rank: {sales_rank}"
                # )
                template = (f"The point of interest has the following attributes: title is {title}; description is {description}; categories is {categories}; price is {price}; sales Rank is {sales_rank}")
                # 将元数据和 template 一起存入
                item_dict[asin].update({'template': template})
                
                search_asin.discard(asin)  # 标记为已找到
                
        except Exception as e:
            print(f"解析错误: {e}\n行内容: {line}")
            continue

# 定义保存路径
output_path = 'datasets/amazon-book/item_dict.json'

# 保存为带缩进的 JSON 格式
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(
        item_dict,
        f,
        ensure_ascii=False,  # 保留非ASCII字符（如中文）
        indent=2            # 缩进美化
    )

print(f"数据已保存至 {output_path}")
# # 统计结果
# not_found = list(search_asin)
# print(f"成功合并 {len(item_dict) - len(not_found)} 条，未找到 {len(not_found)} 条。")

# # 打印合并后的示例（前3个）
# for asin in list(item_dict.keys())[:3]:
#     if 'prompt' in item_dict[asin]:
#         print(f"ASIN: {asin}\nPrompt:\n{item_dict[asin]['prompt']}\n")
#     else:
#         print(f"ASIN: {asin}\n(未找到元数据)\n")