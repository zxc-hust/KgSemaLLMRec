# import json
# import os
# from operator import itemgetter

# def process_entities(input_path, output_path):
#     # 读取原始实体文件
#     with open(input_path, 'r', encoding='utf-8') as f:
#         raw_entities = json.load(f)

#     # 创建处理后的实体列表（保持顺序）
#     processed = []
    
#     # 提取并排序原始实体ID（降序排列）
#     sorted_ids = sorted(raw_entities.keys(), 
#                       key=lambda x: int(x), 
#                       reverse=True)

#     # 重新编号并处理字段
#     for new_id, original_id in enumerate(sorted_ids, 1):
#         entity = raw_entities[original_id]
        
#         # 清洗字段数据
#         name = entity['type.object.name'] \
#               .replace('"@en', '') \
#               .strip('"')
        
#         desc = entity['common.topic.description'] \
#               .replace('"@en', '') \
#               .strip('"')

#         # 构建新实体结构
#         processed.append({
#             "new_id": str(new_id),
#             "original_id": original_id,
#             "name": name,
#             "description": desc
#         })

#     # 转换为最终输出的字典格式
#     output_data = {item['new_id']: item for item in processed}

#     # 写入JSON文件（逐步写入）
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write('{\n')
#         last_index = len(output_data)
#         for i, (key, value) in enumerate(output_data.items(), 1):
#             # 移除临时使用的new_id字段
#             del value['new_id']
            
#             entry = f'  "{key}": {json.dumps(value, ensure_ascii=False)}'
#             entry += ',' if i != last_index else ''
#             f.write(entry + '\n')
#         f.write('}')

#     print(f"处理完成！共处理 {len(output_data)} 个实体")
#     print(f"结果已保存至：{os.path.abspath(output_path)}")

# if __name__ == "__main__":
#     # 配置文件路径
#     input_json = "entity_texts.json"
#     output_json = "processed_entities.json"
    
#     # 运行处理流程
#     process_entities(input_json, output_json)
import json
import os
from operator import itemgetter

def process_entities(input_path, output_path):
    # 读取原始实体文件
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_entities = json.load(f)

    # 创建处理后的实体列表（保持顺序）
    processed = []
    
    # 提取并排序原始实体ID（降序排列）
    sorted_ids = sorted(raw_entities.keys(), 
                        key=lambda x: int(x))

    # 重新编号并处理字段
    for new_id, original_id in enumerate(sorted_ids, 0):
        entity = raw_entities[original_id]
        
        # 清洗字段数据
        name = entity.get('type.object.name', '')
        if name is not None:
            name = name.replace('"@en', '').strip('"')
        else:
            name = ''
        
        desc = entity.get('common.topic.description', '')
        if desc is not None:
            desc = desc.replace('"@en', '').strip('"')
        else:
            desc = ''

        # 构建新实体结构
        processed.append({
            "new_id": str(new_id),
            "original_id": original_id,
            "name": name,
            "description": desc
        })

    # 转换为最终输出的字典格式
    output_data = {item['new_id']: item for item in processed}

    # 写入JSON文件（逐步写入）
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('{\n')
        last_index = len(output_data)
        for i, (key, value) in enumerate(output_data.items(), 1):
            # 移除临时使用的new_id字段
            del value['new_id']
            
            entry = f'  "{key}": {json.dumps(value, ensure_ascii=False)}'
            entry += ',' if i != last_index else ''
            f.write(entry + '\n')
        f.write('}')

    print(f"处理完成！共处理 {len(output_data)} 个实体")
    print(f"结果已保存至：{os.path.abspath(output_path)}")

if __name__ == "__main__":
    # 配置文件路径
    input_json = "entity_texts.json"
    output_json = "entity_texts_processed.json"
    
    # 运行处理流程
    process_entities(input_json, output_json)