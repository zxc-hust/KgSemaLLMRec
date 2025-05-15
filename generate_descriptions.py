# # import json
# # import transformers
# # import torch
# # from tqdm import tqdm

# # # Initialize the text generation pipeline
# # model_path = "Meta-Llama-3-8B-Instruct"
# # pipeline = transformers.pipeline(
# #     "text-generation",
# #     model=model_path,
# #     model_kwargs={"torch_dtype": torch.bfloat16},
# #     device_map="auto",
# # )

# # # Define terminators for text generation
# # terminators = [
# #     pipeline.tokenizer.eos_token_id,
# #     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# # ]

# # # Load the entity data from the JSON file
# # with open('entity_texts_processed.json', 'r', encoding='utf-8') as f:
# #     entities = json.load(f)

# # # Open the output file in write mode (will overwrite if it exists)
# # with open('generated_descriptions.json', 'w', encoding='utf-8') as f:
# #     f.write('{\n')  # Start the JSON object

# #     # Process each entity and generate descriptive text
# #     first_entry = True
# #     # for key, entity_data in entities.items():
# #     for key, entity_data in tqdm(entities.items(), desc="Processing entities", total=len(entities)):
# #         # Clean up language tags from name and description
# #         entity_name = entity_data["name"].replace('"@en', '').replace('"@ja', '').replace('"@zh-TW', '').replace('"@fr', '').strip('"')
# #         entity_desc = entity_data["description"].replace('"@en', '').replace('"@ja', '').replace('"@zh-TW', '').replace('"@fr', '').strip('"')

# #         # Prepare the input messages for the model
# #         messages = [
# #             {
# #                 # "role": "system",
# #                 # "content": "You are a knowledge graph specialist. Your task is to generate comprehensive, coherent descriptive text for entities using their metadata. The output should be a fluent paragraph that:\n"
# #                 #            "1. Starts with the canonical name\n"
# #                 #            "2. Mentions the entity type\n"
# #                 #            "3. Incorporates the description naturally\n"
# #                 #            "4. Maintains factual accuracy\n"
# #                 #            "5. Avoids redundant phrases\n"
# #                 #            "Format: Plain text without markdown or special formatting."
# #                 "role": "system",
# #                 "content": """You are an encyclopedia editor. Generate entity descriptions that:
# #             1. Start directly with the canonical name
# #             2. Deduce entity type from context (e.g., novel, person, event)
# #             3. Extract key attributes from description:
# #             - For books: author, year, genre, significance
# #             - For people: occupation, achievements, nationality
# #             - For concepts: definition, applications
# #             4. Use only 3-5 concise sentences
# #             5. Maintain neutral tone without introductory phrases

# #             Bad: "Here is.../This entity describes..." 
# #             Good: "War and Peace is a historical novel by Leo Tolstoy (1869)..." 

# #             Structure:
# #             [Name] is a [deduced type] [optional subtype]. [Key attribute 1]. [Key attribute 2]. [Significance/unique aspect]."""
# #             },
# #             {
# #                 "role": "user",
# #                 # "content": f"Generate descriptive text for this entity using:\n"
# #                 #            f"Entity ID: {key}\n"
# #                 #            f"Name: {entity_name}\n"
# #                 #            f"Description: {entity_desc}"
# #                 "content": f"""Generate description using ONLY these fields:
# #             Name: {entity_name}
# #             Description: {entity_desc}"""
# #             }
# #         ]

# #         # Generate the descriptive text
# #         outputs = pipeline(
# #             messages,
# #             max_new_tokens=200,
# #             eos_token_id=terminators,
# #             do_sample=True,
# #             temperature=0.4,
# #             top_p=0.9,
# #             repetition_penalty=1.1
# #         )

# #         generated_text = outputs[0]["generated_text"][-1]["content"]

# #         # Write the entry to the file (streaming approach)
# #         if not first_entry:
# #             f.write(',\n')  # Add comma before all but the first entry
# #         first_entry = False
# #         f.write(f'  "{key}": {json.dumps(generated_text, ensure_ascii=False)}')

# #     f.write('\n}')  # Close the JSON object

# # print("Descriptive texts have been generated and saved to 'generated_descriptions.json'.")
# import json
# import transformers
# import torch
# from tqdm import tqdm
# import os

# # Initialize the text generation pipeline
# model_path = "Meta-Llama-3-8B-Instruct"
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_path,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )

# # Define terminators for text generation
# terminators = [
#     pipeline.tokenizer.eos_token_id,
#     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# # Load the entity data from the JSON file
# with open('entity_texts_processed.json', 'r', encoding='utf-8') as f:
#     entities = json.load(f)

# # --- Start of Modified Section ---
# # Check if generated_descriptions.json exists and load existing entries
# existing_entries = {}
# if os.path.exists('generated_descriptions.json'):
#     try:
#         with open('generated_descriptions.json', 'r', encoding='utf-8') as f:
#             existing_entries = json.load(f)
#     except json.JSONDecodeError:
#         print("Warning: generated_descriptions.json is empty or malformed. Starting from scratch.")

# # Get the set of already processed keys
# processed_keys = set(existing_entries.keys())

# # Filter entities to process only new ones
# new_entities = {k: v for k, v in entities.items() if k not in processed_keys}

# # If there are no new entities, exit early
# if not new_entities:
#     print("All entities have already been processed.")
# else:
#     # Open the output file in append mode
#     with open('generated_descriptions.json', 'a', encoding='utf-8') as f:
#         # If there are existing entries, write a comma to continue the JSON object
#         if existing_entries:
#             f.write(',\n')
#         else:
#             f.write('{\n')  # Start the JSON object if it didn't exist
# # --- End of Modified Section ---

#         # Process each new entity and generate descriptive text
#         first_entry = not existing_entries  # True if no existing entries
#         for key, entity_data in tqdm(new_entities.items(), desc="Processing new entities", total=len(new_entities)):
#             # Clean up language tags from name and description
#             entity_name = entity_data["name"].replace('"@en', '').replace('"@ja', '').replace('"@zh-TW', '').replace('"@fr', '').strip('"')
#             entity_desc = entity_data["description"].replace('"@en', '').replace('"@ja', '').replace('"@zh-TW', '').replace('"@fr', '').strip('"')

#             # Prepare the input messages for the model
#             messages = [
#                 {
#                     "role": "system",
#                     "content": """You are an encyclopedia editor. Generate entity descriptions that:
#                 1. Start directly with the canonical name
#                 2. Deduce entity type from context (e.g., novel, person, event)
#                 3. Extract key attributes from description:
#                 - For books: author, year, genre, significance
#                 - For people: occupation, achievements, nationality
#                 - For concepts: definition, applications
#                 4. Use only 3-5 concise sentences
#                 5. Maintain neutral tone without introductory phrases

#                 Bad: "Here is.../This entity describes..." 
#                 Good: "War and Peace is a historical novel by Leo Tolstoy (1869)..." 

#                 Structure:
#                 [Name] is a [deduced type] [optional subtype]. [Key attribute 1]. [Key attribute 2]. [Significance/unique aspect]."""
#                 },
#                 {
#                     "role": "user",
#                     "content": f"""Generate description using ONLY these fields:
#                 Name: {entity_name}
#                 Description: {entity_desc}"""
#                 }
#             ]

#             # Generate the descriptive text
#             outputs = pipeline(
#                 messages,
#                 max_new_tokens=200,
#                 eos_token_id=terminators,
#                 do_sample=True,
#                 temperature=0.4,
#                 top_p=0.9,
#                 repetition_penalty=1.1
#             )

#             generated_text = outputs[0]["generated_text"][-1]["content"]

#             # Write the entry to the file
#             if not first_entry:
#                 f.write(',\n')  # Add comma before all but the first new entry
#             first_entry = False
#             f.write(f'  "{key}": {json.dumps(generated_text, ensure_ascii=False)}')

#         # If there were new entries, close the JSON object
#         if new_entities:
#             f.write('\n}')  # Close the JSON object

# print("Descriptive texts have been generated and saved to 'generated_descriptions.json'.")

import json
import transformers
import torch
from tqdm import tqdm
import os

# 初始化文本生成管道
model_path = "Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# 定义文本生成的终止符
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# 从 JSON 文件加载实体数据
with open('entity_texts_processed.json', 'r', encoding='utf-8') as f:
    entities = json.load(f)

# 读取现有的条目（使用 JSON Lines 格式）
existing_entries = {}
if os.path.exists('generated_descriptions.json'):
    with open('generated_descriptions.json', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                key = entry["key"]
                description = entry["description"]
                existing_entries[key] = description
            except (json.JSONDecodeError, KeyError):
                print(f"警告：跳过格式错误的行：{line.strip()}")

# 过滤出需要处理的新实体
new_entities = {k: v for k, v in entities.items() if k not in existing_entries}

# 如果没有新实体，提前退出
if not new_entities:
    print("所有实体已处理完毕。")
else:
    # 以追加模式打开输出文件
    with open('generated_descriptions.json', 'a', encoding='utf-8') as f:
        # 处理每个新实体并生成描述文本
        for key, entity_data in tqdm(new_entities.items(), desc="处理新实体", total=len(new_entities)):
            # 清理名称和描述中的语言标签
            entity_name = entity_data["name"].replace('"@en', '').replace('"@ja', '').replace('"@zh-TW', '').replace('"@fr', '').strip('"')
            entity_desc = entity_data["description"].replace('"@en', '').replace('"@ja', '').replace('"@zh-TW', '').replace('"@fr', '').strip('"')

            # 准备模型的输入消息
            # messages = [
            #     {
            #         "role": "system",
            #         "content": """你是百科全书编辑。生成实体描述，要求：
            #     1. 直接以规范名称开头
            #     2. 从上下文中推断实体类型（如小说、人物、事件）
            #     3. 从描述中提取关键属性：
            #     - 书籍：作者、年份、体裁、重要性
            #     - 人物：职业、成就、国籍
            #     - 概念：定义、应用
            #     4. 使用 3-5 个简洁句子
            #     5. 保持中立语气，不使用引导性短语

            #     错误示例："这里是.../这个实体描述..." 
            #     正确示例："战争与和平是列夫·托尔斯泰的历史小说（1869）..." 

            #     结构：
            #     [名称] 是 [推断类型] [可选子类型]。[关键属性 1]。[关键属性 2]。[重要性/独特方面]。"""
            #     },
            #     {
            #         "role": "user",
            #         "content": f"""仅使用以下字段生成描述：
            #     名称：{entity_name}
            #     描述：{entity_desc}"""
            #     }
            # ]
            messages = [
                {
                    "role": "system",
                    "content": """You are an encyclopedia editor. Generate entity descriptions that:
                1. Start directly with the canonical name
                2. Deduce entity type from context (e.g., novel, person, event)
                3. Extract key attributes from description:
                - For books: author, year, genre, significance
                - For people: occupation, achievements, nationality
                - For concepts: definition, applications
                4. Use only 3-5 concise sentences
                5. Maintain neutral tone without introductory phrases

                Bad: "Here is.../This entity describes..." 
                Good: "War and Peace is a historical novel by Leo Tolstoy (1869)..." 

                Structure:
                [Name] is a [deduced type] [optional subtype]. [Key attribute 1]. [Key attribute 2]. [Significance/unique aspect]."""
                },
                {
                    "role": "user",
                    "content": f"""Generate description using ONLY these fields:
                Name: {entity_name}
                Description: {entity_desc}"""
                }
            ]

            # 生成描述文本
            outputs = pipeline(
                messages,
                max_new_tokens=200,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1
            )

            generated_text = outputs[0]["generated_text"][-1]["content"]

            # 创建条目并写入文件（每行一个 JSON 对象）
            entry = {"key": key, "description": generated_text}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print("描述文本已生成并保存至 'generated_descriptions.json'。")