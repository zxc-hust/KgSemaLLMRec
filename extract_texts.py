import os
import gzip
from tqdm import tqdm
import json
# ==============================
# 第一步：读取实体列表（带 remap_id）
# ==============================

mid2remap = {}  # m.xxx -> remap_id
non_mids = {}   # remap_id -> mid（记录不以 "m." 开头的 mid）

with open("datasets/amazon-book/entity_list.txt", "r", encoding="utf-8") as f:
    next(f)  # 跳过表头
    for line in f:
        # parts = line.strip().split()
        # if not parts:
        #     continue
        # mid = parts[0]
        # if mid.startswith("m.") or mid.startswith("g."):
        #     remap_id = parts[1] if len(parts) > 1 else mid  # 如果没有 remap_id 就用 mid
        #     mid2remap[mid] = remap_id
        # else:
        #     remap_id = parts[1] if len(parts) > 1 else mid  # 获取 remap_id（如果存在）
        #     non_mids[remap_id] = mid  # 用 remap_id 为键，mid 为值
        parts = line.strip().rsplit(maxsplit=1)  # ✅ 从行尾拆成两部分
        if not parts:
            continue
        if len(parts) == 2:
            mid, remap_id = parts
        else:
            mid = parts[0]
            remap_id = mid  # 如果没有 remap_id，就默认用 mid 自己

        if mid.startswith(("m.", "g.")):
            mid2remap[mid] = remap_id
        else:
            non_mids[remap_id] = mid  # 用 remap_id 为键，mid 为值

with open("non_mids.json", "w", encoding="utf-8") as nf:
    json.dump(non_mids, nf, ensure_ascii=False, indent=2)

# if os.path.exists("entity_texts.json"):
#     with open("entity_texts.json", "r", encoding="utf-8") as jf:
#         entity_texts = json.load(jf)
# else:
# 构造 Freebase URI
target_uris = set(f'<http://rdf.freebase.com/ns/{mid}>' for mid in mid2remap)

# ==============================
# 第二步：读取 RDF 文件并提取文本
# ==============================

entity_texts = {}
matched_count = 0  # 计数器

with gzip.open("freebase-rdf-latest.gz", "rt", encoding="utf-8") as f:
    progress = tqdm(f, desc="Processing RDF", unit="lines") 
    # for i, line in enumerate(tqdm(f, desc="Processing RDF", unit="lines")):
    for i, line in enumerate(progress):  # ✅ 迭代器改为 progress
        if not line.startswith('<http://rdf.freebase.com/ns/'):
            continue

        parts = line.strip().split('\t')
        if len(parts) != 4:
            continue

        subj, pred, obj, _ = parts
        if subj not in target_uris:
            continue

        # 取出 mid 并获取 remap_id
        mid = subj.rsplit('/', 1)[-1].rstrip(">")
        remap_id = mid2remap.get(mid)
        if remap_id is None:
            continue

        if remap_id not in entity_texts:
            # entity_texts[subj] = {"name": None, "description": None}
            entity_texts[remap_id] = {
                "mid": mid,
                "type.object.name": None,
                "common.topic.description": None
            }
            matched_count += 1  # 第一次遇到这个实体就加一
            
        # # name 和 description 通常是字符串字面量
        # if pred == '<http://rdf.freebase.com/ns/type.object.name>' and obj.startswith('"'):
        #     entity_texts[remap_id]["name"] = obj.strip('"')
        # elif pred == '<http://rdf.freebase.com/ns/common.topic.description>' and obj.startswith('"'):
        #     entity_texts[remap_id]["description"] = obj.strip('"')
        pred_key = pred.rsplit('/', 1)[-1].strip('>')  # 提取最后一段谓词（如 type.object.name）
        entity_texts[remap_id][pred_key] = obj.strip('"')

        # 每隔若干次更新一下 tqdm 后缀信息（可调节频率）
        if i % 10_000 == 0:
            progress.set_postfix(matched_entities=matched_count, total_targets=len(target_uris))
            
# 另外保存一份完整的 entity_texts 字典（JSON 格式）
with open("entity_texts.json", "w", encoding="utf-8") as jf:
    json.dump(entity_texts, jf, ensure_ascii=False, indent=2)

# ==============================
# 第三步：输出为 TSV
# ==============================

# with open("mid_texts_all.tsv", "w", encoding="utf-8") as out:
#     out.write("remap_id\tmid\tname\tdescription\n")
#     for uri, info in entity_texts.items():
#         mid = uri.split("/")[-1]
#         remap_id = mid2remap.get(mid, mid)
#         out.write(f"{remap_id}\t{mid}\t{info['name'] or ''}\t{info['description'] or ''}\n")


# ... 上面处理完 entity_texts 和 mid2remap 之后 ...

# 一份带 name / remap_id / mid / description 的 TSV
# with open("mid_texts.tsv", "w", encoding="utf-8") as out:
#     out.write("remap_id\tmid\tname\tdescription\n")
#     for uri, info in entity_texts.items():
#         # mid = uri.split("/")[-1]
#         mid = uri.split("/")[-1].rstrip(">")
#         remap_id = mid2remap.get(mid, "")      # 从 mapping 拿 remap_id
#         name = info.get("name") or ""         # 实体名称
#         description = info.get("description") or ""  # 实体描述
#         out.write(f"{remap_id}\t{mid}\t{name}\t{description}\n")



