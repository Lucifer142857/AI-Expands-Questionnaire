import os
import numpy as np
import pandas as pd
import json
import re

def decide_persona_aspects(df_raw, model):
    print("\n  正在让大模型判断适合输出哪些人物画像维度(以JSON形式)...")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(base_dir, "prompts", "persona_aspect_system.txt"), "r", encoding="utf-8") as f:
        system_prompt = f.read()
    with open(os.path.join(base_dir, "prompts", "persona_aspect_user.txt"), "r", encoding="utf-8") as f:
        user_template = f.read()

    # 将列名拼成一个大字符串替换 {{ HEADERS }}
    headers_text = "\n".join(df_raw.columns.tolist())
    user_prompt = user_template.replace("{{ HEADERS }}", headers_text)

    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    output = model.create_completion(prompt=full_prompt, max_tokens=1024, temperature=0.7)
    raw_response = output["choices"][0]["text"].strip()
    cleaned_response = re.sub(r'```json|```', '', raw_response).strip()
    aspects_text = cleaned_response
    print("  [大模型输出的Persona Aspects in JSON]")
    print(aspects_text)
    return aspects_text

def generate_personas(df_raw, X, final_labels, cluster_centers, k, model):
    """
    1) 先通过 decide_persona_aspects 得到 JSON 格式的维度
    2) 再为每簇选1条代表问卷, 让模型生成 "典型用户画像" (JSON)
    3) 让用户交互式输入要去除的簇编号 => 返回 (new_persona_descs, new_labels)
    """
    print("\n[4/7] 生成典型用户画像 ...")
    aspects_json = decide_persona_aspects(df_raw, model)

    dimension_list = []
    try:
        data = json.loads(aspects_json)
        if isinstance(data, dict):
            dimension_list = [
                *data.get("base_attributes", []),
                *data.get("cognitive_traits", []),
                *data.get("behavior_patterns", [])
            ]
    except:
        # fallback
        dimension_list = ["年级", "专业类别", "对AIGC了解程度", "其它"]

    # 找每簇代表
    cluster_reps = {}
    for ci in range(k):
        idxs = np.where(final_labels == ci)[0]
        if len(idxs) == 0:
            continue
        subset = X[idxs].astype(float)
        center = cluster_centers[ci].astype(float)
        dists = np.linalg.norm(subset - center, axis=1)
        rep_idx = idxs[np.argmin(dists)]
        cluster_reps[ci] = df_raw.iloc[rep_idx]

    # 用提示词生成人物画像
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(base_dir, "prompts", "persona_generation_system.txt"), "r", encoding="utf-8") as f:
        sys_prompt = f.read()
    with open(os.path.join(base_dir, "prompts", "persona_generation_user.txt"), "r", encoding="utf-8") as f:
        user_template = f.read()

    persona_map_temp = {}
    for ci, rep in cluster_reps.items():
        idxs = np.where(final_labels == ci)[0]  # 重新获取该簇样本索引
        print(f"\n  正在生成簇 {ci} 的画像...")
        lines = [f"{col}: {val}" for col, val in rep.items()]
        rep_text = "\n".join(lines)

        dims_json_str = json.dumps({
            "base_attributes": dimension_list[:3],
            "cognitive_traits": dimension_list[3:6],
            "behavior_patterns": dimension_list[6:9]
        }, ensure_ascii=False)
        cluster_profile = f"这是第{ci}个聚类，样本量={len(idxs)}"

        user_prompt_filled = user_template \
            .replace("{{ DIMENSION_SCHEMA }}", dims_json_str) \
            .replace("{{ CLUSTER_PROFILE }}", cluster_profile) \
            .replace("{{ SAMPLE_ANSWERS }}", rep_text)

        full_prompt = f"{sys_prompt}\n\n{user_prompt_filled}"
        output = model.create_completion(prompt=full_prompt, max_tokens=512, temperature=0.3)
        raw_persona_response = output["choices"][0]["text"].strip()
        cleaned_persona_response = re.sub(r'```json|```', '', raw_persona_response).strip()
        persona_text = cleaned_persona_response
        print("[大模型输出的画像 (JSON)]")
        print(persona_text)
        persona_map_temp[ci] = persona_text

    # 用户交互：输入要去除的簇编号（用空格分隔，直接回车则不去除）
    print("\n=== 所有簇画像生成完毕 ===")
    sorted_keys = sorted(persona_map_temp.keys())
    for ci in sorted_keys:
        print(f"簇 {ci}:")
        print(persona_map_temp[ci])
        print("------------------------------------------------")

    remove_input = input("\n请输入想去除的簇编号(用空格分隔，如 1 3 5，直接回车则不去除): ").strip()
    remove_ids = set()
    for s in remove_input.split():
        try:
            idx = int(s)
            if idx in persona_map_temp:
                remove_ids.add(idx)
        except:
            pass

    new_labels = []
    drop_count = 0
    for lab in final_labels:
        if lab in remove_ids:
            new_labels.append(-1)
            drop_count += 1
        else:
            new_labels.append(lab)

    new_persona_descs = {}
    for cid in persona_map_temp:
        if cid not in remove_ids:
            new_persona_descs[cid] = persona_map_temp[cid]

    print(f"\n去除的簇: {sorted(remove_ids)}, 剩余簇: {sorted(new_persona_descs.keys())}, 去除数量: {drop_count}")
    return new_persona_descs, new_labels
