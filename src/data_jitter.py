import numpy as np
import random
import re
import copy
import json
import os

from .questionnaire_generation import apply_question_logic

def load_allowed_options():
    """
    读取 prompts/question_list.json 文件，构造映射：
    题号（字符串形式） -> 允许的选项字母列表（仅对单选和多选题有效）
    例如："1" -> ["A", "B"]
    """
    allowed = {}
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json_path = os.path.join(base_dir, "prompts", "question_list.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for q in data["questions"]:
        col_name = q["col_name"].strip()
        m = re.match(r'^(\d+)', col_name)
        if m and q["type"] in ["single", "multiple"]:
            qnum = m.group(1)
            opts = q.get("options", [])
            letters = []
            for opt in opts:
                # 假设选项格式为 "A. xxx"
                m2 = re.match(r'^([A-Z])\.', opt.strip())
                if m2:
                    letters.append(m2.group(1))
            allowed[qnum] = letters
    return allowed

def data_jitter(df_raw, ai_responses, multi_cols, scale_cols, target_total):
    """
    对 AI 生成的问卷进行抖动：
      - 逐条处理每个 AI 问卷，依据一定概率随机调整（可能不作任何改动），
      - 处理后的问卷数量与 AI 问卷数量一致，
      - 将抖动后的问卷存入新的 CSV（与 AI 问卷 CSV 行数一致）。
    """
    print("\n[6/7] 数据抖动 ...")
    allowed_options = load_allowed_options()
    jittered_list = []

    for row in ai_responses:
        new_row = copy.deepcopy(row)

        # === 处理第五题逻辑 ===
        q5_col = None
        for col in new_row:
            if re.match(r'^5(\D|$)', col):  # 匹配以5开头的列名
                q5_col = col
                break
        if q5_col:
            q5_value = new_row.get(q5_col, '').strip()
            # 若第五题为 A，则有 5% 概率改为 B，并将 6~36题置为 "(跳过)"
            if q5_value == 'A' and np.random.rand() < 0.05:
                new_row[q5_col] = 'B'
                for col_j in new_row:
                    if re.match(r'^(6|7|8|9|\d{2,})', col_j):  # 题号 ≥6 的列
                        new_row[col_j] = '(跳过)'
                new_row = apply_question_logic(new_row)
                new_row["抖动来源"] = f"原问卷{new_row.get('原问卷序号', '?')}-5题A变B"
                jittered_list.append(new_row)
                continue  # 此问卷已处理，进入下一条
            # 若第五题为 B，则将 6~36题全部置为 "(跳过)"
            elif q5_value == 'B':
                for col_j in new_row:
                    if re.match(r'^(6|7|8|9|\d{2,})', col_j):
                        new_row[col_j] = '(跳过)'
                new_row = apply_question_logic(new_row)
                new_row["抖动来源"] = f"原问卷{new_row.get('原问卷序号', '?')}-5题B处理"
                jittered_list.append(new_row)
                continue

        # === 处理第三十五题逻辑 ===
        q35_col = None
        for col in new_row:
            if re.match(r'^35(\D|$)', col):
                q35_col = col
                break
        if q35_col and new_row.get(q35_col, '').strip() not in ['(跳过)']:
            q35_value = new_row.get(q35_col, '').strip()
            if q35_value in ['A', 'B'] and np.random.rand() < 0.10:
                new_val = 'B' if q35_value == 'A' else 'A'
                new_row[q35_col] = new_val
                if new_val == 'B':
                    for col_j in new_row:
                        if re.match(r'^36(\D|$)', col_j):
                            new_row[col_j] = '(跳过)'
                    new_row = apply_question_logic(new_row)
                    new_row["抖动来源"] = f"原问卷{new_row.get('原问卷序号', '?')}-35题翻转为B"
                    jittered_list.append(new_row)
                    continue
                else:
                    q36_cols = [col for col in new_row if re.match(r'^36(\D|$)', col)]
                    allowed = allowed_options.get('36', [])
                    if allowed:
                        chosen = random.choice(allowed)
                        for col_j in q36_cols:
                            new_row[col_j] = chosen

        # === 对除特殊题外的其它题进行随机抖动 ===
        for col in df_raw.columns:
            # 跳过不需处理的列
            if col in ["簇编号", "原问卷序号"] or re.match(r'^(5|35|36)', col):
                continue
            old_val = new_row.get(col, "")
            # 多选题处理：以 0.2 的概率修改答案
            if col in multi_cols and old_val not in ["(跳过)"]:
                if np.random.rand() < 0.2:
                    parts = re.split(r'[;；,|、┋]+', str(old_val))
                    parts = [p.strip() for p in parts if p.strip()]
                    if parts and np.random.rand() < 0.5 and len(parts) > 1:
                        parts.pop(np.random.randint(len(parts)))
                    else:
                        m_q = re.match(r'^(\d+)', col)
                        allowed = allowed_options.get(m_q.group(1), []) if m_q else []
                        available = [a for a in allowed if a not in parts]
                        if available:
                            parts.append(random.choice(available))
                    new_row[col] = "、".join(sorted(set(parts)))
            # 量表题处理：以 0.2 的概率对数值做 ±1 调整（保持在 1～7 范围内）
            elif col in scale_cols and old_val not in ["(跳过)"]:
                if np.random.rand() < 0.2:
                    try:
                        ival = int(old_val)
                        ival += random.choice([-1, 1])
                        ival = max(1, min(7, ival))
                        new_row[col] = str(ival)
                    except:
                        pass
            # 单选题处理：以 0.1 的概率随机修改答案，确保在允许选项范围内且与原答案不同
            else:
                m_q = re.match(r'^(\d+)', col)
                if m_q:
                    qnum = m_q.group(1)
                    if qnum in allowed_options and old_val not in ["(跳过)"]:
                        # 仅处理未归入多选和量表题的单选题
                        if (col not in multi_cols) and (col not in scale_cols):
                            if np.random.rand() < 0.1:
                                opts = allowed_options.get(qnum, [])
                                if old_val in opts:
                                    possible = [x for x in opts if x != old_val]
                                else:
                                    possible = opts
                                if possible:
                                    new_row[col] = random.choice(possible)
                                    if "抖动来源" in new_row:
                                        new_row["抖动来源"] += f"; 单选题{col}抖动"
                                    else:
                                        new_row["抖动来源"] = f"原问卷{new_row.get('原问卷序号', '?')}-单选题{col}抖动"
        new_row = apply_question_logic(new_row)
        if "抖动来源" not in new_row:
            new_row["抖动来源"] = f"原问卷{new_row.get('原问卷序号', '?')}-随机抖动"
        jittered_list.append(new_row)

    print(f"  抖动后问卷数: {len(jittered_list)}")
    return jittered_list
