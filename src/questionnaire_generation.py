import os
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from .config import AI_OUTPUT_CSV, TARGET_TOTAL
from tqdm import tqdm

# 使用数字编号作为唯一标识：QUESTION_DICT 的键为题号（字符串形式）
QUESTION_DICT = {}


def load_question_config():
    global QUESTION_DICT
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json_path = os.path.join(base_dir, "prompts", "question_list.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 构造映射：题目数字编号 => question info
    for q in data["questions"]:
        col_name = q["col_name"]
        m = re.match(r'^(\d+)', col_name)
        if m:
            q_num = m.group(1)  # 题号
            QUESTION_DICT[q_num] = q


def convert_single_answer(ans, qinfo):
    # 如果单选答案为数字，则转换为对应的字母（1->A, 2->B, ...）
    if re.match(r'^\d+$', ans):
        num = int(ans)
        if 1 <= num <= 26:
            return chr(ord('A') + num - 1)
    return ans


def convert_multiple_answer(ans):
    # 多选题答案可能包含多个选项，使用'、'或'┋'分隔
    tokens = re.split(r'[、┋]', ans)
    converted_tokens = []
    for token in tokens:
        token = token.strip()
        if re.match(r'^\d+$', token):
            num = int(token)
            if 1 <= num <= 26:
                token = chr(ord('A') + num - 1)
        converted_tokens.append(token)
    return "、".join(converted_tokens)


def convert_matrix_answer(ans, qinfo):
    # 对于量表题，如果答案是单个字母，则转换为对应数字（A->1, B->2, ...）
    if re.match(r'^[A-Z]$', ans):
        return str(ord(ans) - ord('A') + 1)
    return ans


def apply_question_logic(row_answers):
    """
    跳题规则：
    1) 若 第5题 为 "B"（否），则 6~36题全部置为 "(跳过)"
    2) 若 第35题 为 "B"，则 36题置为 "(跳过)"
    """
    key_q5 = "5"  # 题号5
    key_q35 = "35"  # 题号35
    q5_ans = None
    q35_ans = None
    for key, val in row_answers.items():
        m = re.match(r'^(\d+)', key)
        if m:
            num = m.group(1)
            if num == key_q5:
                q5_ans = val
            if num == key_q35:
                q35_ans = val
    if q5_ans == "B":
        for key in list(row_answers.keys()):
            m = re.match(r'^(\d+)', key)
            if m:
                num = m.group(1)
                if 6 <= int(num) <= 36:
                    row_answers[key] = "(跳过)"
    if q35_ans == "B":
        for key in list(row_answers.keys()):
            m = re.match(r'^(\d+)', key)
            if m and m.group(1) == "36":
                row_answers[key] = "(跳过)"
    return row_answers


def generate_questionnaires(df_raw, new_labels, persona_descs, model_wrapper):
    print("\n[5/7] 生成新问卷 (大模型) ...")
    load_question_config()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(base_dir, "prompts", "questionnaire_generation_system.txt"), "r", encoding="utf-8") as f:
        sys_prompt_raw = f.read()
    with open(os.path.join(base_dir, "prompts", "questionnaire_generation_user.txt"), "r", encoding="utf-8") as f:
        user_prompt_raw = f.read()

    # 构造问卷文本：逐题列出，包括题目及选项信息
    question_list_text = []
    for qnum, q in QUESTION_DICT.items():
        text_line = q["col_name"]
        if q["type"] in ["single", "multiple"]:
            options = q.get("options", [])
            if options:
                text_line += "\n选项：" + "、".join(options)
        question_list_text.append(text_line)

    full_prompt_template = (
            sys_prompt_raw + "\n\n" +
            user_prompt_raw
            .replace("{{ PERSONA_TEXT }}", "{{PERSONA_TEXT_PLACEHOLDER}}")
            .replace("{{ QUESTION_TEXT }}", "\n".join(question_list_text))
    )

    if not os.path.exists(AI_OUTPUT_CSV):
        pd.DataFrame(columns=df_raw.columns.tolist() + ["簇编号", "原问卷序号"]).to_csv(AI_OUTPUT_CSV, index=False)

    # 统计每个有效簇（new_labels != -1）的数量
    cluster_counts = Counter([lab for lab in new_labels if lab != -1])
    total_valid = sum(cluster_counts.values())
    target_counts = {}
    for lab in persona_descs:
        count = cluster_counts.get(lab, 0)
        target = int(round(TARGET_TOTAL * (count / total_valid))) if total_valid > 0 else 1
        target_counts[lab] = max(1, target)

    new_rows = []
    total_target = sum(target_counts.values())
    pbar = tqdm(total=total_target, desc="生成问卷进度")

    for lab in persona_descs:
        persona_json = persona_descs[lab]
        try:
            persona_dict = json.loads(persona_json)
            persona_text = "\n".join(f"{k}: {v}" for k, v in persona_dict.items())
        except:
            persona_text = persona_json
        full_prompt_final = full_prompt_template.replace("{{PERSONA_TEXT_PLACEHOLDER}}", persona_text)

        generated = 0
        while generated < target_counts[lab]:
            try:
                resp = model_wrapper.create_completion(
                    prompt=full_prompt_final,
                    temperature=0.15 + np.random.uniform(-0.05, 0.05),
                    max_tokens=2048
                )
                raw_text = resp["choices"][0]["text"].strip()
                raw_json = re.sub(r'```json|```', '', raw_text).strip()
                data = json.loads(raw_json)
                answers = data.get("answers", [])

                # 构建答案字典
                row_dict = {}
                for item in answers:
                    key = item["col_name"]
                    answer = item["answer"]
                    row_dict[key] = answer

                # 检查输出问卷中是否包含数字（作为题号）
                if not any(re.search(r'\d+', key) for key in row_dict.keys()):
                    print("【格式警告】该条输出问卷的题目中不含数字，已舍弃。")
                    continue  # 不计入生成数量

                # 构造标准化答案：按照 CSV 表头顺序匹配
                standardized_row = {}
                for col in df_raw.columns:
                    m = re.match(r'^(\d+)', col)
                    if m:
                        qnum = m.group(1)
                        found = False
                        for key in row_dict.keys():
                            m2 = re.match(r'^(\d+)', key)
                            if m2 and m2.group(1) == qnum:
                                standardized_row[col] = row_dict[key]
                                found = True
                                break
                        if not found:
                            standardized_row[col] = "(跳过)"
                    else:
                        standardized_row[col] = row_dict.get(col, "(跳过)")
                # 添加簇编号和原问卷序号
                standardized_row["簇编号"] = str(lab)
                standardized_row["原问卷序号"] = len(new_rows) + 1

                standardized_row = apply_question_logic(standardized_row)
                # 针对每个题目根据 QUESTION_DICT 的题型进行转换及校验
                for key in list(standardized_row.keys()):
                    m = re.match(r'^(\d+)', key)
                    if m:
                        qnum = m.group(1)
                        if qnum in QUESTION_DICT:
                            qinfo = QUESTION_DICT[qnum]
                            orig_ans = standardized_row[key]
                            if qinfo["type"] == "single":
                                converted_ans = convert_single_answer(orig_ans, qinfo)
                                standardized_row[key] = converted_ans
                                if not re.match(r'^[A-Z]$', converted_ans) and converted_ans != "(跳过)":
                                    print(
                                        f"[格式警告] 题号 {qnum} 单选题答案应为单个大写字母或(跳过)，实际: {converted_ans}")
                            elif qinfo["type"] == "multiple":
                                converted_ans = convert_multiple_answer(orig_ans)
                                standardized_row[key] = converted_ans
                                if converted_ans != "(跳过)" and not re.match(r'^[A-Z]([、┋][A-Z])*$', converted_ans):
                                    print(
                                        f"[格式警告] 题号 {qnum} 多选题答案应为多个字母组合或(跳过)，实际: {converted_ans}")
                            elif qinfo["type"] == "matrix_7":
                                converted_ans = convert_matrix_answer(orig_ans, qinfo)
                                standardized_row[key] = converted_ans
                                if converted_ans != "(跳过)" and not re.match(r'^[1-7]$', converted_ans):
                                    print(f"[格式警告] 题号 {qnum} 量表题答案应为1~7或(跳过)，实际: {converted_ans}")
                new_rows.append(standardized_row)
                pd.DataFrame([standardized_row]).to_csv(AI_OUTPUT_CSV, mode='a', header=False, index=False)
                generated += 1
                pbar.update(1)
            except Exception as e:
                print(f"生成问卷失败: {e}")
    pbar.close()
    return new_rows
