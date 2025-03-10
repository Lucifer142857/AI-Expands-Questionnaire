import pandas as pd
import numpy as np
import re
import json
import os

from .config import DATA_PATH, BASE_DIR


def load_question_list():
    """
    读取 prompts/question_list.json 文件，返回按顺序排列的 col_name 列表
    """
    json_path = os.path.join(BASE_DIR, "prompts", "question_list.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 返回 JSON 文件中所有题目的 col_name，注意使用 strip() 清除前后空格
    col_names = [q["col_name"].strip() for q in data["questions"]]
    return col_names


def load_question_types():
    """
    读取 prompts/question_list.json 文件，构造映射：
    题号（字符串形式） -> 题型（"single"、"multiple"、"matrix_7"）
    题号采用 JSON 中 col_name 前面的数字
    """
    question_types = {}
    json_path = os.path.join(BASE_DIR, "prompts", "question_list.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for q in data["questions"]:
        col_name = q["col_name"].strip()
        m = re.match(r'^(\d+)', col_name)
        if m:
            qnum = m.group(1)
            question_types[qnum] = q["type"]
    return question_types


def load_and_preprocess():
    print("\n[1/7] 读取与预处理数据 ...")
    # 注意根据实际情况调整 CSV 的编码，比如 "gbk"、"utf-8-sig" 等
    df_raw = pd.read_csv(DATA_PATH, encoding="gbk")
    print(f"  原始数据量: {df_raw.shape[0]}, 列: {df_raw.shape[1]}")

    # 统一表头：将 CSV 的列名替换为 JSON 文件中定义的 col_name
    json_cols = load_question_list()
    if len(json_cols) != df_raw.shape[1]:
        print(f"[警告] JSON 中题目数量({len(json_cols)})与 CSV 列数({df_raw.shape[1]})不一致，使用 CSV 原始表头。")
    else:
        df_raw.columns = json_cols
        print("  成功将 CSV 表头替换为 JSON 中定义的题目名称。")

    # 清洗空行并重置索引
    df_raw.dropna(how='all', inplace=True)
    df_raw.reset_index(drop=True, inplace=True)

    # 根据 JSON 中题型确定各列类型（利用 CSV 列的顺序进行匹配）
    question_types = load_question_types()
    single_cols = []
    multi_cols = []
    scale_cols = []
    # 假设 CSV 的列顺序与 JSON 中题目顺序一致，第 i 列对应题号 i（1-indexed）
    for idx, col in enumerate(df_raw.columns):
        qnum = str(idx + 1)
        if qnum in question_types:
            qtype = question_types[qnum]
            if qtype == "single":
                single_cols.append(col)
            elif qtype == "multiple":
                multi_cols.append(col)
            elif qtype == "matrix_7":
                scale_cols.append(col)
            else:
                single_cols.append(col)
        else:
            single_cols.append(col)

    print(f"  清洗后: {df_raw.shape[0]} 条问卷")
    print("  单选列:", single_cols)
    print("  多选列:", multi_cols)
    print("  量表列:", scale_cols)

    # 根据题型对数据进行编码
    df_encoded = pd.DataFrame()
    for col in df_raw.columns:
        if col in multi_cols:
            all_opts = set()
            for val in df_raw[col].dropna():
                parts = re.split(r'[;,|、┋]+', str(val))
                all_opts.update(p.strip() for p in parts if p.strip())
            for opt in sorted(all_opts):
                df_encoded[f"{col}::{opt}"] = df_raw[col].fillna("").apply(
                    lambda x: 1 if opt in re.split(r'[;,|、┋]+', str(x)) else 0
                )
        elif col in scale_cols:
            df_encoded[col] = pd.to_numeric(
                df_raw[col].replace(r"\(跳过\)", np.nan, regex=True),
                errors='coerce'
            ).fillna(0)
        else:
            if pd.api.types.is_numeric_dtype(df_raw[col]):
                df_encoded[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
            else:
                uniq_vals = df_raw[col].unique()
                if len(uniq_vals) < 50:
                    dummies = pd.get_dummies(df_raw[col].fillna("NA"), prefix=col)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                else:
                    df_encoded[col] = pd.factorize(df_raw[col])[0]

    return df_raw, df_encoded, single_cols, multi_cols, scale_cols
