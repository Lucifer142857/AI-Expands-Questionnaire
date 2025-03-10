import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from . import config

def cronbach_alpha(df_subset):
    """
    简易版 Cronbach's Alpha 计算
    """
    df_valid = df_subset.dropna(axis=0)
    k = df_valid.shape[1]
    if k < 2:
        return np.nan
    item_vars = df_valid.var(axis=0, ddof=1)
    total_var = df_valid.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return 0.0
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return alpha

def save_and_analyze(df_raw, ai_responses, jittered_responses, final_labels, new_labels):
    """
    将 AI 生成的问卷和抖动问卷分别保存，并做简单分析/打印。
    注意：原始问卷列放在前面，新增的元信息列（如“簇编号”、“原问卷序号”、“抖动来源”）附加在后面。
    """
    print("\n[7/7] 结果保存与分析 ...")

    # 转换为 DataFrame
    df_ai = pd.DataFrame(ai_responses)
    orig_cols = df_raw.columns.tolist()
    extra_cols = [col for col in df_ai.columns if col not in orig_cols]
    df_ai = df_ai[orig_cols + extra_cols]

    # 保存AI问卷
    df_ai.to_csv(config.AI_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"  AI 问卷已保存: {config.AI_OUTPUT_CSV} (行数={len(df_ai)})")

    # 保存抖动问卷
    df_jit = pd.DataFrame(jittered_responses)
    jit_extra_cols = [col for col in df_jit.columns if col not in orig_cols]
    df_jit = df_jit[orig_cols + jit_extra_cols]
    df_jit.to_csv(config.JITTER_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"  抖动问卷已保存: {config.JITTER_OUTPUT_CSV} (行数={len(df_jit)})")

    print("\n=== AI问卷示例(前2行) ===")
    print(df_ai.head(2))
    print("\n=== 抖动问卷示例(前2行) ===")
    print(df_jit.head(2))

    # 示例分析：各选项的占比变化
    # 这里假设仅对单选/多选题进行可视化
    # 你可根据实际需求自行扩展/修改
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    single_multi_cols = []
    for c in df_raw.columns:
        # 简易判断：若原始列是object类型，可能是单选/多选
        if df_raw[c].dtype == object:
            single_multi_cols.append(c)

    for col in single_multi_cols:
        orig_count = df_raw[col].replace("(跳过)", pd.NA).value_counts(normalize=True)
        ai_count = df_ai[col].replace("(跳过)", pd.NA).value_counts(normalize=True)
        compare_df = pd.DataFrame({"Original": orig_count, "AI": ai_count}).fillna(0)
        compare_df.plot.bar(rot=0, figsize=(6,4), title=f"{col} 选项分布对比")
        plt.ylabel("占比")
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, f"dist_compare_{col}.png"))
        plt.close()

    # 簇数量占比对比
    orig_labels_series = pd.Series([lab for lab in new_labels if lab != -1])
    orig_cluster_counts = orig_labels_series.value_counts(normalize=True)
    if "簇编号" in df_ai.columns:
        ai_cluster_counts = df_ai["簇编号"].astype(str).value_counts(normalize=True)
        cluster_compare = pd.DataFrame({"Original": orig_cluster_counts, "AI": ai_cluster_counts}).fillna(0)
        cluster_compare.plot.bar(rot=0, title="各簇样本占比（原始 vs AI）")
        plt.ylabel("占比")
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, "cluster_compare.png"))
        plt.close()

    # 信度系数（示例：使用AIGC困难题 + AIGC满意度题）
    difficulties_cols = ["10. 使用AIGC困难之『生成内容和需求不匹配』(1~7)",
                         "11. 使用AIGC困难之『操作界面复杂』(1~7)",
                         "12. 使用AIGC困难之『隐私及数据安全』(1~7)",
                         "13. 使用AIGC困难之『版权归属等伦理』(1~7)",
                         "14. 使用AIGC困难之『收费过高』(1~7)",
                         "15. 使用AIGC困难之『网络延迟或性能』(1~7)",
                         "16. 使用AIGC困难之『内容创新性不高』(1~7)",
                         "17. 使用AIGC困难之『担心过度依赖』(1~7)"]
    sat_cols = ["26. AIGC工具满意度之『输出内容质量』(1~7)",
                "27. AIGC工具满意度之『操作便捷性』(1~7)",
                "28. AIGC工具满意度之『响应速度』(1~7)",
                "29. AIGC工具满意度之『功能多样性』(1~7)",
                "30. AIGC工具满意度之『客户支持服务』(1~7)",
                "31. AIGC工具满意度之『安全隐私保护』(1~7)"]

    df_raw_diff = df_raw[difficulties_cols].replace("(跳过)", pd.NA).apply(pd.to_numeric, errors='coerce')
    df_ai_diff = df_ai[difficulties_cols].replace("(跳过)", pd.NA).apply(pd.to_numeric, errors='coerce')
    df_raw_sat = df_raw[sat_cols].replace("(跳过)", pd.NA).apply(pd.to_numeric, errors='coerce')
    df_ai_sat = df_ai[sat_cols].replace("(跳过)", pd.NA).apply(pd.to_numeric, errors='coerce')

    # 仅保留使用过AIGC工具的用户
    used_mask_orig = df_raw["5. 您是否使用过AIGC工具？ [单选题]*"] == "A. 是"
    used_mask_ai = df_ai["5. 您是否使用过AIGC工具？ [单选题]*"] == "A. 是"

    alpha_diff_orig = cronbach_alpha(df_raw_diff[used_mask_orig])
    alpha_diff_ai = cronbach_alpha(df_ai_diff[used_mask_ai])
    alpha_sat_orig = cronbach_alpha(df_raw_sat[used_mask_orig])
    alpha_sat_ai = cronbach_alpha(df_ai_sat[used_mask_ai])

    print(f"\n[信度] 使用AIGC困难题 Cronbach's α: 原始={alpha_diff_orig:.3f}, AI={alpha_diff_ai:.3f}")
    print(f"[信度] AIGC满意度题 Cronbach's α: 原始={alpha_sat_orig:.3f}, AI={alpha_sat_ai:.3f}")
    alpha_df = pd.DataFrame({
        "Original": [alpha_diff_orig, alpha_sat_orig],
        "AI": [alpha_diff_ai, alpha_sat_ai]
    }, index=["困难题组","满意度题组"])
    ax = alpha_df.plot(kind="bar", rot=0, title="Cronbach's Alpha 对比", ylim=(0,1))
    ax.set_ylabel("Alpha系数")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "alpha_compare.png"))
    plt.close()
