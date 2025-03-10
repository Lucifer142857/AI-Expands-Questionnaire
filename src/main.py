import sys
import os

from . import config
from . import preprocess, clustering, persona_generation, questionnaire_generation, data_jitter, analysis
from .model_loader import load_model_for_4steps, load_model_for_question

def show_progress(name, i, total):
    bar_len = 50
    fill = int(round(bar_len * i / float(total)))
    perc = round(100.0 * i / float(total), 1)
    bar = "█" * fill + "-" * (bar_len - fill)
    print(f"{name}: |{bar}| {perc}% 完成", end="\r")
    if i == total:
        print()

def main():
    print("\n========== 大学生 AIGC 问卷扩充项目 ==========")

    # 1) 数据读取 & 预处理
    show_progress("步骤 1/7", 0, 1)
    df_raw, df_encoded, single_cols, multi_cols, scale_cols = preprocess.load_and_preprocess()
    show_progress("步骤 1/7", 1, 1)

    # 2) 层次聚类
    show_progress("步骤 2/7", 0, 1)
    X, k, _ = clustering.hierarchical_clustering(df_encoded)
    show_progress("步骤 2/7", 1, 1)

    # 3) 最终聚类
    show_progress("步骤 3/7", 0, 1)
    final_labels, cluster_centers = clustering.final_clustering(X, k)
    show_progress("步骤 3/7", 1, 1)

    # 4) 人物画像
    show_progress("步骤 4/7", 0, 1)
    model_4steps = load_model_for_4steps()
    persona_descs, new_labels = persona_generation.generate_personas(
        df_raw, X, final_labels, cluster_centers, k, model_4steps
    )
    show_progress("步骤 4/7", 1, 1)

    # 5) 生成问卷
    show_progress("步骤 5/7", 0, 1)
    model_question = load_model_for_question()
    ai_responses = questionnaire_generation.generate_questionnaires(
        df_raw, new_labels, persona_descs, model_question
    )
    show_progress("步骤 5/7", 1, 1)

    # 6) 数据抖动（根据配置决定是否执行）
    show_progress("步骤 6/7", 0, 1)
    if config.JITTER_ENABLED:
        jittered = data_jitter.data_jitter(
            df_raw, ai_responses, multi_cols, scale_cols, config.TARGET_TOTAL
        )
    else:
        jittered = ai_responses.copy()
    show_progress("步骤 6/7", 1, 1)

    # 7) 保存 & 分析
    show_progress("步骤 7/7", 0, 1)
    analysis.save_and_analyze(df_raw, ai_responses, jittered, final_labels, new_labels)
    show_progress("步骤 7/7", 1, 1)

    print("\n======= 流程结束，所有结果已保存到 outputs/ 文件夹 =======\n")

if __name__ == "__main__":
    main()
