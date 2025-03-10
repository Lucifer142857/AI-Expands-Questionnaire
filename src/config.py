import os

# === 基础路径 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "survey_data.csv")

# === 输出目录 ===
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

AI_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ai_simulated_responses.csv")
JITTER_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "jittered_responses.csv")

# 模型配置示例（可按需求切换本地模型或OpenAI接口）
USE_OPENAI_FOR_4STEPS = True
USE_OPENAI_FOR_QUESTION = True

MODEL_PATH_4STEPS = os.path.join(BASE_DIR, "models", "Mistral-7B-Instruct-v0.3.Q5_K_M.gguf")
MODEL_PATH_QUESTION = os.path.join(BASE_DIR, "models", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf")

DEEPSEEK_API_KEY = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
OPENAI_MODEL_NAME = "deepseek-reasoner"

N_THREADS = 16
USE_GPU_LAYERS = 9999  # 视显存情况

# 问卷扩充参数
TARGET_TOTAL = 1000    # AI 模拟问卷生成的目标数量

# 是否对 AI 生成的问卷进行抖动处理（True：抖动；False：不抖动）
JITTER_ENABLED = True

# 可选：要忽略分布可视化的列
IGNORE_COLS = []
