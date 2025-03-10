# Survey Clustering and Augmentation Project

本项目针对「大学生 AIGC 使用情况调查问卷」数据，提供完整的处理、分析与智能扩充方案。

## 核心流程

1. **数据读取与预处理**  
   - 自动识别单选、多选、量表题并进行编码
2. **层次聚类分析**  
   - 使用层次聚类（Ward法）进行初步分析，支持交互式选择聚类数
3. **优化聚类**  
   - 对比 KMeans++ 与 GMM 聚类效果，基于轮廓系数选择最佳方案
4. **用户画像生成**  
   - 结合大模型判断画像要点，基于聚类中心样本生成典型人物画像
5. **问卷生成**  
   - 根据画像和示例答案生成多样化问卷回答
6. **数据增强**  
   - 实施随机扰动，应用问卷逻辑约束，扩展至目标数量
7. **结果分析**  
   - 保存至 `data/augmented_survey_data.csv`，提供可视化对比

## 目录结构

survey_clustering_project/
├── data/
│   ├── survey_data.csv          # 原始问卷数据
│   └── augmented_survey_data.csv # 增强后数据
├── models/
│   └── model.gguf               # 本地大模型文件
├── src/
│   ├── __init__.py
│   ├── config.py                # 参数配置
│   ├── preprocess.py            # 数据预处理
│   ├── clustering.py            # 聚类分析
│   ├── persona_generation.py    # 画像生成
│   ├── questionnaire_generation.py # 问卷生成
│   ├── data_jitter.py           # 数据扰动
│   ├── analysis.py              # 结果分析
│   └── main.py                  # 主流程
├── README.md
└── requirements.txt

```
## 快速开始

### 准备工作
1. 准备数据文件：
   - 原始问卷保存为 `data/survey_data.csv`
   - 参考格式：CSV 编码 UTF-8
2. 模型配置：
   - 将 GGUF 格式模型文件放置于 `models/model.gguf`
3. 参数调整：
   ```python
   # config.py 示例配置
   N_THREADS = 8                 # CPU 线程数
   USE_GPU_LAYERS = 20           # GPU 加速层数
   NEW_PER_ORIGINAL = 5          # 每条原始数据生成5条新数据
```

### 安装依赖

```bash
python -m venv venv
source venv/bin/activate         # Linux/macOS
venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

### 运行项目

```bash
python src/main.py
```

程序将显示进度条和大模型生成过程，完整运行时间约15-30分钟（取决于数据量）

## 结果查看

生成数据路径：`data/augmented_survey_data.csv`  
包含以下增强特征：

- 原始数据标记（`is_original` 字段）
- 扰动后的多样化回答
- 符合逻辑的跳题处理

## 问卷逻辑约束

```python
# 在 data_jitter.py 中实现的核心逻辑
if 第5题 == "否":
    第6～22题 = "(跳过)"
if 第23题 == "否":
    第24题 = "(跳过)"
```

## 常见问题

**Q1: 模型加载失败**  
✅ 检查项：  

- 确认模型文件路径正确  
- 检查 config.py 中的线程配置  
- 验证 GPU 驱动兼容性  

**Q2: 如何调整生成数量**  
修改配置参数：  

```python
NEW_PER_ORIGINAL = 10    # 每条原始生成10条
TARGET_TOTAL = 1000      # 总目标样本量
```

**Q3: 输出格式异常**  
在 `questionnaire_generation.py` 中强化提示词：

```python
PROMPT_TEMPLATE = f"""
请严格按照以下格式生成回答：
1. 单选题用[选项]标注
2. 量表题用1-7数字
3. 多选题用分号分隔
{示例答案}...
"""
```

**Q4: 性能优化建议**  

- 启用 GPU 加速层（需6GB+显存）
- 增加 `N_THREADS` 至物理核心数
- 使用量化版模型（如 Q4_K_M 量化）

> 更多技术细节请参考各模块代码注释
