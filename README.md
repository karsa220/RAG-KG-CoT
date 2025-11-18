# 🧠 AcademicRAG: Retrieval-Augmented Research Assistant  
**一个基于 RAG + GLM-4.5 的科研论文智能检索与总结系统**

AcademicRAG 是一个面向科研工作者、学生与开发者的 AI 助手，能够：

1. 🔎 自动检索真实科研论文（OpenAlex / ArXiv / 本地库）  
2. 📚 构建向量索引（FAISS）进行语义检索  
3. 🧠 使用 GLM-4.5 进行 CoT 推理、总结、趋势分析  
4. 📈 输出研究热点、代表性论文、发展趋势以及可选题方向  

本系统可用于：

- 论文综述生成（Survey Assist）  
- 找选题 / 研究方向（Topic Discovery）  
- 调研某领域最新进展（State-of-the-Art Finder）  
- 推荐参考文献（Literature Recommendation）  
- 做科研调研任务的 Agent 基础模块  

---

# 🔥 功能特点

### 🔍 语义检索（RAG）
- OpenAlex/ArXiv 实时获取论文  
- 标题 + 摘要 自动向量化  
- FAISS 高效检索（毫秒级）  

### 🧠 GLM-4.5 CoT 智能总结
- 使用 ZhipuAI 官方 SDK  
- 支持深度思考模式（thinking=enabled）  
- 自动生成结构化科研报告：  
  - 研究热点  
  - 经典方法  
  - 最新趋势（2023–2025）  
  - 推荐论文  
  - 潜在未来研究方向  

### 🧪 Benchmark（基准评测）
- 支持 Precision@K, Recall@K, nDCG  
- 支持 Gold Summary 自动评估（ROUGE / BERTScore）  
- 支持科研检索任务的人工评测  

### 🧱 模块化 + 可扩展
- 可接入任意向量模型  
- 可接入任意大模型（GPT, Gemini, GLM）  
- 可扩展为自动科研 Agent（Planning → Retrieve → Validate）  

---

# 🏗️ 系统架构

User Query
↓
Paper Retriever (OpenAlex / ArXiv / Local)
↓
Embedding + FAISS Vector Search (Top-K)
↓
Evidence Pack (论文摘要 + 元数据)
↓
GLM-4.5 CoT Reasoning
↓
Final Structured Output (热点 / 论文 / 趋势 / 建议)

yaml
复制代码

---

# 🚀 快速开始（Quick Start）

## 1. 克隆项目
```bash
git clone https://github.com/xxx/AcademicRAG.git
cd AcademicRAG
