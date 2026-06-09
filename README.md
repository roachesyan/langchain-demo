# LangChain 跑步天气顾问 Demo

基于 LangChain 四大核心模块（Chain、Agent、Tools、Memory），结合智谱 GLM-5.1 模型，根据天气、预警、空气质量判断今天是否适合户外跑步。

## 前置条件

- Python >= 3.11
- 智谱 AI API Key（[获取地址](https://open.bigmodel.cn/)）

## 快速开始

### 方式一：本地运行

```bash
# 1. 复制配置文件并填入 API Key
cp .env.example .env

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install .

# 4. 运行
python main.py
```

### 方式二：Docker 运行

```bash
# 1. 复制配置文件并填入 API Key
cp .env.example .env

# 2. 构建并运行
docker compose up --build
```

## 配置说明

编辑 `.env` 文件：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ZHIPU_API_KEY` | 智谱 AI API Key（必填） | - |
| `ZHIPU_BASE_URL` | 智谱 API 地址 | `https://open.bigmodel.cn/api/paas/v4/` |
| `ZHIPU_MODEL` | 模型名称 | `glm-5.1` |
| `LOCATION` | 查询城市 | `重庆市渝中区` |

## 项目结构

```
├── .env.example         # 配置模板
├── pyproject.toml       # 项目依赖
├── Dockerfile           # Docker 构建
├── docker-compose.yml   # Docker 编排
├── config.py            # 配置加载
├── weather_tools.py     # 工具模块（天气、预警、空气质量）
└── main.py              # 主程序（Agent + Memory）
```

## LangChain 四大模块对应关系

| 模块 | 文件 | 说明 |
|------|------|------|
| **Tools** | `weather_tools.py` | 3 个外部 API 工具：当前天气、天气预警、空气质量 |
| **Agent** | `main.py` | 自主决策调用哪些工具，综合分析后给出跑步建议 |
| **Chain** | `main.py` | AgentExecutor 编排 工具调用→数据分析→建议输出 的流水线 |
| **Memory** | `main.py` | 支持多轮对话，追问时能记住之前的上下文 |
