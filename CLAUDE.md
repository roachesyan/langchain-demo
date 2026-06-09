# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LangChain 跑步天气顾问 Demo — 培训教材项目。演示 LangChain 四大核心模块（Tools、Agent、Chain、Memory），结合智谱 AI GLM-5.1 模型，根据天气数据判断是否适合户外跑步。

## Commands

```bash
# 安装依赖
pip install .

# 运行
python main.py

# CLI 入口（通过 pyproject.toml [project.scripts] 注册）
running-advisor

# Docker
docker compose up --build
```

## Architecture

### Data Flow

```
User Input → Agent (decides which tools to call)
  → get_current_weather  (wttr.in API)
  → get_weather_alerts   (derived from weather data)
  → get_air_quality      (wttr.in air_quality field)
→ Agent synthesizes all tool results → formatted running advice
→ Memory stores conversation for multi-turn follow-up
```

### Key Components

- **`main.py`** — Agent + Chain + Memory: `ChatOpenAI` (Zhipu via OpenAI-compat API) → `create_tool_calling_agent` → `AgentExecutor` → `RunnableWithMessageHistory`. Entry point: `run()`.
- **`weather_tools.py`** — Three `@tool`-decorated functions: `get_current_weather`, `get_weather_alerts`, `get_air_quality`. All use `wttr.in` free API (no key needed).
- **`config.py`** — Loads `ZHIPU_API_KEY`, `ZHIPU_BASE_URL`, `ZHIPU_MODEL`, `LOCATION` from `.env` via `python-dotenv`.

### LLM Integration

Zhipu AI accessed through OpenAI-compatible endpoint (`https://open.bigmodel.cn/api/paas/v4/`) using `langchain-openai`'s `ChatOpenAI`. Model: `glm-5.1`.

## Configuration

All config lives in `.env` (not committed). Copy `.env.example` and fill in `ZHIPU_API_KEY`. `LOCATION` defaults to `重庆市渝中区`.
