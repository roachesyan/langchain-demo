"""
LangChain 跑步天气顾问 Demo
=============================

演示 LangChain 四大核心模块（对应培训课件）:

  1. 工具 (Tools)   → 连接外部天气 API（见 weather_tools.py）
  2. 模型 (Model)   → 智谱 GLM-5.1 大语言模型
  3. 代理 (Agent)   → 自主决策调用哪些工具，综合分析给出跑步建议
  4. 记忆 (Memory)  → 支持多轮对话，记住之前聊过的内容

运行方式:
  pip install .
  python main.py
"""

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import ZHIPU_API_KEY, ZHIPU_BASE_URL, ZHIPU_MODEL, LOCATION
from weather_tools import get_current_weather, get_weather_alerts, get_air_quality

# ============================================================
# 1. 模型 (Model) - 初始化智谱 GLM-5.1
# ============================================================
# 通过 OpenAI 兼容接口接入智谱 AI，无需安装专属 SDK

llm = ChatOpenAI(
    model=ZHIPU_MODEL,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
    temperature=0.7,
)

# ============================================================
# 2. 工具 (Tools) - 注册给 Agent 使用的外部接口
# ============================================================

tools = [get_current_weather, get_weather_alerts, get_air_quality]

# ============================================================
# 3. 代理 (Agent) - 自主决策者
# ============================================================
# Agent 接收用户问题后，自主判断需要调用哪些工具，
# 然后综合所有工具返回的信息，给出最终建议。

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    # 系统提示词 - 定义 Agent 的角色和行为规范
    ("system", """你是一位专业的跑步天气顾问。

任务：根据用户所在城市的天气数据，判断今天是否适合户外跑步。

分析维度：
  - 温度：15~25°C 最佳；<5°C 或 >35°C 不建议
  - 湿度：40%~70% 舒适；>80% 闷热
  - 风速：<20 km/h 适宜；>40 km/h 危险
  - 降水：无雨最佳；中大雨不建议
  - 能见度：>5 km 安全；<2 km 有风险
  - 空气质量：AQI<100 良好；>150 不建议户外运动
  - 紫外线：指数>6 需防晒

输出格式：
  1. 总体评分：X / 10
  2. 适合跑步的因素 / 不适合的因素
  3. 跑步建议（推荐时段、装备、注意事项）

当前城市：{{location}}""".replace("{{location}}", LOCATION)),

    # 用户输入
    ("user", "{input}"),

    # Agent 工具调用的中间步骤（必须保留）
    MessagesPlaceholder(variable_name="agent_scratchpad"),

    # 对话历史（用于 Memory）
    MessagesPlaceholder(variable_name="chat_history"),
])

agent = create_tool_calling_agent(llm, tools, PROMPT_TEMPLATE)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
)

# ============================================================
# 4. 记忆 (Memory) - 跨对话保持上下文
# ============================================================
# 用一个简单的字典存储不同 session 的对话历史。
# 生产环境可替换为 Redis / 数据库等持久化存储。

_session_store: dict[str, ChatMessageHistory] = {}


def _get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ============================================================
# 运行入口
# ============================================================

def ask(question: str, session_id: str = "default") -> str:
    """向跑步顾问提问（支持多轮对话）"""
    response = agent_with_memory.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    return response["output"]


def run() -> None:
    """CLI 入口（pyproject.toml [project.scripts] 引用）"""
    print("=" * 60)
    print(f"  跑步天气顾问 - {LOCATION}")
    print("=" * 60)

    answer = ask("今天适合跑步吗？请帮我查一下天气、预警和空气质量。")
    print(f"\n{'=' * 60}")
    print("  跑步建议")
    print("=" * 60)
    print(answer)

    print(f"\n{'=' * 60}")
    print("  多轮对话演示 (Memory)")
    print("=" * 60)
    follow_up = ask("那如果改成傍晚跑步呢，时间上有什么建议？")
    print(follow_up)


if __name__ == "__main__":
    run()
