"""配置模块 - 从 .env 加载所有配置"""

import os
from dotenv import load_dotenv

load_dotenv()

# 智谱 AI 配置
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-5.1")

# 地点配置
LOCATION = os.getenv("LOCATION", "重庆市渝中区")
