"""工具模块 (Tools) - 连接外部系统和 API 的接口

对应培训课件中的「工具」概念：
  "连接外部系统和 API 的接口，让 Agent 有能力影响外部世界"

本模块定义了三个天气相关工具，供 Agent 自主决定是否调用。
"""

import requests
from langchain_core.tools import tool


@tool
def get_current_weather(location: str) -> str:
    """获取指定城市的当前天气信息，包括温度、湿度、风速、天气状况、降水量等。"""
    try:
        url = f"https://wttr.in/{location}?format=j1&lang=zh"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        cur = data["current_condition"][0]
        desc = cur.get("lang_zh", [{}])[0].get("value", cur["weatherDesc"][0]["value"])

        lines = [
            f"地点: {location}",
            f"天气: {desc}",
            f"温度: {cur['temp_C']}°C (体感 {cur['FeelsLikeC']}°C)",
            f"湿度: {cur['humidity']}%",
            f"风速: {cur['windspeedKmph']} km/h, 风向 {cur['winddir16Point']}",
            f"能见度: {cur['visibility']} km",
            f"降水量: {cur['precipMM']} mm",
            f"紫外线指数: {cur['uvIndex']}",
        ]

        today = data["weather"][0]
        lines.append(f"今日温度范围: {today['mintempC']}°C ~ {today['maxtempC']}°C")

        return "\n".join(lines)
    except Exception as e:
        return f"获取天气失败: {e}"


@tool
def get_weather_alerts(location: str) -> str:
    """获取指定城市的天气预警信息，根据当前气象数据判断是否存在恶劣天气风险。"""
    try:
        url = f"https://wttr.in/{location}?format=j1"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        cur = data["current_condition"][0]
        temp = int(cur["temp_C"])
        feels = int(cur["FeelsLikeC"])
        wind = int(cur["windspeedKmph"])
        vis = int(cur["visibility"])
        uv = int(cur["uvIndex"])
        precip = float(cur["precipMM"])

        alerts = []

        if temp >= 38:
            alerts.append(f"[红色高温] 气温 {temp}°C，严禁户外运动")
        elif temp >= 35:
            alerts.append(f"[橙色高温] 气温 {temp}°C，不建议户外跑步")
        elif temp <= -5:
            alerts.append(f"[低温预警] 气温 {temp}°C，注意防寒保暖")
        elif temp <= 0:
            alerts.append(f"[低温提示] 气温 {temp}°C，跑步时注意保暖")

        if abs(temp - feels) >= 5:
            alerts.append(f"[体感差异] 实际 {temp}°C，体感 {feels}°C，注意增减衣物")

        if wind >= 50:
            alerts.append(f"[大风预警] 风速 {wind} km/h，不宜户外运动")
        elif wind >= 30:
            alerts.append(f"[大风提示] 风速 {wind} km/h，逆风跑步较吃力")

        if vis <= 1:
            alerts.append(f"[低能见度] 能见度 {vis} km，跑步有安全隐患")
        elif vis <= 3:
            alerts.append(f"[能见度一般] 能见度 {vis} km，注意交通安全")

        if precip >= 10:
            alerts.append(f"[强降水] 降水量 {precip} mm，不宜户外跑步")
        elif precip > 0:
            alerts.append(f"[有降水] 降水量 {precip} mm，路面湿滑注意安全")

        if uv >= 8:
            alerts.append(f"[强紫外线] 指数 {uv}，必须做好防晒")
        elif uv >= 6:
            alerts.append(f"[紫外线较强] 指数 {uv}，建议涂抹防晒霜")

        return "\n".join(alerts) if alerts else "当前无天气预警，气象条件良好"

    except Exception as e:
        return f"获取预警失败: {e}"


@tool
def get_air_quality(location: str) -> str:
    """获取指定城市的空气质量指数 (AQI)，包括 PM2.5、PM10 等数据。"""
    try:
        url = f"https://wttr.in/{location}?format=j1"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        cur = data["current_condition"][0]

        if "air_quality" not in cur:
            return f"暂无 {location} 的详细空气质量数据，建议关注当地环保部门发布的 AQI 信息"

        aq = cur["air_quality"]
        pm25 = float(aq.get("pm2_5", 0))
        pm10 = float(aq.get("pm10", 0))
        us_epa = int(aq.get("us-epa-index", 0))

        level_map = {
            1: "优 (Good)",
            2: "良 (Moderate)",
            3: "轻度污染 (Unhealthy for Sensitive)",
            4: "中度污染 (Unhealthy)",
            5: "重度污染 (Very Unhealthy)",
            6: "严重污染 (Hazardous)",
        }
        level = level_map.get(us_epa, "未知")

        return (
            f"地点: {location}\n"
            f"PM2.5: {pm25:.1f} μg/m³\n"
            f"PM10: {pm10:.1f} μg/m³\n"
            f"AQI 等级: {level} (EPA Index: {us_epa})"
        )

    except Exception as e:
        return f"获取空气质量失败: {e}"
