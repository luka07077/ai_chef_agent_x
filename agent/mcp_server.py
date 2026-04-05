import sys
import os
import json
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from fridge_manager import fridge_db
from fridge_manager.warning_system import check_expiring_items, check_allergen_conflict
from utils.logger_handler import log_tool_call
from conf import get_agent_config

"""
自定义 MCP 服务端
承载核心业务逻辑：冰箱管理、预警系统、生鲜下单、营养查询、天气查询。
所有配置项从 agent_config.yaml 读取，杜绝硬编码。
"""

config = get_agent_config()
mcp_config = config["mcp"]

mcp = FastMCP("AIChef_Core_Service")


@mcp.tool()
@log_tool_call
def get_fridge_inventory(user_id: str) -> str:
    """
    当需要知道用户冰箱里有什么食材、食材数量以及是否快过期时，调用此工具。

    Args:
        user_id: 用户的唯一标识符，如 "user_001"
    """
    inventory = fridge_db.get_active_inventory(user_id)
    if not inventory:
        return "冰箱里空空如也，没有任何可用食材。"
    return json.dumps(inventory, ensure_ascii=False, indent=2)


@mcp.tool()
@log_tool_call
def add_food_to_fridge(user_id: str, item_name: str, quantity: float, unit: str, days_to_expire: int = 7) -> str:
    """
    当需要向用户的虚拟冰箱中录入新食材时，调用此工具。
    支持去重合并：同名食材会累加数量而非重复添加。

    Args:
        user_id: 用户的唯一标识符
        item_name: 食材名称，如 "西红柿"
        quantity: 数量
        unit: 单位，如 "个"、"kg"
        days_to_expire: 预计保质期天数，默认7天
    """
    fridge_db.add_food_item(user_id, item_name, quantity, unit, days_to_expire)
    return f"已成功将 {item_name} {quantity}{unit} 录入冰箱，保质期 {days_to_expire} 天。"


@mcp.tool()
@log_tool_call
def check_fridge_warnings(user_id: str) -> str:
    """
    扫描冰箱中的临期/过期食材并生成预警报告。
    当用户询问"有什么快过期的"、"冰箱安全吗"，或在推荐菜谱前主动调用此工具检查。

    Args:
        user_id: 用户的唯一标识符
    """
    warnings = check_expiring_items(user_id)

    parts = []
    if warnings["expired"]:
        expired_names = [f"{i['item_name']}({i['quantity']}{i['unit']})" for i in warnings["expired"]]
        parts.append(f"【已过期】{', '.join(expired_names)}，建议立即丢弃！")

    if warnings["expiring_soon"]:
        soon_names = [f"{i['item_name']}(剩{i['days_left']}天)" for i in warnings["expiring_soon"]]
        parts.append(f"【即将过期】{', '.join(soon_names)}，建议优先使用！")

    if not parts:
        return "冰箱内所有食材状态良好，暂无预警。"

    return "\n".join(parts)


@mcp.tool()
@log_tool_call
def check_allergen_safety(user_id: str, ingredients: str) -> str:
    """
    致敏源安全拦截：检查一组食材是否包含用户的过敏原。
    在推荐菜谱或用户提到特定食材时，必须调用此工具进行安全检查。

    Args:
        user_id: 用户的唯一标识符
        ingredients: 逗号分隔的食材列表，如 "花生油,大蒜,鲜虾,白菜"
    """
    ingredient_list = [i.strip() for i in ingredients.split(",") if i.strip()]
    result = check_allergen_conflict(user_id, ingredient_list)

    if not result["is_safe"]:
        conflicts = ", ".join(result["conflicting_items"])
        return f"【安全预警】检测到致敏食材：{conflicts}！请立即更换这些食材，切勿使用！"

    return "安全检查通过，所有食材均不含用户的已知过敏原。"


@mcp.tool()
@log_tool_call
def order_fresh_groceries(item_name: str, quantity: float, unit: str) -> str:
    """
    当发现缺少食材时，调用此工具通过网络请求向外部供应商系统下单。

    Args:
        item_name: 需要购买的食材名称
        quantity: 购买数量
        unit: 购买单位
    """
    test_user = config["user"]["default_user_id"]
    api_url = mcp_config["order_api_url"]
    timeout = mcp_config["request_timeout"]

    payload = {
        "action": "place_order",
        "user_id": test_user,
        "items": [{"name": item_name, "qty": quantity, "unit": unit}]
    }

    try:
        response = requests.post(api_url, json=payload, timeout=timeout)

        if response.status_code == 200:
            default_days = config["fridge"]["default_expire_days"]
            fridge_db.add_food_item(test_user, item_name, quantity, unit, days_to_expire=default_days)
            return f"已通过网络请求成功下单：{item_name} {quantity}{unit}。数据已同步入库。"
        else:
            return f"供应商接口返回异常，状态码: {response.status_code}"

    except requests.exceptions.Timeout:
        return "下单失败：连接供应商接口超时，请稍后重试。"
    except requests.exceptions.RequestException as e:
        return f"下单失败：网络请求发生异常 ({str(e)})"


@mcp.tool()
@log_tool_call
def get_nutrition_info(food_name: str) -> str:
    """
    查询食材的营养成分信息（卡路里、蛋白质、脂肪、碳水）。

    Args:
        food_name: 食材名称，支持中文/英文 (例如: "西红柿", "tomato")
    """
    # api_key = os.environ.get("SPOONACULAR_API_KEY")
    api_key = mcp_config["nutrition_api_key"]
    api_base = mcp_config["nutrition_api_base"]
    timeout = mcp_config["request_timeout"]

    fallback_db = {
        "西红柿": "18 kcal/100g | 蛋白质0.9g, 脂肪0.2g, 碳水3.9g",
        "鸡蛋": "143 kcal/100g | 蛋白质12.6g, 脂肪9.9g, 碳水1.1g",
        "牛肉": "250 kcal/100g | 蛋白质26g, 脂肪17g, 碳水0g",
        "鸡胸肉": "165 kcal/100g | 蛋白质31g, 脂肪3.6g, 碳水0g",
        "大米": "130 kcal/100g | 蛋白质2.7g, 脂肪0.3g, 碳水28g",
        "牛奶": "42 kcal/100ml | 蛋白质3.4g, 脂肪1.5g, 碳水5g",
        "土豆": "77 kcal/100g | 蛋白质2g, 脂肪0.1g, 碳水17g",
    }

    def _fallback_lookup(name):
        for k, v in fallback_db.items():
            if k in name or name in k:
                return f"[本地数据] {k} 营养信息: {v}"
        return None

    if not api_key:
        result = _fallback_lookup(food_name)
        return result or f"未配置 SPOONACULAR_API_KEY 且本地无缓存，无法查询 {food_name} 的营养。"

    headers = {"User-Agent": "AIChef_Agent/1.0"}

    try:
        # ── Step 1：搜索接口，获取食材在 Spoonacular 数据库中的唯一 ID ─────────
        search_resp = requests.get(
            f"{api_base}/food/ingredients/search",
            params={"apiKey": api_key, "query": food_name, "number": 1},
            headers=headers,
            timeout=timeout
        )
        if search_resp.status_code != 200:
            raise Exception(f"搜索接口异常，HTTP {search_resp.status_code}")

        results = search_resp.json().get("results", [])
        if not results:
            return _fallback_lookup(food_name) or f"未查询到'{food_name}'的营养数据，可尝试英文名称。"

        ingredient_id   = results[0]["id"]
        ingredient_name = results[0]["name"]

        # ── Step 2：详情接口，用 ID 查询每 100g 的营养数据 ─────────────────────
        # 必须传 amount + unit，且加 nutrition=true，否则响应中不含 nutrition 字段
        detail_resp = requests.get(
            f"{api_base}/food/ingredients/{ingredient_id}/information",
            params={"apiKey": api_key, "amount": 100, "unit": "g", "nutrition": "true"},
            headers=headers,
            timeout=timeout
        )
        if detail_resp.status_code != 200:
            raise Exception(f"详情接口异常，HTTP {detail_resp.status_code}")

        # 详情接口返回字段极多，只取核心 4 项；用 lower() 做 key 匹配更稳健
        nutrients = detail_resp.json().get("nutrition", {}).get("nutrients", [])
        nd = {n["name"].lower(): round(n["amount"], 1) for n in nutrients}

        calories = nd.get("calories", "未知")
        protein  = nd.get("protein", "未知")
        fat      = nd.get("fat", "未知")
        carbs    = nd.get("carbohydrates", nd.get("net carbohydrates", "未知"))

        return (f"【Spoonacular】{food_name}（{ingredient_name}）每100g: "
                f"热量 {calories} kcal | 蛋白质 {protein}g | 脂肪 {fat}g | 碳水 {carbs}g")

    except Exception as e:
        return _fallback_lookup(food_name) or f"营养API请求异常: {str(e)}，且本地无 {food_name} 缓存数据。"


@mcp.tool()
@log_tool_call
def get_local_weather(city: str) -> str:
    """
    当需要根据天气情况推荐饮食、菜谱或汤品时，调用此工具查询实时天气。

    Args:
        city: 城市名称，例如 "北京", "Shanghai"
    """
    api_base = mcp_config["weather_api_base"]
    timeout = mcp_config["request_timeout"]

    try:
        response = requests.get(f"{api_base}/{city}?format=j1", timeout=timeout)

        if response.status_code == 200:
            data = response.json()
            cc = data['current_condition'][0]
            temp_c = cc['temp_C']
            desc = cc['lang_zh'][0]['value'] if 'lang_zh' in cc else cc['weatherDesc'][0]['value']
            feels_like = cc['FeelsLikeC']
            return f"【实时天气】{city}：{desc}，气温：{temp_c}°C（体感 {feels_like}°C）。"
        else:
            return f"天气服务暂时限流。请假设{city}当前为阴雨降温天气（气温15°C），并据此推荐。"

    except requests.exceptions.Timeout:
        return f"请求天气接口超时。请假设{city}当前为阴雨降温天气，并据此推荐。"
    except Exception as e:
        return f"查询天气时发生异常: {str(e)}"


if __name__ == "__main__":
    print("AIChef_Core_Service MCP 服务端正在启动 (基于 stdio)...", file=sys.stderr)
    mcp.run(transport='stdio')
