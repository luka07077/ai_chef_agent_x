import datetime

from fridge_manager import fridge_db
from conf import get_agent_config
from utils.logger_handler import get_logger

"""
预警系统模块 (Warning System)
处理业务闭环中的"预警"环节：临期食材扫描、致敏源安全拦截。
"""

logger = get_logger("ai_chef.warning")


def check_expiring_items(user_id: str, warning_days: int = None) -> dict:
    """
    检查指定用户冰箱中的临期或已过期食材。

    Returns:
        dict: 包含 'expired' 和 'expiring_soon' 食材列表
    """
    if warning_days is None:
        config = get_agent_config()
        warning_days = config["fridge"]["warning_days"]

    inventory = fridge_db.get_active_inventory(user_id)

    result = {"expired": [], "expiring_soon": []}
    today = datetime.datetime.now()

    for item in inventory:
        exp_date = datetime.datetime.strptime(item['expiration_date'], "%Y-%m-%d")
        delta_days = (exp_date - today).days

        if delta_days < 0:
            result["expired"].append(item)
        elif 0 <= delta_days <= warning_days:
            item_copy = dict(item)
            item_copy['days_left'] = delta_days
            result["expiring_soon"].append(item_copy)

    return result


def check_allergen_conflict(user_id: str, proposed_ingredients: list) -> dict:
    """
    致敏源安全拦截检查。

    Args:
        user_id: 用户标识
        proposed_ingredients: 计划使用的食材名称列表

    Returns:
        dict: 包含 'is_safe' 和 'conflicting_items'
    """
    prefs = fridge_db.get_user_preferences(user_id)
    allergies_str = prefs.get("allergies", "")

    if not allergies_str or allergies_str == "无":
        return {"is_safe": True, "conflicting_items": []}

    user_allergies = [a.strip() for a in allergies_str.split(",")]
    conflicts = []

    for ingredient in proposed_ingredients:
        for allergy in user_allergies:
            if allergy in ingredient:
                conflicts.append(ingredient)
                break

    if conflicts:
        logger.warning(f"致敏源拦截: 用户 {user_id}, 冲突食材: {conflicts}")

    return {
        "is_safe": len(conflicts) == 0,
        "conflicting_items": conflicts
    }


if __name__ == "__main__":
    test_user = "user_001"

    print("--- 1. 临期食材预警 ---")
    warnings = check_expiring_items(test_user)
    print(f"  临期: {[i['item_name'] for i in warnings['expiring_soon']]}")
    print(f"  过期: {[i['item_name'] for i in warnings['expired']]}")

    print("\n--- 2. 致敏源安全拦截 ---")
    check = check_allergen_conflict(test_user, ["花生碎", "猪肉"])
    if not check["is_safe"]:
        print(f"  拦截！致敏食材: {check['conflicting_items']}")
    else:
        print("  安全，未发现致敏源。")
