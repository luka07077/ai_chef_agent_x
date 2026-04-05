import sqlite3
import os
from datetime import datetime, timedelta

from conf import get_project_root, get_agent_config
from utils.logger_handler import get_logger, log_tool_call

"""
虚拟冰箱数据库模块 (Virtual Fridge Database)
管理食材生命周期和用户偏好，数据统一存放在 data/fridge.db。
"""

logger = get_logger("ai_chef.fridge")

# 数据库路径：统一存放在项目根目录的 data/ 下
DB_PATH = os.path.join(get_project_root(), "data", "fridge.db")


def _get_conn():
    """获取数据库连接（确保 data 目录存在）"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表结构"""
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            item_name TEXT NOT NULL,
            quantity REAL NOT NULL,
            unit TEXT NOT NULL,
            add_date TEXT NOT NULL,
            expiration_date TEXT NOT NULL,
            status INTEGER DEFAULT 0
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS preferences (
            user_id TEXT PRIMARY KEY,
            allergies TEXT,
            dietary_goals TEXT
        )
    ''')

    conn.commit()
    conn.close()
    logger.info(f"虚拟冰箱数据库初始化完成，路径: {DB_PATH}")


@log_tool_call
def add_food_item(user_id: str, item_name: str, quantity: float, unit: str, days_to_expire: int):
    """
    向虚拟冰箱中添加食材（UPSERT 逻辑：同名且未耗尽的食材会累加数量并刷新保质期）。
    """
    conn = _get_conn()
    cursor = conn.cursor()

    add_date = datetime.now().strftime("%Y-%m-%d")
    expiration_date = (datetime.now() + timedelta(days=days_to_expire)).strftime("%Y-%m-%d")

    # 查找是否已有同名、同单位、未耗尽的食材
    cursor.execute('''
        SELECT id, quantity FROM inventory
        WHERE user_id = ? AND item_name = ? AND unit = ? AND status = 0
    ''', (user_id, item_name, unit))
    existing = cursor.fetchone()

    if existing:
        # UPSERT：累加数量，刷新保质期为更晚的那个
        new_qty = existing["quantity"] + quantity
        cursor.execute('''
            UPDATE inventory
            SET quantity = ?, expiration_date = MAX(expiration_date, ?), add_date = ?
            WHERE id = ?
        ''', (new_qty, expiration_date, add_date, existing["id"]))
        logger.info(f"食材合并: {item_name} -> 数量更新为 {new_qty}{unit}")
    else:
        cursor.execute('''
            INSERT INTO inventory (user_id, item_name, quantity, unit, add_date, expiration_date, status)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        ''', (user_id, item_name, quantity, unit, add_date, expiration_date))
        logger.info(f"新食材入库: {item_name} {quantity}{unit}")

    conn.commit()
    conn.close()


@log_tool_call
def consume_food_item(user_id: str, item_name: str, quantity: float = None):
    """
    消耗食材：减少数量或标记为耗尽。
    如果不指定 quantity，则直接将该食材标记为耗尽。
    """
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, quantity FROM inventory
        WHERE user_id = ? AND item_name = ? AND status = 0
        ORDER BY expiration_date ASC LIMIT 1
    ''', (user_id, item_name))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return False

    if quantity is None or quantity >= row["quantity"]:
        cursor.execute('UPDATE inventory SET status = 1 WHERE id = ?', (row["id"],))
    else:
        new_qty = row["quantity"] - quantity
        cursor.execute('UPDATE inventory SET quantity = ? WHERE id = ?', (new_qty, row["id"]))

    conn.commit()
    conn.close()
    return True


def get_active_inventory(user_id: str) -> list:
    """获取指定用户冰箱里所有未耗尽的食材（包括已过期的，供 UI 和预警系统使用）"""
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT item_name, quantity, unit, expiration_date
        FROM inventory
        WHERE user_id = ? AND status = 0
        ORDER BY expiration_date ASC
    ''', (user_id,))

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_user_preferences(user_id: str, allergies: str, dietary_goals: str):
    """更新用户的偏好设置（如忌口和健康目标）"""
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute('''
        REPLACE INTO preferences (user_id, allergies, dietary_goals)
        VALUES (?, ?, ?)
    ''', (user_id, allergies, dietary_goals))

    conn.commit()
    conn.close()


def get_user_preferences(user_id: str) -> dict:
    """获取用户的偏好设置"""
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute('SELECT allergies, dietary_goals FROM preferences WHERE user_id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return {"allergies": "无", "dietary_goals": "无"}


# 初始化：确保表结构存在
init_db()


if __name__ == "__main__":
    test_user = "user_001"

    # 测试 UPSERT：连续添加同名食材，验证不会重复
    add_food_item(test_user, "鸡蛋", 10, "个", 14)
    add_food_item(test_user, "鸡蛋", 5, "个", 14)  # 应累加为 15
    add_food_item(test_user, "牛肉", 0.5, "kg", 3)
    add_food_item(test_user, "西红柿", 3, "个", 5)

    update_user_preferences(test_user, allergies="花生,海鲜", dietary_goals="高蛋白,减脂")

    print(f"\n--- {test_user} 的虚拟冰箱库存 ---")
    for item in get_active_inventory(test_user):
        print(f"  {item['item_name']}: {item['quantity']}{item['unit']}, 过期: {item['expiration_date']}")

    print(f"\n--- {test_user} 的偏好设置 ---")
    print(get_user_preferences(test_user))
