import os
import json
from langchain_core.messages import HumanMessage, AIMessage, messages_from_dict, messages_to_dict

from conf import get_project_root, get_agent_config
from utils.logger_handler import get_logger

"""
AI 私厨的状态与记忆管理器 (State & Memory Manager)
负责多轮对话状态的持久化、Session 隔离以及上下文窗口裁剪。
"""

logger = get_logger("ai_chef.state")


class StateManager:

    def __init__(self, memory_dir: str = None):
        if memory_dir is None:
            self.memory_dir = os.path.join(get_project_root(), "memory_sessions")
        else:
            self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)

    def _get_session_file(self, session_id: str) -> str:
        return os.path.join(self.memory_dir, f"{session_id}.json")

    def load_history(self, session_id: str) -> list:
        """加载指定用户的历史对话记录"""
        file_path = self._get_session_file(session_id)
        if not os.path.exists(file_path):
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                dicts = json.load(f)
                return messages_from_dict(dicts)
        except Exception as e:
            logger.warning(f"读取记忆异常，将开启全新对话: {e}")
            return []

    def save_history(self, session_id: str, messages: list):
        """将消息列表序列化并持久化到本地"""
        file_path = self._get_session_file(session_id)
        dicts = messages_to_dict(messages)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dicts, f, ensure_ascii=False, indent=2)

    def add_conversation(self, session_id: str, human_text: str, ai_text: str, max_keep: int = None):
        """向指定的 Session 中添加一轮新对话，并执行记忆裁剪。"""
        if max_keep is None:
            config = get_agent_config()
            max_keep = config["state"]["max_history_messages"]

        history = self.load_history(session_id)
        history.append(HumanMessage(content=human_text))
        history.append(AIMessage(content=ai_text))

        # 滑动窗口裁剪
        if len(history) > max_keep:
            history = history[-max_keep:]

        self.save_history(session_id, history)

    def clear_memory(self, session_id: str):
        """清除指定用户的记忆"""
        file_path = self._get_session_file(session_id)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"已清除用户 {session_id} 的所有记忆。")
