import os
from typing import List, Tuple
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from conf import get_project_root
from utils.logger_handler import get_logger

"""
MCP 客户端与工具集成器 (MCP Client Tools)
同时连接：
1. 【chef_core_service】: 自定义 Python 业务服务 (冰箱管理、买菜、营养查询、天气)
2. 【local_filesystem】: 官方 Node.js 文件系统服务 (读取 local_privacy 目录下的私密文件)

注意：local_privacy 中的文件不进入公网 RAG 向量库，
Agent 通过 MCP 的 read_file / list_directory 工具按需读取。
"""

logger = get_logger("ai_chef.mcp_client")


async def load_mcp_tools() -> Tuple[List[BaseTool], MultiServerMCPClient]:
    """
    异步加载所有 MCP 服务，并返回合并后的 LangChain 工具列表 + 客户端句柄。
    """
    project_root = get_project_root()
    custom_server_path = os.path.join(project_root, "agent", "mcp_server.py")
    privacy_dir = os.path.join(project_root, "local_privacy")

    os.makedirs(privacy_dir, exist_ok=True)

    servers_config = {
        # 服务 A：自定义业务逻辑 (Python)
        "chef_core_service": {
            "command": "python",
            "args": [custom_server_path],
            "transport": "stdio",
        },
        # 服务 B：官方通用文件系统服务 (Node.js)
        # 仅允许 Agent 安全地读取 local_privacy 目录下的私密文件
        "local_filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", privacy_dir],
            "transport": "stdio",
        },
    }

    logger.info("正在连接 MCP 微服务网络 (Python 业务服务 + 文件系统服务)...")

    client = MultiServerMCPClient(servers_config)
    tools = await client.get_tools()

    logger.info(f"成功从 MCP 网络加载了 {len(tools)} 个工具:")
    for tool in tools:
        logger.info(f"  - {tool.name}")

    return tools, client
