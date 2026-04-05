import os
import yaml

"""
配置加载器：统一读取 YAML 配置文件，供项目各模块使用。
"""

_CONF_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CONF_DIR)

# 缓存，避免重复读取
_agent_config = None
_prompt_config = None


def get_project_root() -> str:
    return _PROJECT_ROOT


def get_agent_config() -> dict:
    global _agent_config
    if _agent_config is None:
        path = os.path.join(_CONF_DIR, "agent_config.yaml")
        with open(path, "r", encoding="utf-8") as f:
            _agent_config = yaml.safe_load(f)
    return _agent_config


def get_prompt_config() -> dict:
    global _prompt_config
    if _prompt_config is None:
        path = os.path.join(_CONF_DIR, "prompt_config.yaml")
        with open(path, "r", encoding="utf-8") as f:
            _prompt_config = yaml.safe_load(f)
    return _prompt_config
