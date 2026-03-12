import requests
import json
from func.log.default_log import DefaultLog
from func.config.default_config import defaultConfig
from func.tools.singleton_mode import singleton

@singleton
class AliyunLLM:
    log = DefaultLog().getLogger()
    config = defaultConfig().get_config()

    def __init__(self):
        aliyun_cfg = self.config.get("llm", {}).get("aliyun", {})
        self.api_key = aliyun_cfg.get("api_key", "")
        self.base_url = aliyun_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = aliyun_cfg.get("model", "qwen-plus")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages, **kwargs):
        """
        调用阿里云百炼兼容OpenAI接口
        :param messages: 消息列表，格式 [{"role": "user", "content": "..."}]
        :return: 回复内容字符串
        """
        if not self.api_key:
            self.log.error("阿里云百炼API Key未配置，无法生成摘要")
            return "摘要生成失败"

        url = f"{self.base_url}/chat/completions"
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 300
        }

        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            # 兼容OpenAI格式
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            self.log.exception("阿里云百炼摘要生成异常")
            return "摘要生成失败"