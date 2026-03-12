from func.log.default_log import DefaultLog
from func.config.default_config import defaultConfig
import ollama
import requests
import json
from func.tools.singleton_mode import singleton

@singleton
class Ollama:
    log = DefaultLog().getLogger()
    config = defaultConfig().get_config()

    # 从配置读取 Ollama 参数
    ollama_url: str = config["llm"]["ollama"]["ollama_url"]

    # 从配置读取 Ollama 参数
    ollama_url: str = config["llm"]["ollama"]["ollama_url"]
    use_cloud: bool = config["llm"]["ollama"].get("use_cloud", False)  # 新增开关

    # 根据 use_cloud 选择模型名
    if use_cloud:
        model: str = config["llm"]["ollama"].get("model_cloud", "deepseek-v3.1:671b-cloud")
    else:
        model: str = config["llm"]["ollama"].get("model_local", "qwen2.5")

    temperature: float = config["llm"]["ollama"].get("temperature", 0.7)
    top_p: float = config["llm"]["ollama"].get("top_p", 0.9)
    max_tokens: int = config["llm"]["ollama"].get("max_tokens", 200)
    if use_cloud:
        temperature = config["llm"]["ollama"].get("cloud_temperature", temperature)
        max_tokens = config["llm"]["ollama"].get("cloud_max_tokens", max_tokens)

    def generate(self, messages, options=None):
        """
        使用 messages 调用 Ollama chat 接口
        messages: 列表，格式 [{"role": "system"/"user"/"assistant", "content": "..."}]
        options: 可选参数字典
        """
        # 合并参数
        opt = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens  # 将 max_tokens 映射为 num_predict
        }
        if options:
            # 如果 options 中有 max_tokens，也映射到 num_predict
            if "max_tokens" in options:
                options["num_predict"] = options.pop("max_tokens")
            opt.update(options)

        # 添加停止词，防止模型生成多余内容（Qwen 模板使用 <|im_end|>）
        if "stop" not in opt:
            opt["stop"] = ["<|im_end|>"]

        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "options": opt,
                    "stream": False
                },
                timeout=(5, 300)
            )
            response.raise_for_status()
            result = response.json()
            assistant_message = result["message"]["content"]
            return assistant_message
        except Exception as e:
            self.log.exception(f"Ollama 生成异常：{e}")
            return "我听不懂你说什么"

    def generate_stream(self, messages, options=None):
        """
        流式生成，返回生成器，每次产出一个文本块
        """
        opt = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens
        }
        if options:
            if "max_tokens" in options:
                options["num_predict"] = options.pop("max_tokens")
            opt.update(options)

        if "stop" not in opt:
            opt["stop"] = ["<|im_end|>"]

        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "options": opt,
                    "stream": True
                },
                stream=True,
                timeout=(5, 300)
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]
                    if line == '[DONE]':
                        break
                    try:
                        data = json.loads(line)
                        if 'message' in data and 'content' in data['message']:
                            chunk = data['message']['content']
                            yield chunk
                    except json.JSONDecodeError:
                        self.log.warning(f"无法解析JSON行: {line}")
        except Exception as e:
            self.log.exception(f"Ollama 流式生成异常：{e}")
            yield ""  # 或抛异常让上层处理