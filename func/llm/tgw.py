from func.log.default_log import DefaultLog
from func.config.default_config import defaultConfig
import requests
import json
from func.tools.singleton_mode import singleton

@singleton
class Tgw:
    # 设置控制台日志
    log = DefaultLog().getLogger()
    # 加载配置
    config = defaultConfig().get_config()

    tgw_url: str = config["llm"]["text-generation-webui"]["tgw_url"]
    history = []

    def remove_parenthesis_content(self, text):
    # 查找英文左括号 '(' 或中文左括号 '（'
        for bracket in ['(', '（']:
            pos = text.find(bracket)
            if pos > 0:  # 括号不在开头时才截取
                return text[:pos].rstrip()  # 去掉括号前的空白
        return text

    # text-generation-webui接口调用-LLM回复
    # mode:instruct/chat/chat-instruct  preset:Alpaca/Winlone(自定义)  character:角色卡Rengoku/Ninya
    def chat(self, messages, uid, username, character, relation):
        """
        发送聊天请求
        :param messages: 消息列表，格式 [{"role": "system"/"user"/"assistant", "content": "..."}]
        :param uid: 用户ID
        :param username: 用户名
        :param character: 角色设定（对应preset）
        :param relation: 关系（备用）
     :return: 助手回复文本
        """
        headers = {"Content-Type": "application/json"}

        # ---------- 消息清洗开始 ----------
        cleaned_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            # 确保 content 是字符串
            if not isinstance(content, str):
                content = str(content)
            # 防止 content 为空（某些 API 会拒绝空内容）
            if not content.strip():
                content = "..."  # 替换为占位符
            cleaned_messages.append({"role": role, "content": content})

        # 确保第一条消息不是 assistant（某些 API 要求第一条必须是 system 或 user）
        if cleaned_messages and cleaned_messages[0]["role"] == "assistant":
            cleaned_messages.insert(0, {"role": "system", "content": "你是一个虚拟主播，请自然地与用户对话。"})
        # ---------- 消息清洗结束 ----------
        self.log.info(f"最终发送的 messages: {json.dumps(cleaned_messages, ensure_ascii=False)}")
        data = {
            "mode": "chat",
            "character": username,
            "your_name": username,
            "messages": cleaned_messages,
            "preset": character,
            "do_sample": True,
            "max_new_tokens": 200,
            "seed": -1,
            "add_bos_token": True,
            "ban_eos_token": False,
            "skip_special_tokens": True,
            "instruction_template": "Alpaca",
        }
        try:
            response = requests.post(
                self.tgw_url, headers=headers, json=data, verify=False, timeout=(5, 300)
            )
        except Exception as e:
            self.log.exception(f"【{content}】信息回复异常")
            return "我听不懂你说什么"
        assistant_message = response.json()["choices"][0]["message"]["content"]
        assistant_message = self.remove_parenthesis_content(assistant_message)
        # history.append({"role": "assistant", "content": assistant_message})
        return assistant_message