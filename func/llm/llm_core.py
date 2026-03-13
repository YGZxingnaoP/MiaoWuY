# 聊天功能
from func.log.default_log import DefaultLog
import re
import threading
import random
import json
import uuid
import os
import atexit
from threading import Thread
from func.config.default_config import defaultConfig
from func.tools.singleton_mode import singleton
from func.obs.obs_init import ObsInit
from func.tools.string_util import StringUtil
from func.vtuber.action_oper import ActionOper
from func.tts.tts_core import TTsCore
from func.llm.fastgpt import FastGpt
from func.llm.tgw import Tgw
from func.gobal.data import LLmData
from func.gobal.data import CommonData
from func.llm.memory import MemoryManager
from func.llm.aliyun import AliyunLLM
from func.llm.mem0.memory_manager import Mem0Manager
from func.llm.ollama_llm import Ollama
from func.llm.character import CharacterCard

@singleton
class LLmCore:
    # 设置控制台日志
    log = DefaultLog().getLogger()
    commonData = CommonData()
    llmData = LLmData()  # llm数据

    actionOper = ActionOper()  # 动作
    ttsCore = TTsCore()  #语音

    # 选择大语言模型
    local_llm_type: str = llmData.local_llm_type
    if local_llm_type == "fastgpt":
        fastGpt = FastGpt()
        llm = fastGpt
    elif local_llm_type == "text-generation-webui":
        tgw = Tgw()
        llm = tgw
    elif local_llm_type == "ollama":          # 新增 ollama 分支
        ollama = Ollama()
        llm = ollama
    else:
        fastGpt = FastGpt()
        llm = fastGpt

    def __init__(self):
        self.config = defaultConfig().get_config()
        self.obs = ObsInit().get_ws()
        self.memory_managers = {}
        self.mem0_managers = {}
        self.memory_trigger_keywords = [
            "记得", "知道", "还记得", "记不记得", "忘记", "回忆", "上次", "以前", "聊过", "说过", "提到过", "讨论过", "你记得吗",  "你还记得", "那个事", "那件事", "那个谁"
        ]
        self.aliyun_llm = AliyunLLM()
        ollama_config = self.config.get('llm', {}).get('ollama', {})
        self.ollama_stream = ollama_config.get('stream', False)
        
        # ====== 新增：预设回复缓存 ======
        self.preset_responses = {}  # 关键词 -> 回复列表
        self.load_preset_responses()
        # ====== 新增：连续“不要说话”检测 ======
        self.last_msg_contain_dont_speak = False

        # ====== 新增：统一摘要生成函数（供记忆管理器使用）======
        self._summary_generator = self._create_summary_generator()
        self._pause_timer = None
        atexit.register(self._cleanup)

    def _cleanup(self):
        if self._pause_timer:
            self._pause_timer.cancel()

    def _create_summary_generator(self):
        """返回一个摘要生成函数，供记忆管理器使用"""
        def generator(dialogue_text):
            prompt = f"为以下对话生成一段详细的中文摘要。摘要应准确概括对话的核心内容和关键信息，以小猫娘“喵呜”的身份描述，着重强调人物身份，禁止添加对话中没有的内容。200字左右\n对话：{dialogue_text}\n总结："
            messages = [{"role": "user", "content": prompt}]
            summary = self.aliyun_llm.chat(messages)
            summary = summary.strip()
            if len(summary) > 300:
                summary = summary[:300] + "…"
            return summary
        return generator

    # ====== 新增：加载预设回复 ======
    def load_preset_responses(self):
        preset_dir = "./chatpreset"
        if not os.path.exists(preset_dir):
            os.makedirs(preset_dir, exist_ok=True)
            self.log.info(f"创建预设目录: {preset_dir}")
        for filename in os.listdir(preset_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(preset_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            for key, replies in data.items():
                                if isinstance(replies, list):
                                    if key in self.preset_responses:
                                        self.preset_responses[key].extend(replies)
                                    else:
                                        self.preset_responses[key] = replies
                        self.log.info(f"加载预设文件: {filename}")
                except Exception as e:
                    self.log.error(f"加载预设文件失败 {filename}: {e}")

    # ====== 新增：确保记忆管理器存在并返回 ======
    def _ensure_memory_manager(self, uid_str, username=None):
        """
        根据当前 LLM 类型，获取或创建对应的记忆管理器。
        返回管理器对象，如果不支持记忆则返回 None。
        """
        if self.local_llm_type == "text-generation-webui":
            if uid_str not in self.memory_managers:
                self.memory_managers[uid_str] = MemoryManager(
                    uid=uid_str,
                    long_term_dir="chatrecords",
                    max_pending_rounds=10,
                    short_term_rounds=3,
                    summary_generator=self._summary_generator
                )
            return self.memory_managers[uid_str]
        elif self.local_llm_type == "ollama":
            if uid_str not in self.mem0_managers:
                self.mem0_managers[uid_str] = Mem0Manager(
                    uid=uid_str,
                    max_pending_rounds=10,
                    short_term_rounds=3,
                    summary_generator=self._summary_generator,
                    shared_user_id="shared"
                )
            return self.mem0_managers[uid_str]
        else:
            # fastgpt 等不支持记忆
            return None

    def should_use_long_term_memory(self, message: str) -> bool:
        """判断用户消息是否包含触发记忆的关键词"""
        message_lower = message.lower()  # 如果需要大小写不敏感
        for keyword in self.memory_trigger_keywords:
            if keyword in message:
                return True
            # 新增：未命中关键词且长度 > 15 时，50% 概率触发
        if len(message) > 15:
            return random.random() < 0.5
        return False

    # LLM回复
    def aiResponseTry(self):
        try:
            self.ai_response()
        except Exception as e:
            self.log.exception(f"【ai_response】发生了异常：")
            self.llmData.is_ai_ready = True

    # LLM回复
    def ai_response(self):
        """
        从问题队列中提取一条，生成回复并存入回复队列中
        :return:
        """
        self.llmData.is_ai_ready = False
        llm_json = self.llmData.QuestionList.get()
        # 参数提取
        uid = llm_json["uid"]
        username = llm_json["username"]
        prompt = llm_json["prompt"]
        traceid = llm_json["traceid"]

        # 用户查询标题
        title = prompt
        # query有值是搜索任务，没有值是聊天任务
        if "query" in llm_json:
            # 搜索任务的查询字符，在query字段
            title = llm_json["query"]
            self.obs.show_text("状态提示", f'{self.llmData.Ai_Name}搜索问题"{title}"')
        else:
            self.obs.show_text("状态提示", f'{self.llmData.Ai_Name}思考问题"{title}"')

        # 身份判定
        relation = self.llmData.relations.get(username)
        if relation is None:
            relation = "粉丝"

        # fastgpt
        if self.local_llm_type == "fastgpt":
            # ... 保持不变 ...
            self.log.info(f"[{traceid}]{prompt}")
            character = "撒娇版"
            if re.search(self.llmData.public_sentiment_key, prompt):
                character = "陪聊版"
            else:
                random_number = random.randrange(1, 11)
                if random_number > 4:
                    character = "撒娇版"
                else:
                    character = "陪聊版"
            response = self.llm.chat(prompt, uid, username, character, relation)
        # text-generation-webui
        elif self.local_llm_type == "text-generation-webui":
            # ... 保持不变（但可以使用 _ensure_memory_manager 简化，这里不修改以保持稳定）...
            self.log.info(f"[{traceid}]{prompt}")

            uid_str = str(uid)
            if uid_str not in self.memory_managers:
                def summary_gen(dialogue_text):
                    prompt = f"为以下对话生成一段详细的中文摘要。摘要应准确概括对话的核心内容和关键信息，以小猫娘“喵呜”的身份描述，着重强调人物身份，禁止添加对话中没有的内容。\n对话：{dialogue_text}\n总结："
                    messages = [{"role": "user", "content": prompt}]
                    summary = self.aliyun_llm.chat(messages)
                    summary = summary.strip()
                    if len(summary) > 220:
                        summary = summary[:220] + "…"
                    return summary

                self.memory_managers[uid_str] = MemoryManager(
                    uid=uid_str,
                    long_term_dir="chatrecords",
                    max_pending_rounds=10,
                    short_term_rounds=3,
                    summary_generator=summary_gen

                )
            memory = self.memory_managers[uid_str]

            memory.add_user_message(prompt, username)

            use_long_term = self.should_use_long_term_memory(prompt)
            if use_long_term:
                messages = memory.build_messages(prompt, include_long_term=True)
                self.log.info(f"[{traceid}] 触发长期记忆，加载背景")
            else:
                messages = memory.build_messages(prompt, include_long_term=False)
                self.log.info(f"[{traceid}] 未触发长期记忆，仅使用短期记忆")

            response = self.llm.chat(messages, uid, "喵呜", "MiaoWu", relation)
            response = response.replace("You", username)

            memory.add_assistant_message(response)

            all_content = response

            import re
            split_chars = self.llmData.split_str
            pattern = '|'.join(re.escape(c) for c in split_chars)
            segments = re.split(f'({pattern})', all_content)
            raw_sentences = []
            for i in range(0, len(segments)-1, 2):
                raw_sentences.append(segments[i] + (segments[i+1] if i+1 < len(segments) else ''))
            if len(segments) % 2 == 1:
                raw_sentences.append(segments[-1])
            raw_sentences = [seg.strip() for seg in raw_sentences if seg.strip()]

            min_len = self.llmData.split_limit
            merged_segments = []
            current = ""
            for sent in raw_sentences:
                if not current:
                    current = sent
                else:
                    if len(current) < min_len:
                        current += sent
                    else:
                        merged_segments.append(current)
                        current = sent
            if current:
                merged_segments.append(current)

            if not merged_segments:
                merged_segments = [all_content]

            total = len(merged_segments)
            for idx, seg in enumerate(merged_segments):
                if total == 1:
                    status = "end"
                else:
                    if idx == 0:
                        status = "start"
                    elif idx == total - 1:
                        status = "end"
                    else:
                        status = ""
                jsonStr = {
                    "voiceType": "chat",
                    "traceid": traceid,
                    "chatStatus": status,
                    "question": title if idx == 0 else "",
                    "text": seg,
                    "lanuage": "AutoChange",
                    "seg_index": idx,
                    "total_segments": total
                }
                self.llmData.AnswerList.put(jsonStr)
                self.log.info(f"[{traceid}]分段{idx+1}/{total}: {seg}")

            if "粉色" in all_content or "睡觉" in all_content or "粉红" in all_content or "房间" in all_content or "晚上" in all_content:
                self.actionOper.changeScene("粉色房间")
            elif "清晨" in all_content or "早" in all_content or "睡醒" in all_content:
                self.actionOper.changeScene("清晨房间")
            elif "祭拜" in all_content or "神社" in all_content or "寺庙" in all_content:
                self.actionOper.changeScene("神社")
            elif "花房" in all_content or "花香" in all_content:
                self.actionOper.changeScene("花房")
            elif "岸" in all_content or "海" in all_content:
                self.actionOper.changeScene("海岸花坊")

            current_question_count = self.llmData.QuestionList.qsize()
            self.log.info(f"[{traceid}][AI回复]{all_content}")
            self.log.info(f"[{traceid}]System>>[{username}]的回复已存入队列，当前剩余问题数:{current_question_count}")

            self.llmData.is_ai_ready = True
            return

        # ========== ollama 分支 ==========
        elif self.local_llm_type == "ollama":
            # ... 保持不变 ...
            self.log.info(f"[{traceid}]{prompt}")

            uid_str = str(uid)
            if uid_str not in self.mem0_managers:
                def summary_gen(dialogue_text):
                    prompt = f"为以下对话生成一段详细的中文摘要。摘要应准确概括对话的核心内容和关键信息，以小猫娘“喵呜”的身份描述，着重强调人物身份，禁止添加对话中没有的内容。200字左右\n对话：{dialogue_text}\n总结："
                    messages = [{"role": "user", "content": prompt}]
                    summary = self.aliyun_llm.chat(messages)
                    summary = summary.strip()
                    if len(summary) > 300:
                        summary = summary[:300] + "…"
                    return summary
                self.mem0_managers[uid_str] = Mem0Manager(
                    uid=uid_str,
                    max_pending_rounds=10,
                    short_term_rounds=3,
                    summary_generator=summary_gen,
                    shared_user_id="shared"
                )
            memory = self.mem0_managers[uid_str]
            memory.add_user_message(prompt, username)

            use_long_term = self.should_use_long_term_memory(prompt)
            if use_long_term:
                messages = memory.build_messages(prompt, username=username, include_long_term=True)
                self.log.info(f"[{traceid}] 触发长期记忆，加载背景")
            else:
                messages = memory.build_messages(prompt, username=username, include_long_term=False)
                self.log.info(f"[{traceid}] 未触发长期记忆，仅使用短期记忆")

            llm_config = self.config.get('llm', {})
            ollama_config = llm_config.get('ollama', {})
            options = {
                "temperature": ollama_config.get('temperature', 0.7),
                "num_predict": ollama_config.get('max_tokens', 256),
                "num_ctx": ollama_config.get('num_ctx', 4096),
                "top_p": ollama_config.get('top_p', 0.9),
            }

            role_file = "./character/MiaoWu.yaml"
            try:
                character = CharacterCard(role_file)
                system_prompt = character.build_system_prompt()
                few_shot_messages = character.build_few_shot_messages()
                identity_hint = {"role": "user", "content": f"[当前与你对话的是{relation}]"}
                messages = [{"role": "system", "content": system_prompt}] + few_shot_messages + [identity_hint] + messages
                if character.temperature is not None:
                    options["temperature"] = character.temperature
                if character.max_tokens is not None:
                    options["num_predict"] = character.max_tokens
                self.log.info(f"[{traceid}] 已加载角色卡，系统提示词: {system_prompt[:10]}...")
            except Exception as e:
                self.log.warning(f"[{traceid}] 加载角色卡失败（{e}），将不使用系统提示词")

            response_cfg = self.config.get('response', {})
            timeout_seconds = response_cfg.get('timeout_seconds', 3)
            timeout_phrases = response_cfg.get('timeout_phrases', ["喵呜想一想"])
            timeout_triggered = threading.Event()
            timer = None
            split_chars = self.llmData.split_str
            split_limit = self.llmData.split_limit
            stream_mode = self.ollama_stream and hasattr(self.llm, 'generate_stream')

            enable_thinking = ollama_config.get('enable_thinking', True)
            if not enable_thinking:
                system_index = None
                for i, msg in enumerate(messages):
                    if msg.get('role') == 'system':
                        system_index = i
                        break
                if system_index is not None:
                    original = messages[system_index].get('content', '')
                    messages[system_index]['content'] = f"/no_think\n{original}\n/no_think"
                else:
                    messages.insert(0, {"role": "system", "content": "/no_think\n/no_think\n/no_think"})

            def on_timeout():
                if not timeout_triggered.is_set():
                    if 5 < len(prompt) < 10:
                        phrase = f"{username}说：{prompt}"
                    else:
                        phrase = random.choice(timeout_phrases)
                    self.log.info(f"[{traceid}] 响应超时，插入缓冲语音: {phrase}")
                    self.ttsCore.tts_say(phrase)
                    timeout_triggered.set()

            timer = threading.Timer(timeout_seconds, on_timeout)
            timer.start()

            split_chars = self.llmData.split_str
            split_limit = self.llmData.split_limit
            stream_mode = self.ollama_stream and hasattr(self.llm, 'generate_stream')

            if stream_mode:
                import re
                self.log.info(f"[{traceid}] 使用流式输出（严格 think 丢弃）")
                response_generator = self.llm.generate_stream(messages, options=options)

                all_content = ""
                filtered_content = ""
                temp = ""
                segment_idx = 0
                first_chunk_received = False
                chat_status = "start"

                in_think = False
                prev_filtered_len = 0

                for chunk in response_generator:
                    if not first_chunk_received:
                        timer.cancel()
                        first_chunk_received = True

                    all_content += chunk

                    think_starts = len(re.findall(r'<think', all_content))
                    think_ends = len(re.findall(r'</think>', all_content))
                    in_think = think_starts > think_ends

                    open_parens = len(re.findall(r'[\(（]', all_content))
                    close_parens = len(re.findall(r'[\)）]', all_content))
                    paren_depth = open_parens - close_parens
                    if paren_depth < 0:
                        paren_depth = 0
                    in_paren = paren_depth > 0

                    current_filtered = re.sub(r'<think>.*?</think>', '', all_content, flags=re.DOTALL)
                    current_filtered = re.sub(r'\(.*?\)', '', current_filtered)
                    current_filtered = re.sub(r'（.*?）', '', current_filtered)

                    new_part = current_filtered[prev_filtered_len:]
                    prev_filtered_len = len(current_filtered)
                    filtered_content = current_filtered

                    if not in_think and not in_paren:
                        temp += new_part
                    else:
                        pass

                    if not in_think and not in_paren and len(temp) >= split_limit:
                        last_punct_pos = -1
                        for punct in split_chars:
                            pos = temp.rfind(punct)
                            if pos > last_punct_pos:
                                last_punct_pos = pos
                        if last_punct_pos != -1:
                            send_text = temp[:last_punct_pos+1].strip()
                            temp = temp[last_punct_pos+1:]
                            if send_text:
                                jsonStr = {
                                    "voiceType": "chat",
                                    "traceid": traceid,
                                    "chatStatus": chat_status,
                                    "question": title if segment_idx == 0 else "",
                                    "text": send_text,
                                    "lanuage": "AutoChange",
                                    "seg_index": segment_idx,
                                    "total_segments": -1
                                }
                                self.llmData.AnswerList.put(jsonStr)
                                self.log.info(f"[{traceid}] 流式分段{segment_idx+1}: {send_text}")
                                segment_idx += 1
                                chat_status = ""

                if not in_think and temp.strip():
                    jsonStr = {
                        "voiceType": "chat",
                        "traceid": traceid,
                        "chatStatus": "end",
                        "question": "",
                        "text": temp.strip(),
                        "lanuage": "AutoChange",
                        "seg_index": segment_idx,
                        "total_segments": -1
                    }
                    self.llmData.AnswerList.put(jsonStr)
                    self.log.info(f"[{traceid}] 流式最后分段: {temp.strip()}")
                else:
                    if segment_idx == 0:
                        if filtered_content.strip():
                            jsonStr = {
                                "voiceType": "chat",
                                "traceid": traceid,
                                "chatStatus": "end",
                                "question": title,
                                "text": filtered_content.strip(),
                                "lanuage": "AutoChange",
                                "seg_index": 0,
                                "total_segments": 1
                            }
                            self.llmData.AnswerList.put(jsonStr)
                            self.log.info(f"[{traceid}] 流式全文本: {filtered_content.strip()}")
                    else:
                        jsonStr = {
                            "voiceType": "chat",
                            "traceid": traceid,
                            "chatStatus": "end",
                            "question": "",
                            "text": "",
                            "lanuage": "AutoChange",
                            "seg_index": segment_idx,
                            "total_segments": -1
                        }
                        self.llmData.AnswerList.put(jsonStr)
                        self.log.info(f"[{traceid}] 流式结束标记发送")

                if not first_chunk_received:
                    timer.cancel()
                    self.log.info(f"[{traceid}] 流式未收到任何块，超时可能已触发")

                def remove_analysis(text):
                    import re
                    keywords = ["这段对话", "这段文字", "这个对话"]
                    for kw in keywords:
                        idx = text.find(kw)
                        if idx != -1:
                            text = text[:idx].rstrip()
                            break
                    text = re.sub(r'（[^）]*）', '', text)
                    text = re.sub(r'\([^)]*\)', '', text)
                    return text.strip()

                filtered_content = remove_analysis(filtered_content)
                memory.add_assistant_message(filtered_content)

            else:
                self.log.info(f"[{traceid}] 使用非流式输出")
                response = self.llm.generate(messages, options=options)
                timer.cancel()
                if timeout_triggered.is_set():
                    self.log.info(f"[{traceid}] 模型在超时后返回，缓冲语音已插入")

                response = response.replace("You", username)

                def remove_analysis(text):
                    import re
                    keywords = ["这段对话", "这段文字", "这个对话"]
                    for kw in keywords:
                        idx = text.find(kw)
                        if idx != -1:
                            return text[:idx].rstrip()
                    text = re.sub(r'（[^）]*）', '', text)
                    text = re.sub(r'\([^)]*\)', '', text)
                    return text.strip()

                def remove_think_tags(text):
                    import re
                    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
                
                response = remove_analysis(response)
                response = remove_think_tags(response)

                memory.add_assistant_message(response)
                all_content = response

                import re
                split_chars = self.llmData.split_str
                pattern = '|'.join(re.escape(c) for c in split_chars)
                segments = re.split(f'({pattern})', all_content)
                raw_sentences = []
                for i in range(0, len(segments)-1, 2):
                    raw_sentences.append(segments[i] + (segments[i+1] if i+1 < len(segments) else ''))
                if len(segments) % 2 == 1:
                    raw_sentences.append(segments[-1])
                raw_sentences = [seg.strip() for seg in raw_sentences if seg.strip()]

                min_len = self.llmData.split_limit
                merged_segments = []
                current = ""
                for sent in raw_sentences:
                    if not current:
                        current = sent
                    else:
                        if len(current) < min_len:
                            current += sent
                        else:
                            merged_segments.append(current)
                            current = sent
                if current:
                    merged_segments.append(current)
                if not merged_segments:
                    merged_segments = [all_content]

                total = len(merged_segments)
                for idx, seg in enumerate(merged_segments):
                    if total == 1:
                        status = "end"
                    else:
                        if idx == 0:
                            status = "start"
                        elif idx == total - 1:
                            status = "end"
                        else:
                            status = ""
                    jsonStr = {
                        "voiceType": "chat",
                        "traceid": traceid,
                        "chatStatus": status,
                        "question": title if idx == 0 else "",
                        "text": seg,
                        "lanuage": "AutoChange",
                        "seg_index": idx,
                        "total_segments": total
                    }
                    self.llmData.AnswerList.put(jsonStr)
                    self.log.info(f"[{traceid}]分段{idx+1}/{total}: {seg}")

            if "粉色" in all_content or "睡觉" in all_content or "粉红" in all_content or "房间" in all_content or "晚上" in all_content:
                self.actionOper.changeScene("粉色房间")
            elif "清晨" in all_content or "早" in all_content or "睡醒" in all_content:
                self.actionOper.changeScene("清晨房间")
            elif "祭拜" in all_content or "神社" in all_content or "寺庙" in all_content:
                self.actionOper.changeScene("神社")
            elif "花房" in all_content or "花香" in all_content:
                self.actionOper.changeScene("花房")
            elif "岸" in all_content or "海" in all_content:
                self.actionOper.changeScene("海岸花坊")

            current_question_count = self.llmData.QuestionList.qsize()
            self.log.info(f"[{traceid}][AI回复]{all_content}")
            self.log.info(f"[{traceid}]System>>[{username}]的回复已存入队列，当前剩余问题数:{current_question_count}")

            self.llmData.is_ai_ready = True
            return

        # 其他（如 fastgpt 流式）...
        self.obs.show_text("状态提示", f'{self.llmData.Ai_Name}思考问题"{title}"完成')

        all_content = ""
        temp = ""
        chatStatus = "start"
        for line in response.iter_lines():
            if line:
                str_data = line.decode("utf-8")
                str_data = str_data.replace("data: ", "")
                self.log.info(f"[{traceid}]{str_data}")
                if str_data != "[DONE]":
                    response_json = json.loads(str_data)
                    if response_json["choices"][0]["finish_reason"] != "stop":
                        if "error" in response_json:
                            self.log.error(f"API返回错误: {response_json['error']}")
                            continue
                        choices = response_json.get("choices", [])
                        if not choices:
                            self.log.error("choices为空，跳过")
                            continue
                        choice = choices[0]
                        if "delta" in choice:
                            stream_content = choice["delta"].get("content", "")
                        elif "message" in choice:
                            stream_content = choice["message"].get("content", "")
                        else:
                            stream_content = ""
                        if not stream_content and choice.get("finish_reason") != "stop":
                            continue
                        all_content = all_content + stream_content
                        stream_content = StringUtil.filter_html_tags(stream_content)
                        content = temp + stream_content
                        num = StringUtil.rfind_index_contain_string(self.llmData.split_str, content)
                        if num > 0:
                            split_content = content[0:num]
                            self.log.info(f"[{traceid}]分割后文本:" + split_content)
                            if len(split_content) > self.llmData.split_limit:
                                temp = content[num: len(content)]
                                jsonStr = {"voiceType": "chat", "traceid": traceid, "chatStatus": chatStatus,
                                           "question": title, "text": split_content, "lanuage": "AutoChange"}
                                self.llmData.AnswerList.put(jsonStr)
                                chatStatus = ""
                            else:
                                temp = content
                        else:
                            temp = content
                    else:
                        if temp != "":
                            jsonStr = {"voiceType": "chat", "traceid": traceid, "chatStatus": "end", "question": title,
                                       "text": temp, "lanuage": "AutoChange"}
                            self.llmData.AnswerList.put(jsonStr)
                        else:
                            jsonStr = {"voiceType": "chat", "traceid": traceid, "chatStatus": "end", "question": title,
                                       "text": "", "lanuage": "AutoChange"}
                            self.llmData.AnswerList.put(jsonStr)
                        self.log.info(f"[{traceid}]end:" + temp)
        self.llmData.is_ai_ready = True

        if "粉色" in all_content or "睡觉" in all_content or "粉红" in all_content or "房间" in all_content or "晚上" in all_content:
            self.actionOper.changeScene("粉色房间")
        elif "清晨" in all_content or "早" in all_content or "睡醒" in all_content:
            self.actionOper.changeScene("清晨房间")
        elif "祭拜" in all_content or "神社" in all_content or "寺庙" in all_content:
            self.actionOper.changeScene("神社")
        elif "花房" in all_content or "花香" in all_content:
            self.actionOper.changeScene("花房")
        elif "岸" in all_content or "海" in all_content:
            self.actionOper.changeScene("海岸花坊")

        current_question_count = self.llmData.QuestionList.qsize()
        self.log.info(f"[{traceid}][AI回复]{all_content}")
        self.log.info(f"[{traceid}]System>>[{username}]的回复已存入队列，当前剩余问题数:{current_question_count}")

    # 检查LLM回复线程
    def check_answer(self):
        if not self.llmData.QuestionList.empty() and self.llmData.is_ai_ready:
            self.llmData.is_ai_ready = False
            answers_thread = Thread(target=self.aiResponseTry)
            answers_thread.start()

    # 进入直播间欢迎语
    def check_welcome_room(self):
        count = len(self.llmData.WelcomeList)
        numstr = ""
        if count > 1:
            numstr = f"{count}位"
        userlist = str(self.llmData.WelcomeList).replace("['", "").replace("']", "")
        if len(self.llmData.WelcomeList) > 0:
            traceid = str(uuid.uuid4())
            text = f'欢迎"{userlist}"{numstr}来到{self.commonData.Ai_Name}的直播间喵'
            self.log.info(f"[{traceid}]{text}")
            self.llmData.WelcomeList.clear()
            if self.llmData.is_llm_welcome == True:
                llm_json = {"traceid": traceid, "prompt": text, "uid": 0, "username": self.commonData.Ai_Name}
                self.llmData.QuestionList.put(llm_json)
            else:
                self.ttsCore.tts_say(text)

    # 聊天入口处理（主要修改点）
    def msg_deal(self, traceid, query, uid, user_name):
        text = self.llmData.cmd
        is_contain = StringUtil.has_string_reg_list(f"^{text}", query)
        if is_contain is not None:
            num = StringUtil.is_index_contain_string(text, query)
            queryExtract = query[num: len(query)]  # 提取提问语句
            queryExtract = queryExtract.strip()
            self.log.info(f"[{traceid}]用户对话：" + queryExtract)
            if queryExtract == "":
                return True

            # ====== 连续“不要说话”检测（暂停30秒自动恢复）======
            current_contain = "不要说话" in queryExtract
            if current_contain and self.last_msg_contain_dont_speak:
                self.log.info(f"[{traceid}] 检测到连续“不要说话”，暂停语音输出30秒")
                self.ttsCore.pause()

                # 清空所有待处理队列（可选，避免暂停期间遗留任务）
                while not self.llmData.QuestionList.empty():
                    try:
                        self.llmData.QuestionList.get_nowait()
                    except:
                        break
                while not self.llmData.AnswerList.empty():
                    try:
                        self.llmData.AnswerList.get_nowait()
                    except:
                        break

                # 取消之前的定时器（如果有）
                if self._pause_timer:
                    self._pause_timer.cancel()
                # 创建新的30秒后恢复的定时器
                self._pause_timer = threading.Timer(30.0, self.ttsCore.resume)
                self._pause_timer.daemon = True  # 随主线程退出
                self._pause_timer.start()

                self.last_msg_contain_dont_speak = current_contain
                return True
            self.last_msg_contain_dont_speak = current_contain

            # ====== 新增：预设回复匹配（包含关键词）======
            matched_keyword = None
            matched_reply = None
            for keyword, replies in self.preset_responses.items():
                if keyword in queryExtract:
                    matched_keyword = keyword
                    # 根据用户名过滤回复
                    if user_name == "YGZ醒脑片":
                        # 使用全部回复
                        replies_filtered = replies
                    else:
                        # 排除包含“主人”的回复
                        replies_filtered = [r for r in replies if "主人" not in r]
                    if replies_filtered:
                        matched_reply = random.choice(replies_filtered)
                    else:
                        self.log.info(f"[{traceid}] 用户 {user_name} 无法使用预设关键词 {keyword} 的回复（全部包含主人），继续走 LLM")
                        matched_reply = None
                    break  # 只匹配第一个关键词

            if matched_reply:
                self.log.info(f"[{traceid}] 命中预设关键词“{matched_keyword}”，回复: {matched_reply}")
                # 构造回复 JSON（视为完整的一段，状态为 end）
                jsonStr = {
                    "voiceType": "chat",
                    "traceid": traceid,
                    "chatStatus": "end",
                    "question": queryExtract,
                    "text": matched_reply,
                    "lanuage": "AutoChange",
                    "seg_index": 0,
                    "total_segments": 1
                }
                self.llmData.AnswerList.put(jsonStr)

                # ====== 将本轮对话加入短期记忆 ======
                uid_str = str(uid)
                memory = self._ensure_memory_manager(uid_str, user_name)
                if memory:
                    memory.add_user_message(queryExtract, user_name)
                    memory.add_assistant_message(matched_reply)
                return True

            # 未命中预设，正常走 LLM 流程
            llm_json = {"traceid": traceid, "prompt": query, "uid": uid, "username": user_name}
            self.llmData.QuestionList.put(llm_json)
            return True
        return False

    def add_system_message(self, text, username="主人", uid=0):
        """系统主动发起的消息，直接放入对话队列，由 AI 处理"""
        traceid = str(uuid.uuid4())
        llm_json = {
            "traceid": traceid,
            "prompt": text,
            "uid": uid,
            "username": username
        }
        self.llmData.QuestionList.put(llm_json)
        self.log.info(f"[{traceid}] 系统主动消息: {text}")