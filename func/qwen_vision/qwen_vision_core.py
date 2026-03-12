import threading
import time
import io
import base64
import os
from PIL import ImageGrab, Image
from openai import OpenAI
from func.log.default_log import DefaultLog
from func.tts.tts_core import TTsCore
from func.llm.tgw import Tgw
from func.llm.ollama_llm import Ollama
from func.config.default_config import defaultConfig

class QwenVisionCore:
    def __init__(self):
        self.log = DefaultLog().getLogger()
        self.config = defaultConfig().get_config()
        vision_config = self.config.get('qwen_vision', {})
        self.enabled = vision_config.get('enabled', False)
        self.api_key = vision_config.get('api_key', os.getenv("DASHSCOPE_API_KEY", ""))
        self.base_url = vision_config.get('base_url', "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = vision_config.get('model', "qwen3.5-plus")
        self.prompt = vision_config.get('prompt', "描述一下图片内容，简洁精准")
        self.max_tokens = vision_config.get('max_tokens', 200)
        self.temperature = vision_config.get('temperature', 0.6)
        # 关键词触发配置
        self.keywords = vision_config.get('keywords', [
            "主人在干什么",
            "看看主人的电脑界面",
            "看看主人的屏幕",
            "看看屏幕",
            "看屏幕",
            "看一下屏幕"
        ])
        self.cooldown = vision_config.get('cooldown', 10)  # 冷却时间（秒）
        self.last_trigger_time = 0
        self._task_running = False
        self._lock = threading.Lock()

        # 读取 LLM 类型配置
        llm_config = self.config.get('llm', {})
        self.local_llm_type = llm_config.get('local_llm_type', 'fastgpt')

        # 预初始化可能用到的 LLM 客户端
        self.tgw = None
        self.ollama = None
        if self.local_llm_type == "text-generation-webui":
            self.tgw = Tgw()
        elif self.local_llm_type == "ollama":
            self.ollama = Ollama()
        else:
            self.log.error(f"不支持的 local_llm_type: {self.local_llm_type}，视觉评价将无法生成")
        self.tts = TTsCore()
        self.client = None
        self._init_client()

        # 读取优化配置（新增）
        optimize_config = vision_config.get('optimize', {})
        self.optimize_enabled = optimize_config.get('enabled', False)
        self.optimize_api_key = optimize_config.get('api_key', self.api_key)
        self.optimize_base_url = optimize_config.get('base_url', self.base_url)
        self.optimize_model = optimize_config.get('model', 'qwen-plus')
        self.optimize_prompt_template = optimize_config.get('prompt_template', "把如下内容整合成简短准确的口语化概述，客观实际，不要有任何主观内容：\n{description}，60字左右。")
        self.optimize_max_tokens = optimize_config.get('max_tokens', 300)
        self.optimize_temperature = optimize_config.get('temperature', 0.5)

        # 创建优化客户端（新增）
        self.optimize_client = None
        if self.optimize_enabled:
            if self.optimize_api_key == self.api_key and self.optimize_base_url == self.base_url:
                # 复用主客户端
                self.optimize_client = self.client
            else:
                try:
                    self.optimize_client = OpenAI(
                        api_key=self.optimize_api_key,
                        base_url=self.optimize_base_url
                    )
                    self.log.info(f"优化客户端已创建，使用模型: {self.optimize_model}")
                except Exception as e:
                    self.log.error(f"创建优化客户端失败: {e}")
                    self.optimize_client = None

    def _init_client(self):
        """初始化 OpenAI 客户端（兼容阿里云百炼）"""
        if not self.api_key:
            self.log.error("未配置阿里云百炼 API Key，请在配置文件中设置 qwen_vision.api_key 或环境变量 DASHSCOPE_API_KEY")
            self.client = None
            return
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self.log.info(f"Qwen 客户端已创建，使用模型: {self.model}")
        except Exception as e:
            self.log.error(f"创建 Qwen 客户端失败: {e}")
            self.client = None

    def get_screenshot_base64(self):
        """获取屏幕截图，返回带data URI头的Base64字符串（缩放至1024x1024）"""
        try:
            screenshot = ImageGrab.grab()
            screenshot.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            img_byte_arr = io.BytesIO()
            screenshot.save(img_byte_arr, format='JPEG', quality=85)
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
        except Exception as e:
            self.log.error(f"截图失败: {e}")
            return None

    def call_qwen(self, image_data_uri):
        """调用 Qwen 视觉模型生成图像描述"""
        if not self.client:
            self.log.error("Qwen 客户端未初始化")
            return None
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_uri}
                            },
                            {"type": "text", "text": self.prompt}
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            reply = completion.choices[0].message.content
            return reply
        except Exception as e:
            self.log.error(f"调用 Qwen 视觉模型异常: {e}")
            return None

    def optimize_text(self, text):
        """调用文本模型优化描述文本，返回优化后的字符串"""
        if not self.optimize_client:
            self.log.error("优化客户端未初始化")
            return text

        # 构建提示词
        prompt = self.optimize_prompt_template.format(description=text)

        try:
            completion = self.optimize_client.chat.completions.create(
                model=self.optimize_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.optimize_max_tokens,
                temperature=self.optimize_temperature
            )
            optimized = completion.choices[0].message.content
            self.log.info(f"优化后的描述: {optimized}")
            return optimized
        except Exception as e:
            self.log.error(f"调用优化模型异常: {e}")
            return text  # 优化失败时返回原文本

    def run_once(self):
        """执行一次任务：截图->视觉描述->生成评价->说话"""
        self.log.info("开始执行 Qwen 视觉任务")
        img_uri = self.get_screenshot_base64()
        if not img_uri:
            return
        caption = self.call_qwen(img_uri)
        if not caption:
            self.log.warning("未获取到 Qwen 视觉描述")
            return
        self.log.info(f"Qwen 返回描述: {caption}")
        if self.optimize_enabled and self.optimize_client:
            caption = self.optimize_text(caption)
            self.log.info(f"优化后描述: {caption}")

        # 根据配置的 LLM 类型生成评价
        reply = None
        if self.local_llm_type == "text-generation-webui" and self.tgw:
            prompt = f"你现在正在观看主人的屏幕，屏幕上有{caption}。你觉得主人在干什么呢？"
            messages = [{"role": "user", "content": prompt}]
            reply = self.tgw.chat(
                messages,
                uid="qwen_vision",
                username="喵呜",
                character="MiaoWu",
                relation="主人"
            )
        elif self.local_llm_type == "ollama" and self.ollama:
            # 构造符合 Ollama 的消息格式，可加入角色卡信息
            system_prompt = "你是一只可爱的小猫娘，名叫喵呜。"
            user_prompt = f"你现在正在观看主人的屏幕，屏幕上有{caption}。你觉得主人在干什么呢？"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            reply = self.ollama.generate(messages)
        else:
            self.log.error("没有可用的 LLM 客户端，无法生成评价")

        if reply:
            self.log.info(f"AI评价: {reply}")
            self.tts.tts_say(reply)
        else:
            self.log.warning("未能生成评价")

    def _execute_task(self):
        """包装 run_once，用于线程执行，并清理运行标志"""
        try:
            self.run_once()
        except Exception as e:
            self.log.exception("视觉任务执行异常")
        finally:
            with self._lock:
                self._task_running = False

    def check_and_trigger(self, user_message):
        """
        检查用户消息是否包含触发关键词，若是则启动视觉任务（受冷却时间和并发限制）
        返回 True 表示已触发任务，False 表示未触发或无法触发
        """
        if not self.enabled:
            return False
        if not user_message:
            return False

        # 检查关键词
        matched = False
        for kw in self.keywords:
            if kw in user_message:
                matched = True
                break
        if not matched:
            return False

        # 检查冷却时间
        now = time.time()
        if now - self.last_trigger_time < self.cooldown:
            self.log.debug(f"视觉触发冷却中，忽略关键词触发")
            return False

        # 检查任务是否已在运行
        with self._lock:
            if self._task_running:
                self.log.debug("视觉任务已在执行，忽略本次触发")
                return False
            self._task_running = True

        # 更新触发时间
        self.last_trigger_time = now

        # 启动新线程执行任务
        threading.Thread(target=self._execute_task, daemon=True).start()
        self.log.info(f"已触发视觉任务，用户消息: {user_message}")
        return True