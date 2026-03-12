import threading
import time
import io
import base64
import requests
from PIL import ImageGrab
from func.log.default_log import DefaultLog
from func.tts.tts_core import TTsCore
from func.llm.tgw import Tgw  # 直接使用已有的Tgw类
from func.config.default_config import defaultConfig

class JoyCaptionCore:
    def __init__(self):
        self.log = DefaultLog().getLogger()
        self.config = defaultConfig().get_config()
        joy_config = self.config.get('joycaption', {})
        self.enabled = joy_config.get('enabled', False)
        self.url = joy_config.get('url', 'http://127.0.0.1:7861').rstrip('/')
        self.interval = joy_config.get('interval', 60)
        # JoyCaption 生成参数（可在config中覆盖）
        self.prompt = joy_config.get('prompt', "Write a long descriptive caption for this image in a formal tone.")
        self.max_tokens = joy_config.get('max_tokens', 300)
        self.temperature = joy_config.get('temperature', 0.6)
        self.top_p = joy_config.get('top_p', 0.9)
        self.tts = TTsCore()
        # 初始化Tgw（单例，用于调用text-generation-webui）
        self.tgw = Tgw()
        # Gradio API端点
        self.api_url = f"{self.url}/gradio_api/process_single_image"

    def get_screenshot_base64(self):
        """获取屏幕截图，返回带data URI头的Base64字符串"""
        try:
            screenshot = ImageGrab.grab()
            img_byte_arr = io.BytesIO()
            screenshot.save(img_byte_arr, format='JPEG', quality=85)
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
        except Exception as e:
            self.log.error(f"截图失败: {e}")
            return None

    def call_joycaption(self, image_data_uri):
        """调用JoyCaption的Gradio API生成图像描述"""
        try:
            # 参数顺序必须与UI中process_single_image的输入一致
            payload = {
                "data": [
                    image_data_uri,
                    self.prompt,
                    self.max_tokens,
                    self.temperature,
                    self.top_p
                ]
            }
            response = requests.post(self.api_url, json=payload, timeout=60)
            if response.status_code != 200:
                self.log.error(f"JoyCaption请求失败，状态码: {response.status_code}")
                return None

            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                caption = result["data"][0]
                # 检查是否返回错误提示（如模型未加载）
                if caption and ("请先加载模型" in caption or "模型加载失败" in caption):
                    self.log.error(f"JoyCaption服务返回错误: {caption}")
                    return None
                return caption
            self.log.warning("无法从返回中提取描述文本")
            return None
        except Exception as e:
            self.log.error(f"调用JoyCaption异常: {e}")
            return None

    def run_once(self):
        """执行一次任务：截图->打标->生成评价->说话"""
        self.log.info("开始执行JoyCaption任务")
        img_uri = self.get_screenshot_base64()
        if not img_uri:
            return
        caption = self.call_joycaption(img_uri)
        if not caption:
            self.log.warning("未获取到JoyCaption描述")
            return
        self.log.info(f"JoyCaption返回描述: {caption}")

        # 构造发送给text-generation-webui的提示词
        prompt = f"你现在正在观看屏幕，屏幕内容描述为：{caption}。根据这个场景给出反馈。"
        # 调用Tgw生成回复（参数uid, username, character, relation可固定或从配置获取）
        reply = self.tgw.chat(
            content=prompt,
            uid="joycaption",
            username="系统",
            character="Assistant",
            relation="助手"
        )
        if reply:
            self.log.info(f"AI评价: {reply}")
            self.tts.tts_say(reply)  # 语音播报
        else:
            self.log.warning("未能生成评价")

    def start_background_thread(self):
        """启动独立后台线程"""
        def loop():
            self.log.info("JoyCaption 后台线程已启动")
            while True:
                try:
                    self.run_once()
                except Exception as e:
                    self.log.error(f"JoyCaption 任务执行异常: {e}")
                time.sleep(self.interval)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        self.log.info(f"JoyCaption 后台线程已创建，间隔 {self.interval} 秒")