import threading
import time
import io
import base64
from PIL import ImageGrab, Image
from gradio_client import Client
from func.log.default_log import DefaultLog
from func.tts.tts_core import TTsCore
from func.llm.tgw import Tgw
from func.config.default_config import defaultConfig

class JoyCaptionCore:
    def __init__(self):
        self.log = DefaultLog().getLogger()
        self.config = defaultConfig().get_config()
        joy_config = self.config.get('joycaption', {})
        self.enabled = joy_config.get('enabled', False)
        self.url = joy_config.get('url', 'http://127.0.0.1:7861').rstrip('/')
        self.interval = joy_config.get('interval', 60)
        self.prompt = joy_config.get('prompt', "Write a long descriptive caption for this image in a formal tone.")
        self.max_tokens = joy_config.get('max_tokens', 120)
        self.temperature = joy_config.get('temperature', 0.6)
        self.top_p = joy_config.get('top_p', 0.9)
        self.tts = TTsCore()
        self.tgw = Tgw()
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            self.client = Client(self.url)
            self.log.info(f"Gradio 客户端已创建，连接到 {self.url}")
            try:
                api_info = self.client.view_api()
                self.log.debug(f"API 信息: {api_info}")
            except:
                pass
        except Exception as e:
            self.log.error(f"创建 Gradio 客户端失败: {e}")
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

    def call_joycaption(self, image_data_uri):
        """使用 Gradio 客户端调用 JoyCaption 生成图像描述"""
        if not self.client:
            self.log.error("Gradio 客户端未初始化")
            return None
        try:
            result = self.client.predict(
                {"url": image_data_uri},
                self.prompt,
                self.max_tokens,
                self.temperature,
                self.top_p,
                api_name="/process_single_image"   # 必须与 joy2_gradio.py 中定义的 api_name 一致
            )
            if isinstance(result, str):
                return result
            elif isinstance(result, list) and len(result) > 0:
                return result[0]
            else:
                self.log.warning(f"未知返回格式: {result}")
                return None
        except Exception as e:
            self.log.error(f"调用 JoyCaption 异常: {e}")
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

        # 构造发送给 text-generation-webui 的提示词
        prompt = f"你现在正在观看主人的屏幕，屏幕上有{caption}。你觉得主人在干什么呢？"
        # 调用Tgw生成回复
        # 根据tgw.py，username参数被用作角色卡名，character参数被用作预设
        reply = self.tgw.chat(
            content=prompt,
            uid="joycaption",
            username="喵呜",          # 角色卡名称（请确保角色卡文件名为“喵呜”）
            character="MiaoWu",    # 预设，可保持默认
            relation="主人"
        )
        if reply:
            self.log.info(f"AI评价: {reply}")
            self.tts.tts_say(reply)
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