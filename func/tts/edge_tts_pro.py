import os
import asyncio
import requests
from func.log.default_log import DefaultLog
from func.tts.edge_tts_vits import EdgeTTs
from func.config.default_config import defaultConfig

class EdgeTtsPro:
    """先使用 edge-tts 生成语音，再调用 sovits 变声"""

    def __init__(self):
        self.log = DefaultLog().getLogger()
        self.config = defaultConfig().get_config()
        self.edge_config = self.config["speech"]["edge-tts-pro"]
        self.edge_tts = EdgeTTs()                     # 复用原有的 edge-tts 生成器
        self.sovits_url = self.edge_config["sovits_url"]
        self.sovits_type = self.edge_config.get("sovits_type", "gpt-sovits")
        # 其他可能用到的参数
        self.prompt_language = self.edge_config.get("prompt_language", "zh")
        self.text_language = self.edge_config.get("text_language", "zh")

    def get_vists(self, filename: str, text: str, emotion: str) -> int:
        """
        生成语音并变声
        :param filename: 最终输出文件名（不含路径，保存在 ./output/ 下）
        :param text: 要合成的文本
        :param emotion: 情感（可能用于 edge-tts 或 sovits）
        :return: 0 失败，1 成功
        """
        try:
            # 1. 使用 edge-tts 生成临时音频
            temp_filename = f"temp_{filename}"
            asyncio.run(self.edge_tts.generate(text, self.edge_config["speaker_name"], temp_filename))
            temp_path = os.path.abspath(os.path.join(".", "output", f"{temp_filename}.mp3"))
            if not os.path.exists(temp_path):
                self.log.error(f"edge-tts 生成失败：{temp_path}")
                return 0

            # 2. 调用 sovits 变声服务
            final_path = os.path.abspath(os.path.join(".", "output", f"{filename}.mp3"))
            success = self._call_sovits(temp_path, final_path, text)
            if not success:
                self.log.error("sovits 变声失败")
                return 0

            # 3. 删除临时文件
            os.remove(temp_path)
            return 1

        except Exception as e:
            self.log.exception(f"EdgeTtsPro 处理异常：{e}")
            return 0

    def _call_sovits(self, input_audio: str, output_audio: str, text: str) -> bool:
        """
        根据 sovits_type 调用不同的变声 API
        """
        if self.sovits_type == "gpt-sovits":
            return self._call_gpt_sovits(input_audio, output_audio, text)
        elif self.sovits_type == "rvc":
            return self._call_rvc(input_audio, output_audio)
        else:
            self.log.error(f"不支持的 sovits_type: {self.sovits_type}")
            return False

    def _call_gpt_sovits(self, input_audio: str, output_audio: str, text: str) -> bool:
        """
        调用 GPT-SoVITS 的变声接口（适配 api.py 的 POST JSON 接口）
        api.py 根路径接受 JSON，例如：
        {
            "refer_wav_path": "xxx.wav",
            "prompt_text": "参考文本",
            "prompt_language": "zh",
            "text": "要合成的文本",
            "text_language": "zh"
        }
        """
        try:
            url = self.sovits_url.rstrip('/') + '/'  # 确保是 http://127.0.0.1:9880/

            # 构造 JSON 数据（使用绝对路径）
            payload = {
                "refer_wav_path": input_audio,
                "prompt_text": text,  # 如果参考音频已有默认文本，也可留空，但建议传入
                "prompt_language": self.prompt_language,
                "text": text,
                "text_language": self.text_language
            }

            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                with open(output_audio, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                self.log.error(f"GPT-SoVITS 返回错误：{response.status_code} - {response.text}")
                return False
        except Exception as e:
            self.log.exception(f"调用 GPT-SoVITS 异常：{e}")
            return False

    def _call_rvc(self, input_audio: str, output_audio: str) -> bool:
        """
        调用 RVC 变声接口（示例，需根据实际 RVC 服务 API 调整）
        """
        try:
            url = self.sovits_url.rstrip('/') + '/convert'
            files = {"audio": open(input_audio, "rb")}
            data = {"model": "your_model_name"}  # 可从配置中读取
            response = requests.post(url, data=data, files=files, timeout=30)
            if response.status_code == 200:
                with open(output_audio, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                self.log.error(f"RVC 返回错误：{response.status_code}")
                return False
        except Exception as e:
            self.log.exception(f"调用 RVC 异常：{e}")
            return False