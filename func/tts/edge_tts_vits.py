from func.tools.singleton_mode import singleton
from func.config.default_config import defaultConfig
import edge_tts
import asyncio
import aiohttp
from aiohttp_socks import ProxyConnector  # 正确导入

@singleton
class EdgeTTs:
    # 加载配置
    config = defaultConfig().get_config()
    speaker_name = config["speech"]["edge-tts"]["speaker_name"]
    # 读取代理配置
    proxy = config.get("translate", {}).get("HttpProxies", None)

    def __init__(self):
        if self.proxy:
            print(f"使用代理: {self.proxy}")

    # 生成语音
    async def generate(self, text, voice, filename):
        if self.proxy:
            # 使用 aiohttp_socks 的 ProxyConnector
            connector = ProxyConnector.from_url(self.proxy)
            async with aiohttp.ClientSession(connector=connector) as session:
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=voice,
                    rate="+20%",
                    volume="+20%",
                    session=session
                )
                await communicate.save(f"./output/{filename}.mp3")
        else:
            async with aiohttp.ClientSession() as session:
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=voice,
                    rate="+20%",
                    volume="+20%",
                    session=session
                )
                await communicate.save(f"./output/{filename}.mp3")

    # 获取语音
    def get_vists(self, filename, text, emotion):
        asyncio.run(self.generate(text, self.speaker_name, filename))