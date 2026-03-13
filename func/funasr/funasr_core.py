import asyncio
import threading
import json
import pyaudio
import websockets
import numpy as np
import collections
import webrtcvad
import time
from typing import Optional, Callable

from func.log.default_log import DefaultLog
from func.config.default_config import defaultConfig


class FunASRCore:
    def __init__(self, callback: Optional[Callable[[str, str, str], None]] = None):
        self.log = DefaultLog().getLogger()
        full_config = defaultConfig().get_config()
        self.funasr_config = full_config.get('funasr', {})
        self.enabled = self.funasr_config.get('enabled', False)
        if not self.enabled:
            return

        self.server_url = self.funasr_config.get('server_url', 'ws://127.0.0.1:10095/')
        self.mode = self.funasr_config.get('mode', '2pass')
        self.hotwords = self.funasr_config.get('hotwords', [])
        self.username = self.funasr_config.get('username', '访客')
        self.uid = self.funasr_config.get('uid', 'funasr_user')
        self.energy_threshold = self.funasr_config.get('energy_threshold', 500)
        self.power_save_enabled = self.funasr_config.get('power_save_enabled', False)
        self.power_save_silence_seconds = self.funasr_config.get('power_save_silence_seconds', 30)
        self.power_save_check_interval = self.funasr_config.get('power_save_check_interval', 0.5)
        self.vad_energy_threshold = self.funasr_config.get('vad_energy_threshold', 500)

        # 音频参数
        self.CHUNK = 960
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.callback = callback

        # AI 核心接口
        from func.gobal.data import CommonData
        self.commonData = CommonData()
        self.api_base = f"http://127.0.0.1:{self.commonData.port}"

        # 创建独立的 VAD 实例
        self.vad = webrtcvad.Vad(1)

        # 队列：存放待处理的完整句子音频（按顺序）
        self.segment_audio_queue = asyncio.Queue()
        # 心跳间隔（秒）
        self.heartbeat_interval = 30
        # ====== 新增：句子合并延迟发送 ======
        self.merge_timeout = 2  # 等待合并的时间（秒）
        self.pending_tasks = {}  # 说话人 -> asyncio.Task
        self.pending_texts = {}  # 说话人 -> 累积文本

    def start(self):
        if not self.enabled:
            self.log.info("FunASR 未启用")
            return
        if self.thread and self.thread.is_alive():
            self.log.warning("FunASR 已在运行")
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        self.log.info("FunASR 识别线程已启动")

    def stop(self):
        self.running = False
        # 取消所有待发送任务
        for task in self.pending_tasks.values():
            task.cancel()
        self.pending_tasks.clear()
        self.pending_texts.clear()

        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=5)
        self.log.info("FunASR 识别线程已停止")

    def _run_async_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main())
        except Exception as e:
            self.log.error(f"FunASR 主协程异常: {e}")
        finally:
            self.loop.close()
            self.log.info("FunASR 事件循环结束")

    async def _main(self):
        while self.running:
            try:
                async with websockets.connect(self.server_url, subprotocols=["binary"]) as ws:
                    self.log.info(f"已连接到 FunASR 服务器 {self.server_url}")
                    await self._send_config(ws)
                    send_task = asyncio.create_task(self._audio_sender(ws))
                    recv_task = asyncio.create_task(self._message_receiver(ws))
                    done, pending = await asyncio.wait(
                        [send_task, recv_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
                    if self.running:
                        self.log.info("连接意外断开，1秒后重连...")
                        await asyncio.sleep(1)
            except Exception as e:
                if self.running:
                    self.log.error(f"FunASR 连接异常: {e}，1秒后重连")
                    await asyncio.sleep(1)
                else:
                    break

    async def _send_config(self, ws):
        config = {
            "chunk_size": [5, 10, 5],
            "wav_name": "mic",
            "is_speaking": True,
            "chunk_interval": 10,
            "itn": False,
            "mode": self.mode
        }
        if self.hotwords:
            hotwords_dict = {}
            for hw in self.hotwords:
                parts = hw.split()
                if len(parts) >= 2 and parts[-1].isdigit():
                    hotwords_dict[' '.join(parts[:-1])] = int(parts[-1])
            if hotwords_dict:
                config["hotwords"] = json.dumps(hotwords_dict, ensure_ascii=False)
        await ws.send(json.dumps(config))
        self.log.debug(f"发送启动配置: {config}")

    async def _audio_sender(self, ws):
        p = pyaudio.PyAudio()
        stream = None
        try:
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            self.log.info("麦克风已打开，开始发送音频流")

            frame_duration = self.CHUNK / (self.RATE * 2)  # 30ms
            idle_silence_frames = int(self.power_save_silence_seconds / frame_duration) if self.power_save_enabled else float('inf')
            idle_send_interval = int(1.0 / frame_duration)

            idle_mode = False
            consecutive_silent = 0
            frame_counter = 0
            last_send_time = time.time()  # 记录上次发送时间

            while self.running:
                # 心跳检查：如果距离上次发送超过心跳间隔，发送一帧静音
                now = time.time()
                if now - last_send_time >= self.heartbeat_interval:
                    silent_frame = b'\x00' * (self.CHUNK * 2)  # 静音 PCM 数据
                    await ws.send(silent_frame)
                    self.log.debug("发送心跳帧")
                    last_send_time = now
                    # 跳过本次音频采集，避免心跳过于频繁
                    await asyncio.sleep(0)
                    continue

                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                except Exception as e:
                    self.log.error(f"音频读取错误: {e}")
                    await asyncio.sleep(0.1)
                    continue

                # 能量检测 + VAD
                audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                energy = np.sqrt(np.mean(audio_array ** 2))
                if energy < self.vad_energy_threshold:
                    is_speech = False
                else:
                    try:
                        is_speech = self.vad.is_speech(data, self.RATE)
                    except Exception:
                        is_speech = False

                if is_speech:
                    consecutive_silent = 0
                    if idle_mode:
                        idle_mode = False
                        self.log.info("检测到语音，退出空闲模式")
                else:
                    consecutive_silent += 1

                # 空闲模式判断
                if not idle_mode and consecutive_silent >= idle_silence_frames:
                    idle_mode = True
                    self.log.info(f"静音超过 {self.power_save_silence_seconds} 秒，进入空闲模式")
                    frame_counter = 0

                # 决定是否发送数据
                send_this_frame = False
                if not idle_mode:
                    send_this_frame = True
                else:
                    frame_counter += 1
                    if frame_counter >= idle_send_interval:
                        send_this_frame = True
                        frame_counter = 0

                if send_this_frame:
                    await ws.send(data)
                    last_send_time = time.time()

                await asyncio.sleep(0)

        except Exception as e:
            self.log.error(f"音频发送错误: {e}", exc_info=True)
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            if p:
                p.terminate()
            self.log.info("音频捕获已停止")

    async def _message_receiver(self, ws):
        async for message in ws:
            if isinstance(message, bytes):
                continue
            try:
                data = json.loads(message)
                text = data.get("text")
                is_final = data.get("is_final", False)
                mode = data.get("mode", "")
                if text and text.strip() and ("offline" in mode or is_final):
                    self.log.info(f"识别到最终文本: {text}")
                    # 从消息中获取声纹信息
                    spk_name = data.get("spk_name", "未知")
                    spk_score = data.get("spk_score", 0.0)
                    await self._handle_result(text, spk_name, self.uid, spk_score)
            except json.JSONDecodeError:
                self.log.warning(f"收到非 JSON 消息: {message[:100]}")

    async def _handle_result(self, text: str, speaker_name: str, speaker_uid: str, score: float):
        if not text.strip():
            return

        # 仅处理目标说话人
        if speaker_name != "YGZ醒脑片":
            self.log.debug(f"忽略非目标说话人: {speaker_name}")
            return

        # ====== 句子合并逻辑 ======
        # 如果该说话人已有待发送任务，取消它并合并文本
        if speaker_name in self.pending_tasks:
            self.pending_tasks[speaker_name].cancel()
            # 累积新文本（用空格连接）
            self.pending_texts[speaker_name] = self.pending_texts.get(speaker_name, "") + " " + text
        else:
            # 首次出现，直接设置累积文本
            self.pending_texts[speaker_name] = text

        # 创建新的延迟发送任务
        task = asyncio.create_task(self._delayed_send(speaker_name, speaker_uid))
        self.pending_tasks[speaker_name] = task

    async def _send_to_llm(self, text: str, speaker_name: str, speaker_uid: str):
        """实际发送文本给大模型的逻辑（从原 _handle_result 提取）"""
        if self.callback:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.callback, text, speaker_uid, speaker_name)
        else:
            url = f"{self.api_base}/msg"
            payload = {"msg": text, "uid": speaker_uid, "username": speaker_name}
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as resp:
                        if resp.status == 200:
                            self.log.info(f"已送入 AI 核心: {text[:30]}... (说话人: {speaker_name})")
                        else:
                            self.log.error(f"发送失败: {resp.status}")
            except Exception as e:
                self.log.error(f"HTTP 异常: {e}")

    async def _delayed_send(self, speaker_name: str, speaker_uid: str):
        """延迟发送任务：等待合并超时后发送累积文本"""
        try:
            await asyncio.sleep(self.merge_timeout)
            text = self.pending_texts.pop(speaker_name, "").strip()
            if text:
                await self._send_to_llm(text, speaker_name, speaker_uid)
        except asyncio.CancelledError:
            # 任务被取消，清理累积文本
            self.pending_texts.pop(speaker_name, None)
            raise
        finally:
            # 无论成功或取消，都移除任务引用
            self.pending_tasks.pop(speaker_name, None)