import asyncio
import threading
import json
import time
import pyaudio
import websockets
import numpy as np
import collections
import webrtcvad
from typing import Optional, Callable

from func.log.default_log import DefaultLog
from func.config.default_config import defaultConfig
from func.speaker.speaker_verification import SpeakerVerification

class FunASRCore:
    """
    FunASR 实时语音识别客户端（持续监听模式）
    1. 连接 WebSocket 服务器
    2. 捕获麦克风音频并发送（静音时也发送数据，保持连接）
    3. 接收识别结果，最终结果通过回调或 HTTP 发送给 AI 核心
    """

    def __init__(self, callback: Optional[Callable[[str, str, str], None]] = None):
        self.log = DefaultLog().getLogger()
        full_config = defaultConfig().get_config()
        # 以下两个属性保留（变量名不改变），但不再用于匹配逻辑
        self.segment_counter = 0
        self.segment_futures = {}

        # 存放完整句子音频的队列
        self.segment_audio_queue = asyncio.Queue()

        # 读取 FunASR 配置
        self.funasr_config = full_config.get('funasr', {})
        self.enabled = self.funasr_config.get('enabled', False)
        if not self.enabled:
            return

        self.server_url = self.funasr_config.get('server_url', 'ws://127.0.0.1:10095/')
        self.mode = self.funasr_config.get('mode', '2pass')
        self.hotwords = self.funasr_config.get('hotwords', [])
        self.username = self.funasr_config.get('username', '访客')
        self.uid = self.funasr_config.get('uid', 'funasr_user')
        self.silence_seconds = self.funasr_config.get('silence_seconds', 3)
        self.energy_threshold = self.funasr_config.get('energy_threshold', 500)
        self.send_blocks = self.funasr_config.get('send_blocks', 17)
        # 读取省电模式配置（带默认值）
        self.power_save_enabled = self.funasr_config.get('power_save_enabled', False)
        self.power_save_silence_seconds = self.funasr_config.get('power_save_silence_seconds', 30)
        self.power_save_check_interval = self.funasr_config.get('power_save_check_interval', 0.5)
        #vad能量阈值判断
        self.vad_energy_threshold = self.funasr_config.get('vad_energy_threshold', 500)

        # 音频参数
        self.CHUNK = 960                     # 60ms @16kHz (实际为30ms，960字节 = 480样本)
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # 外部回调函数
        self.callback = callback

        # AI 核心接口地址
        from func.gobal.data import CommonData
        self.commonData = CommonData()
        self.api_base = f"http://127.0.0.1:{self.commonData.port}"

        # 读取声纹配置
        sv_config = full_config.get('speaker_verification', {})
        self.sv_enabled = sv_config.get('enabled', False)
        if self.sv_enabled:
            sv_threshold = sv_config.get('similarity_threshold', 0.7)
            sv_embeddings_path = sv_config.get('embeddings_path', 'speaker_embeddings.npz')
            self.speaker_verifier = SpeakerVerification(
                embeddings_path=sv_embeddings_path,
                threshold=sv_threshold
            )
            self.log.info("声纹识别模块已初始化")
        else:
            self.speaker_verifier = None

        # 当前识别到的说话人（线程共享，使用锁保护）
        self.current_speaker_uid = self.uid
        self.current_speaker_username = self.username
        self.speaker_lock = asyncio.Lock()

        # 保留原有变量名（但实际状态由内部局部变量管理）
        self.speech_accumulated = bytearray()
        self.in_speech = False
        self.speech_start_frame = 0
        self.silent_frames = 0

    def start(self):
        """启动后台识别线程"""
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
        """停止识别线程"""
        self.running = False
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
                async with websockets.connect(self.server_url) as ws:
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
        """
        从麦克风读取音频，实现新的句子切分与提交逻辑：
        - 状态机：空闲(IDLE) -> 句子中(IN_SENTENCE) -> 挂起(HANGOVER) -> 提交/返回句子中
        - 静音持续1秒触发进入HANGOVER，再持续1秒静音则提交句子
        - HANGOVER期间若检测到语音，回到IN_SENTENCE继续累积
        - 提高VAD灵敏度（模式1）
        - 新增省电模式：长时间无语音时关闭音频流，定期检测唤醒
        """
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
            self.log.info("麦克风已打开，开始持续发送音频流")

            # 帧率计算（每帧30ms）
            frame_duration = self.CHUNK / (self.RATE * 2)  # 16位2字节，样本数=CHUNK/2，时长=样本数/16000
            frames_per_second = int(1 / frame_duration)    # 约33帧/秒
            # 1秒对应的帧数（取整确保至少1秒）
            one_sec_frames = int(0.5 / frame_duration) + 1  # 例如34帧，实际约1.02秒

            # 句子有效所需的最小语音帧数（1秒）
            min_speech_frames = one_sec_frames
            # 句子内静音触发hangover的阈值（1秒）
            silent_frames_threshold = one_sec_frames
            # hangover状态下静音提交阈值（1秒）
            hangover_threshold = one_sec_frames

            # 环形缓冲区，存储最近1秒的帧（留余量）
            buffer_maxlen = one_sec_frames + 5
            audio_buffer = collections.deque(maxlen=buffer_maxlen)
            flag_buffer = collections.deque(maxlen=buffer_maxlen)

            # 状态变量
            idle_silent_frames = 0          # 空闲态连续静音帧数（用于清空队列和省电）
            consecutive_speech = 0           # 空闲态连续语音帧数（用于触发句子开始）

            in_sentence = False              # 是否在句子中
            in_hangover = False               # 是否在挂起等待中
            sentence_audio = bytearray()      # 当前累积的句子音频
            sentence_speech_frames = 0        # 句子中语音帧数
            sentence_silent_frames = 0        # 句子内连续静音帧数
            hangover_silent_frames = 0        # hangover内连续静音帧数

            # 额外缓冲区，处理音频读取可能返回非整帧
            leftover = b''

            # 清空队列的静音阈值（5秒）
            clear_silence_frames = int(5.0 / frame_duration)

            # 如果启用了声纹，适当提高VAD灵敏度（模式1：略敏感）
            if self.speaker_verifier and hasattr(self.speaker_verifier, 'vad') and hasattr(self.speaker_verifier.vad, 'set_mode'):
                try:
                    self.speaker_verifier.vad.set_mode(1)
                    self.log.debug("VAD模式已设置为1（略敏感）")
                except Exception as e:
                    self.log.warning(f"设置VAD模式失败: {e}，使用默认模式")

            # ========== 新增省电模式相关变量 ==========
            power_save_mode = False
            if self.power_save_enabled:
                power_save_frames = int(self.power_save_silence_seconds / frame_duration)
            else:
                power_save_frames = float('inf')  # 永不进入
            # =======================================

            while self.running:
                if power_save_mode:
                    # ========== 省电模式：周期性检测声音 ==========
                    await asyncio.sleep(self.power_save_check_interval)
                    try:
                        # 临时打开音频流检测
                        p_temp = pyaudio.PyAudio()
                        stream_temp = p_temp.open(
                            format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK,
                            start=False
                        )
                        stream_temp.start_stream()
                        # 读取一小段音频（例如0.2秒），确保整帧数
                        read_frames = int(0.2 * self.RATE / self.CHUNK) * self.CHUNK
                        data = stream_temp.read(read_frames, exception_on_overflow=False)
                        stream_temp.stop_stream()
                        stream_temp.close()
                        p_temp.terminate()

                        # 简单能量检测（或使用VAD）
                        audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                        energy = np.sqrt(np.mean(audio_array ** 2))
                        if energy > self.energy_threshold:  # 能量超过阈值视为有声音
                            self.log.info("检测到声音，退出省电模式")
                            power_save_mode = False
                            # 重新初始化 PyAudio 和流
                            p = pyaudio.PyAudio()
                            stream = p.open(
                                format=self.FORMAT,
                                channels=self.CHANNELS,
                                rate=self.RATE,
                                input=True,
                                frames_per_buffer=self.CHUNK
                            )
                            # 重置状态变量，避免误触发
                            idle_silent_frames = 0
                            consecutive_speech = 0
                            in_sentence = False
                            in_hangover = False
                            sentence_audio = bytearray()
                            sentence_speech_frames = 0
                            sentence_silent_frames = 0
                            hangover_silent_frames = 0
                            # 将检测到的音频数据作为剩余数据，避免丢失开头
                            leftover = data
                            continue
                    except Exception as e:
                        self.log.error(f"省电模式检测异常: {e}")
                        # 出错后等待稍长，避免频繁报错
                        await asyncio.sleep(1)
                    # 继续下一次检测
                    continue

                # ========== 正常模式：从麦克风读取音频 ==========
                # 确保累积到一整帧
                while len(leftover) < self.CHUNK:
                    try:
                        chunk = stream.read(self.CHUNK, exception_on_overflow=False)
                    except Exception as e:
                        self.log.error(f"音频读取错误: {e}")
                        chunk = b''
                    if not chunk:
                        await asyncio.sleep(0.01)
                        continue
                    leftover += chunk
                data = leftover[:self.CHUNK]
                leftover = leftover[self.CHUNK:]

                # 发送给ASR
                await ws.send(data)

                # 计算能量（用于预过滤）
                audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                energy = np.sqrt(np.mean(audio_array ** 2))

                # 能量阈值判断：若能量低于设定值，直接视为静音
                if energy < self.vad_energy_threshold:
                    is_speech = False
                else:
                    # VAD判断
                    try:
                        is_speech = self.speaker_verifier.vad.is_speech(data, self.RATE) if self.speaker_verifier else False
                    except webrtcvad.Error as e:
                        self.log.warning(f"VAD异常: {e}，视为静音")
                        is_speech = False

                # 更新环形缓冲区（用于回溯）
                audio_buffer.append(data)
                flag_buffer.append(is_speech)

                # ========== 状态机处理 ==========
                if in_hangover:
                    # 挂起状态：等待确认是否真的结束
                    sentence_audio.extend(data)          # 继续累积音频（包括静音）
                    if is_speech:
                        # 检测到新语音，回到句子中，合并
                        self.log.debug("Hangover期间检测到语音，继续累积句子")
                        in_hangover = False
                        in_sentence = True
                        sentence_silent_frames = 0
                        hangover_silent_frames = 0
                    else:
                        hangover_silent_frames += 1
                        if hangover_silent_frames >= hangover_threshold:
                            # 确认结束，提交句子
                            if sentence_speech_frames >= min_speech_frames:
                                await self.segment_audio_queue.put(bytes(sentence_audio))
                            # 重置所有状态
                            in_hangover = False
                            in_sentence = False
                            sentence_audio = bytearray()
                            sentence_speech_frames = 0
                            sentence_silent_frames = 0
                            hangover_silent_frames = 0
                            # 空闲静音计数器保留（用于清空队列和省电）
                    # 否则继续hangover

                elif in_sentence:
                    # 句子中：累积音频，监控静音长度
                    sentence_audio.extend(data)
                    if is_speech:
                        sentence_speech_frames += 1
                        sentence_silent_frames = 0
                    else:
                        sentence_silent_frames += 1

                    # 检查是否达到静音阈值，触发进入hangover
                    if sentence_silent_frames >= silent_frames_threshold:
                        in_sentence = False
                        in_hangover = True
                        hangover_silent_frames = 0

                else:
                    # 空闲状态：检测语音开始
                    if is_speech:
                        idle_silent_frames = 0
                        consecutive_speech += 1
                        if consecutive_speech >= 2:   # 连续2帧语音触发开始
                            # 将缓冲区所有历史帧加入句子（回溯约1秒）
                            for hist_frame, hist_flag in zip(audio_buffer, flag_buffer):
                                sentence_audio.extend(hist_frame)
                                if hist_flag:
                                    sentence_speech_frames += 1
                            in_sentence = True
                            consecutive_speech = 0
                            self.log.debug("检测到语音开始，进入句子状态")
                    else:
                        consecutive_speech = 0
                        idle_silent_frames += 1
                        # 空闲状态下长时间无语音，清空待处理队列（防止残留）
                        if idle_silent_frames >= clear_silence_frames:
                            cleared = 0
                            while not self.segment_audio_queue.empty():
                                try:
                                    self.segment_audio_queue.get_nowait()
                                    cleared += 1
                                except asyncio.QueueEmpty:
                                    break
                            if cleared > 0:
                                self.log.info(f"静音超5秒，清空队列，清除了 {cleared} 个待处理音频")
                            idle_silent_frames = 0

                # ========== 检查是否进入省电模式 ==========
                if self.power_save_enabled and not in_sentence and not in_hangover:
                    if idle_silent_frames >= power_save_frames:
                        self.log.info(f"静音超过{self.power_save_silence_seconds}秒，进入省电模式")
                        power_save_mode = True
                        # 关闭当前音频流，释放资源
                        if stream:
                            stream.stop_stream()
                            stream.close()
                        p.terminate()
                        stream = None
                        p = None
                        # 跳过后续休眠，直接进入下一轮循环（省电模式）
                        continue

                # 让步，避免阻塞事件循环
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

    # 保留原方法（变量名不变）
    async def _identify_speaker_async(self, audio_bytes, segment_id):
        pass

    async def _message_receiver(self, ws):
        async for message in ws:
            if isinstance(message, bytes):
                continue
            try:
                data = json.loads(message)
                # 精简日志，只保留关键信息
                text = data.get("text")
                is_final = data.get("is_final", False)
                mode = data.get("mode", "")
                if text and text.strip() and ("offline" in mode or is_final):
                    self.log.info(f"识别到最终文本: {text}")
                    # 无限等待对应的句子音频（确保严格对应）
                    audio_bytes = await self.segment_audio_queue.get()

                    loop = asyncio.get_running_loop()
                    speaker_name, speaker_uid = await loop.run_in_executor(
                        None, self._identify_speaker_sync, audio_bytes
                    )

                    await self._handle_result(text, speaker_name, speaker_uid)

            except json.JSONDecodeError:
                self.log.warning(f"收到非 JSON 消息: {message[:100]}")

    def _trim_silence(self, audio_bytes, sample_rate=16000):
        """
        使用 VAD 切除音频首尾静音，返回只包含主要语音的片段。
        若音频中无语音帧，返回空字节串。
        """
        if not self.speaker_verifier or not hasattr(self.speaker_verifier, 'vad'):
            return audio_bytes  # 无法切除则返回原音频

        vad = self.speaker_verifier.vad
        frame_duration_ms = 30  # 与发送端保持一致
        frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 字节数（16bit）

        # 分帧
        frames = [audio_bytes[i:i+frame_size] for i in range(0, len(audio_bytes), frame_size)]
        speech_flags = []
        for frame in frames:
            if len(frame) == frame_size:
                try:
                    is_speech = vad.is_speech(frame, sample_rate)
                except Exception:
                    is_speech = False
            else:
                is_speech = False
            speech_flags.append(is_speech)

        # 找到第一个和最后一个语音帧索引
        speech_indices = [i for i, flag in enumerate(speech_flags) if flag]
        if not speech_indices:
            return b''

        first = speech_indices[0]
        last = speech_indices[-1]
        start_byte = first * frame_size
        end_byte = (last + 1) * frame_size
        return audio_bytes[start_byte:end_byte]

    def _identify_speaker_sync(self, audio_bytes):
        if not self.speaker_verifier:
            return "未知", "未知"
        # 过滤过短音频（少于1秒）
        if len(audio_bytes) < 32000:  # 1秒字节数
            return "未知", "未知"
        # 切除首尾静音
        trimmed = self._trim_silence(audio_bytes)
        if len(trimmed) < 8000:      
            return "未知", "未知"
        name, sim = self.speaker_verifier.identify_speaker(audio_bytes)
        self.log.info(f"声纹识别结果: {name}, 相似度: {sim:.2f}")
        if name != "未知":
            return name, name
        else:
            return "未知", "未知"

    async def _handle_result(self, text: str, speaker_name: str, speaker_uid: str):
        if not text.strip():
            return

        if self.sv_enabled and speaker_name == "未知":
            # 不输出日志，符合精简要求
            return

        if self.callback:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.callback, text, speaker_uid, speaker_name)
        else:
            url = f"{self.api_base}/msg"
            payload = {
                "msg": text,
                "uid": speaker_uid,
                "username": speaker_name
            }
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as resp:
                        if resp.status == 200:
                            self.log.info(f"已送入 AI 核心: {text[:30]}...")
                        else:
                            self.log.error(f"发送失败: {resp.status}")
            except Exception as e:
                self.log.error(f"HTTP 异常: {e}")