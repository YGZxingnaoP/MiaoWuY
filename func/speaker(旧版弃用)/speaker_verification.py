import numpy as np
import collections
from resemblyzer import VoiceEncoder, preprocess_wav
from webrtcvad import Vad

class SpeakerVerification:
    """
    声纹识别模块，使用 Resemblyzer 提取说话人embedding，并结合VAD分割语音段。
    """
    def __init__(self, embeddings_path='speaker_embeddings.npz', threshold=0.7):
        self.encoder = VoiceEncoder()
        self.vad = Vad(2)  # 激进程度 0-3，2 适中
        self.embeddings_path = embeddings_path
        self.threshold = threshold
        self.registered_embeddings = {}  # {name: embedding}
        self.load_registered()

        # VAD状态
        self.in_speech = False
        self.speech_frames = []          # 存放当前语音段的原始音频字节
        self.sample_rate = 16000

    def load_registered(self):
        """加载已注册的说话人embedding"""
        try:
            data = np.load(self.embeddings_path, allow_pickle=True)
            self.registered_embeddings = data['embeddings'].item()
            print(f"[声纹] 加载了 {len(self.registered_embeddings)} 个说话人")
        except FileNotFoundError:
            print("[声纹] 未找到注册文件，将创建新的")
            self.registered_embeddings = {}
        except Exception as e:
            print(f"[声纹] 加载失败: {e}")
            self.registered_embeddings = {}

    def save_registered(self):
        """保存注册的embedding到文件"""
        np.savez(self.embeddings_path, embeddings=self.registered_embeddings)
        print(f"[声纹] 已保存 {len(self.registered_embeddings)} 个说话人到 {self.embeddings_path}")

    def register_speaker(self, name, audio_bytes):
        """
        注册新说话人
        :param name: 说话人名称（唯一标识）
        :param audio_bytes: PCM音频数据（16kHz, 16bit, 单声道），应为完整的一句话
        :return: bool 是否成功
        """
        # 转换为float32 [-1, 1]
        wav = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        # 提取embedding
        embed = self.encoder.embed_utterance(wav)
        self.registered_embeddings[name] = embed
        self.save_registered()
        return True

    def vad_process(self, audio_chunk):
        """
        对音频块进行VAD处理，返回语音状态和是否结束一句话
        :param audio_chunk: 60ms的PCM数据块（16000Hz, 16bit, 单声道）
        :return: (is_speech, is_end, speech_bytes)
                 is_speech: 当前块是否包含语音
                 is_end: 是否一句话结束（此时speech_bytes包含整句话的音频）
                 speech_bytes: 如果is_end为True，则返回整句话的音频字节串；否则为None
        """
        # webrtcvad要求帧长10/20/30ms，这里我们将60ms分成两个30ms帧处理
        frame_len = int(self.sample_rate * 0.03) * 2  # 30ms的字节数（16bit）
        frames = [audio_chunk[i:i+frame_len] for i in range(0, len(audio_chunk), frame_len)]
        has_speech = False
        for frame in frames:
            if len(frame) == frame_len:
                if self.vad.is_speech(frame, self.sample_rate):
                    has_speech = True
                    break

        if has_speech:
            if not self.in_speech:
                self.in_speech = True
                self.speech_frames = [audio_chunk]
            else:
                self.speech_frames.append(audio_chunk)
            return True, False, None
        else:
            if self.in_speech:
                self.in_speech = False
                full_speech = b''.join(self.speech_frames)
                self.speech_frames = []
                return False, True, full_speech
            else:
                return False, False, None

    def identify_speaker(self, audio_bytes):
        """
        识别一段音频的说话人
        :param audio_bytes: PCM音频数据（整句话）
        :return: (name, similarity) 如果相似度超过阈值返回注册名，否则返回 ("未知", 最高相似度)
        """
        if len(audio_bytes) < 0.5 * self.sample_rate * 2:  # 少于0.5秒不识别
            return "未知", 0.0
        wav = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        embed = self.encoder.embed_utterance(wav)
        best_name = "未知"
        best_sim = 0.0
        for name, reg_embed in self.registered_embeddings.items():
            sim = np.dot(embed, reg_embed) / (np.linalg.norm(embed) * np.linalg.norm(reg_embed))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim >= self.threshold:
            return best_name, best_sim
        else:
            return "未知", best_sim