# tts语音合成
import uuid
import logging
import traceback
import re
import subprocess
import queue
import os
import threading
import time
import sys
import atexit
from threading import Lock
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from func.log.default_log import DefaultLog
from func.vtuber.emote_oper import EmoteOper
from func.vtuber.action_oper import ActionOper
from func.tools.string_util import StringUtil
from func.translate.duckduckgo_translate import DuckduckgoTranslate
from func.tts.gtp_vists import GtpVists
from func.tts.bert_vits2 import BertVis2
from func.tts.edge_tts_vits import EdgeTTs
from func.tts.edge_tts_pro import EdgeTtsPro
from func.tts.player import MpvPlay
from func.obs.obs_init import ObsInit
from func.tools.singleton_mode import singleton
from func.gobal.data import TTsData
from func.gobal.data import LLmData
from func.gobal.data import SingData
from func.config.default_config import defaultConfig

@singleton
class TTsCore:
    # 设置控制台日志
    log = DefaultLog().getLogger()

    mpvPlay = MpvPlay()  # 播放器
    emoteOper = EmoteOper()  # 表情
    actionOper = ActionOper()  # 动作
    duckduckgoTranslate = DuckduckgoTranslate()  # 翻译

    ttsData = TTsData()  # tts数据
    llmData = LLmData()  # llm数据
    singData = SingData()  # 唱歌数据
    # 选择语音
    select_vists = ttsData.select_vists
    if select_vists == "gpt-sovits":
        vists = GtpVists()
    elif select_vists == "bert-vists":
        vists = BertVis2()
    elif select_vists == "edge-tts":
        vists = EdgeTTs()
    elif select_vists == "edge-tts-pro":
        vists = EdgeTtsPro()
    else:
        vists = GtpVists()


    def __init__(self):
        self.proxy_process = None
        self._start_proxy()
        self.obs = ObsInit().get_ws()
        # 播放队列（文件路径）
        self.play_queue = queue.Queue()
        # 字幕队列（字幕JSON）
        self.subtitle_queue = queue.Queue()
        # 保护 SayCount 的锁（已有 ttsData.SayCount，但为了线程安全，可以加锁）
        self.count_lock = Lock()

        # 顺序控制结构
        self.pending_lock = Lock()
        self.pending_segments = {}   # traceid -> 状态字典

        # 启动播放线程
        self.play_thread = Thread(target=self._play_worker, daemon=True)
        self.play_thread.start()
        # 启动字幕线程
        self.subtitle_thread = Thread(target=self._subtitle_worker, daemon=True)
        self.subtitle_thread.start()

        # 加载复读配置
        config = defaultConfig().get_config()
        speech_config = config.get('speech', {})
        self.repeat_enabled = speech_config.get('repeat_enabled', False)
        self.repeat_timeout = speech_config.get('repeat_timeout', 2)

    def _start_proxy(self):
        """启动 pproxy 代理服务"""
        # 读取代理配置，判断是否需要启动
        config = defaultConfig().get_config()
        proxy_addr = config.get("translate", {}).get("HttpProxies", None)
        if not proxy_addr:
            return  # 未配置代理，不启动

        # 解析代理地址，获取端口
        # 假设格式为 http://127.0.0.1:8080
        if not proxy_addr.startswith("http://"):
            return
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(proxy_addr)
            host = parsed.hostname
            port = parsed.port
            if host != "127.0.0.1" or not port:
                return
        except:
            return

        # 检查端口是否已被占用（避免重复启动）
        if self._is_port_in_use(port):
            print(f"代理端口 {port} 已被占用，假设已有代理运行")
            return

        # 启动 pproxy 子进程
        try:
            # 命令：runtime\python.exe -m pproxy -l http://127.0.0.1:8080
            # 如果需要通过现有 SOCKS5 转发，可以添加 -r 参数
            # 这里以最简单的纯 HTTP 代理为例
            python_exe = os.path.join(os.path.dirname(sys.executable), "python.exe")
            cmd = [python_exe, "-m", "pproxy", "-l", proxy_addr]
            self.proxy_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            print(f"代理服务已启动: {proxy_addr}")
            # 注册退出时关闭
            atexit.register(self._stop_proxy)
        except Exception as e:
            print(f"启动代理失败: {e}")

    def _is_port_in_use(self, port):
        """检查端口是否被占用"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return False
            except OSError:
                return True

    def _stop_proxy(self):
        """终止代理进程"""
        if self.proxy_process:
            try:
                self.proxy_process.terminate()
                self.proxy_process.wait(timeout=3)
                print("代理服务已停止")
            except:
                self.proxy_process.kill()
            self.proxy_process = None

    def _play_worker(self):
        """顺序播放音频文件的线程，带可配置的复读机制"""
        last_played = None  # 存储上一个播放的文件路径及标记 (file_path, is_last)
        while True:
            if last_played is not None:
                # 处于等待复读状态，尝试获取新文件
                try:
                    file_path, is_last = self.play_queue.get(timeout=self.repeat_timeout)
                except queue.Empty:
                    # 超时，执行复读（如果启用）
                    if self.repeat_enabled:
                        self.log.info(f"{self.repeat_timeout}秒内无新语音，复读: {last_played[0]}")
                        try:
                            self.mpvPlay.mpv_play("mpv.exe", last_played[0], 100, "0")
                        except Exception as e:
                            self.log.exception(f"复读播放失败: {last_played[0]}")
                        finally:
                            try:
                                os.remove(last_played[0])
                            except:
                                pass
                    else:
                        # 复读禁用，直接删除等待的文件
                        try:
                            os.remove(last_played[0])
                        except:
                            pass
                    last_played = None
                    continue
                else:
                    # 有新文件，删除等待复读的文件
                    try:
                        os.remove(last_played[0])
                    except:
                        pass
                    last_played = None
                    self.log.info(f"开始播放: {file_path}")
            else:
                # 正常等待新文件
                file_path, is_last = self.play_queue.get()
                self.log.info(f"开始播放: {file_path}")

            # 播放新文件
            try:
                self.mpvPlay.mpv_play("mpv.exe", file_path, 100, "0")
            except Exception as e:
                self.log.exception(f"播放失败: {file_path}")
                try:
                    os.remove(file_path)
                except:
                    pass
                continue

            # 播放成功，根据是否为最后一句和复读配置决定后续行为
            if is_last:
                # 最后一句，直接删除
                try:
                    os.remove(file_path)
                except:
                    pass
                last_played = None
            else:
                if self.repeat_enabled:
                    # 非最后一句且复读启用，进入等待复读状态
                    last_played = (file_path, is_last)
                else:
                    # 复读禁用，直接删除
                    try:
                        os.remove(file_path)
                    except:
                        pass
                    last_played = None

    def _subtitle_worker(self):
        """顺序处理字幕的线程"""
        while True:
            reply_json = self.subtitle_queue.get()
            self.ttsData.ReplyTextList.put(reply_json)
            self.log.info(reply_json)

    def _add_segment(self, traceid, seg_index, total, file_path, reply_json, is_end=False):
        """将分段添加到顺序缓冲区，当轮到它时自动放入播放和字幕队列"""
        with self.pending_lock:
            if traceid not in self.pending_segments:
                self.pending_segments[traceid] = {
                    "next": 0,
                    "total": total,
                    "buffer": {},
                    "lock": Lock(),
                    "traceid": traceid
                }
            tracker = self.pending_segments[traceid]

        with tracker["lock"]:
            tracker["buffer"][seg_index] = (file_path, reply_json, is_end)
            self._flush_buffer(tracker)

    def _flush_buffer(self, tracker):
        while tracker["next"] in tracker["buffer"]:
            idx = tracker["next"]
            file_path, reply_json, is_end = tracker["buffer"].pop(idx)
            self.log.info(f"[{tracker['traceid']}] 播放片段 {idx+1}/{tracker['total']}")
            self.subtitle_queue.put(reply_json)
            self.play_queue.put((file_path, is_end))
            tracker["next"] += 1

            # 如果 total 未知且当前片段是 end，则立即清理
            if tracker["total"] == -1 and is_end:
                with self.pending_lock:
                    if tracker["traceid"] in self.pending_segments:
                        del self.pending_segments[tracker["traceid"]]
                return  # 已清理，退出

        # 如果 total 已知且所有分段已播放，则清理
        if tracker["total"] != -1 and tracker["next"] >= tracker["total"]:
            with self.pending_lock:
                if tracker["traceid"] in self.pending_segments:
                    del self.pending_segments[tracker["traceid"]]
    # 直接合成语音播放
    def tts_say(self,text):
        try:
            traceid = str(uuid.uuid4())
            json =  {"voiceType":"other","traceid":traceid,"chatStatus":"end","question":"","text":text,"lanuage":""}
            self.tts_say_do(json)
        except Exception as e:
            self.log.exception("【tts_say】发生了异常：")

    # 直接合成语音播放-聊天用
    def tts_chat_say(self,json):
        try:
            self.tts_say_do(json)
        except Exception as e:
            #self.is_tts_ready = True
            #self.llmData.is_stream_out = False
            self.log.exception(f"【tts_chat_say】发生了异常：")

    # 直接合成语音播放 {"question":question,"text":text,"lanuage":"ja"}
    def tts_say_do(self,json):

        # 提取字段（包括可选的 seg_index/total_segments）
        seg_index = json.get("seg_index", 0)
        total_segments = json.get("total_segments", 1)
        is_segmented = "seg_index" in json   # 判断是否为分段任务

        # 安全递增 SayCount（使用锁）
        with self.count_lock:
            self.ttsData.SayCount += 1
            filename = f"say{self.ttsData.SayCount}"

        question = json["question"]
        text = json["text"]
        replyText = text
        lanuage = json["lanuage"]
        voiceType = json["voiceType"]
        traceid = json["traceid"]
        chatStatus = json["chatStatus"]

        # 退出标识
        if text == "" and chatStatus == "end":
            replyText_json = {"traceid": traceid, "chatStatus": chatStatus, "text": ""}
            self.subtitle_queue.put(replyText_json)  # 通过队列处理
            self.log.info(replyText_json)
            return

        # 识别表情
        jsonstr = self.emoteOper.emote_content(text)
        self.log.info(f"[{traceid}]输出表情{jsonstr}")
        emotion = "happy"
        if len(jsonstr) > 0:
            emotion = jsonstr[0]["content"]

        # 感情值增加
        moodNum = self.emoteOper.mood(emotion)

        # 触发翻译日语
        if lanuage == "AutoChange":
            self.log.info(f"[{traceid}]当前感情值:{moodNum}")
            if re.search(".*日(文|语).*", question) or re.search(".*日(文|语).*说.*", text):
                trans_json = self.duckduckgoTranslate.translate(text, "zh-Hans", "ja")
                if StringUtil.has_field(trans_json, "translated"):
                    text = trans_json["translated"]
            elif re.search(".*英(文|语).*", question) or re.search(
                    ".*英(文|语).*说.*", text
            ):
                trans_json = self.duckduckgoTranslate.translate(text, "zh-Hans", "en")
                if StringUtil.has_field(trans_json, "translated"):
                    text = trans_json["translated"]
            elif moodNum > 270 or emotion == "angry":
                trans_json = self.duckduckgoTranslate.translate(text, "zh-Hans", "ja")
                if StringUtil.has_field(trans_json, "translated"):
                    text = trans_json["translated"]

        # 合成语音
        pattern = "(《|》|（|）)"  # 过滤特殊字符，这些字符会影响语音合成
        text = re.sub(pattern, "", text)

        status = self.vists.get_vists(filename, text, emotion)
        if status == 0:
            return
        if question != "":
            self.obs.show_text("状态提示", f'{self.llmData.Ai_Name}语音合成"{question}"完成')


        # 判断同序列聊天语音合成时候，其他语音合成任务等待
        # if voiceType!="chat":
        #     while self.llmData.is_stream_out==True:
        #         time.sleep(1)

        # ============ 【线程锁】播放语音【时间会很长】 ==================
        #self.ttsData.say_lock.acquire()
        #self.ttsData.is_tts_ready = False
        #if chatStatus == "start":
            #self.llmData.is_stream_out = True

        # 输出表情
        emote_thread = Thread(target=self.emoteOper.emote_show, args=(jsonstr,))
        emote_thread.start()

        # 输出回复字幕
        replyText_json = {"traceid": traceid, "chatStatus": chatStatus, "text": replyText}
        self.subtitle_queue.put(replyText_json)

        # 循环摇摆动作
        yaotou_thread = Thread(target=self.actionOper.auto_swing)
        yaotou_thread.start()

        # 将音频文件路径放入播放队列（由播放线程顺序播放）
        # 构建音频路径（使用 os.path.join 更安全）
        audio_file = os.path.join(".", "output", f"{filename}.mp3")
        # 构建字幕 JSON
        replyText_json = {"traceid": traceid, "chatStatus": chatStatus, "text": replyText}

        if is_segmented:
            # 分段任务：交给顺序缓冲区，并标记是否为 end 段
            self._add_segment(traceid, seg_index, total_segments, audio_file, replyText_json, is_end=(chatStatus=="end"))
        else:
            # 非分段任务（如欢迎语）：直接入队
            self.subtitle_queue.put(replyText_json)
            is_last = (chatStatus == "end")   # 标记是否为最后一句
            self.play_queue.put((audio_file, is_last))
        # ========================= end =============================

        # 删除语音文件
        #subprocess.run(f"del /f .\output\{filename}.mp3 1>nul", shell=True)

    # 语音合成线程池
    tts_chat_say_pool = ThreadPoolExecutor(
        max_workers=2, 
        thread_name_prefix="tts_chat_say"
    )
    # 如果语音已经放完且队列中还有回复 则创建一个生成并播放TTS的线程
    def check_tts(self):
        if not self.llmData.AnswerList.empty():
            json = self.llmData.AnswerList.get()
            traceid = json["traceid"]
            text = json["text"]
            self.log.info(
                f"[{traceid}]text:{text},is_tts_ready:{self.ttsData.is_tts_ready},SayCount:{self.ttsData.SayCount},is_singing:{self.singData.is_singing}")
            # 合成语音
            self.tts_chat_say_pool.submit(self.tts_chat_say, json)


    # http接口：聊天回复弹框处理
    def http_chatreply(self):
        status = "失败"
        if not self.ttsData.ReplyTextList.empty():
            json_str = self.ttsData.ReplyTextList.get()
            text = json_str["text"]
            traceid = json_str["traceid"]
            chatStatus = json_str["chatStatus"]
            status = "成功"
        jsonStr = "({\"traceid\": \"" + traceid + "\",\"chatStatus\": \"" + chatStatus + "\",\"status\": \"" + status + "\",\"content\": \"" + text.replace(
            "\"", "'").replace("\r", " ").replace("\n", "<br/>") + "\"})"
        return jsonStr