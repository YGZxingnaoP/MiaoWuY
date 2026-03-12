# -*- coding: utf-8 -*-
"""
纯本地MP3唱歌点播核心（最终修复版）
- 播放期间不持有锁，停止指令立即生效
- 强制终止 mpv 进程树，音乐立刻静音
- 完全对称结构，稳定可靠
"""
import os
import json
import time
import subprocess
import glob
from threading import Thread

from func.tools.singleton_mode import singleton
from func.log.default_log import DefaultLog
from func.gobal.data import SingData
from func.obs.obs_init import ObsInit
from func.tools.string_util import StringUtil

from func.tts.tts_core import TTsCore
from func.llm.llm_core import LLmData
from func.image.image_core import ImageCore
from func.obs.obs_websocket import VideoControl
from func.vtuber.emote_oper import EmoteOper
from func.vtuber.action_oper import ActionOper
from func.tts.player import MpvPlay   # 仅兼容保留


@singleton
class SingCore:
    log = DefaultLog().getLogger()
    singData = SingData()
    llmData = LLmData()
    ttsCore = TTsCore()
    imageCore = ImageCore()
    emoteOper = EmoteOper()
    actionOper = ActionOper()
    mpvPlay = MpvPlay()

    def __init__(self):
        self.obs = ObsInit().get_ws()
        self.current_mpv_process = None   # 当前 mpv 子进程
        self.stop_requested = False       # 停止请求标志

    # -------------------- 唱歌指令 --------------------
    def singTry(self, songname, username):
        try:
            if songname:
                self.sing(songname, username)
        except Exception:
            self.log.exception("【singTry】异常")
            self.singData.is_singing = 2

    def sing(self, songname, username):
        songname = songname.strip()
        if not songname:
            return
        song_dir = f"./output/{songname}/"
        if self.exist_song_queues(self.singData.SongMenuList, songname):
            self.ttsCore.tts_say(f"回复{username}：歌曲《{songname}》已经在歌单中")
            return
        mp3_path = self._find_mp3_file(song_dir, songname)
        if mp3_path:
            self.log.info(f"找到本地歌曲: {mp3_path}")
            self.ttsCore.tts_say(f"回复{username}：好的，播放《{songname}》")
            self.singData.SongMenuList.put({
                "username": username,
                "songname": songname,
                "mp3_path": mp3_path,
                "query": songname
            })
        else:
            self.log.info(f"本地无此歌曲: {song_dir}")
            self.ttsCore.tts_say(f"回复{username}：不知道《{songname}》这首歌曲")

    # -------------------- 停止指令 --------------------
    def stopTry(self):
        """异步处理停止指令"""
        try:
            self.stop_playing()
            self.ttsCore.tts_say("好的，已停止播放")
        except Exception:
            self.log.exception("【stopTry】异常")

    def stop_playing(self):
        """强制终止当前 mpv 进程树，清空队列，恢复背景音乐（立即生效）"""
        try:
            self.singData.play_song_lock.acquire()
            self.stop_requested = True

            if self.singData.is_singing == 1 and self.current_mpv_process:
                self.log.info("正在强制停止唱歌...")
                pid = self.current_mpv_process.pid
                if pid:
                    # 1. 先尝试温和终止
                    try:
                        self.current_mpv_process.terminate()
                        self.current_mpv_process.wait(timeout=1)
                    except Exception:
                        pass

                    # 2. 若进程仍在运行，使用 taskkill 强制杀死进程树
                    if self.current_mpv_process.poll() is None:
                        try:
                            subprocess.run(
                                ['taskkill', '/F', '/T', '/PID', str(pid)],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=3
                            )
                        except Exception:
                            pass

                # 3. 清理进程对象
                self.current_mpv_process = None
                self.singData.sing_play_flag = 0

                # 4. 清空待播队列
                while not self.singData.SongMenuList.empty():
                    self.singData.SongMenuList.get()
                self.singData.SongNowName = {}
                self.singData.is_singing = 2

                # 5. 恢复背景音乐
                self.obs.control_video("背景音乐", VideoControl.PLAY.value)
                self.log.info("唱歌已停止，队列清空")
            else:
                self.log.info("收到停止指令，但当前无播放")
        except Exception:
            self.log.exception("stop_playing 异常")
        finally:
            self.singData.play_song_lock.release()

    # -------------------- 播放核心（静默模式）--------------------
    def play_song(self, songname, mp3_path, username, query):
        """播放单曲，播放期间不持有锁，支持立即打断"""
        try:
            self.log.info(f"开始播放《{songname}》")
            # 搜图、表情、动作
            Thread(target=self.imageCore.searchimg_output,
                   args=({"prompt": query, "username": username},)).start()
            self.emoteOper.emote_ws(1, 0.2, "唱歌")
            self.ttsCore.tts_say(f"回复{username}：我准备唱《{songname}》")
            Thread(target=self.actionOper.auto_swing).start()

            # 启动 mpv 子进程（静默模式）
            self.stop_requested = False
            Thread(target=self._sing_play,
                   args=("mpv.exe", mp3_path, 70, "0")).start()

            # 等待播放结束或被终止（此时锁已释放，停止指令可立即获取锁）
            while self.singData.sing_play_flag == 1 and not self.stop_requested:
                time.sleep(0.3)

            return 1
        except Exception:
            self.log.exception(f"《{songname}》播放异常")
            return 3
        finally:
            self.emoteOper.emote_ws(1, 0.2, "唱歌")

    def _sing_play(self, mpv_name, song_path, volume, start):
        """启动静默 mpv 进程并等待结束"""
        startupinfo = None
        creationflags = 0
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            creationflags = subprocess.CREATE_NO_WINDOW

        cmd = [
            mpv_name,
            '--no-video',
            '--no-terminal',
            '--msg-level=all=no',
            song_path,
            f'--volume={volume}',
            f'--start={start}'
        ]
        self.singData.sing_play_flag = 1
        self.current_mpv_process = subprocess.Popen(
            cmd,
            startupinfo=startupinfo,
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        self.current_mpv_process.wait()
        self.singData.sing_play_flag = 0
        self.current_mpv_process = None

    # -------------------- 队列轮询（锁仅用于临界区）--------------------
    def check_playSongMenuList(self):
        """串行播放队列，播放期间释放锁"""
        if self.singData.SongMenuList.empty() or self.singData.is_singing != 2:
            return

        # 1. 获取队列数据（持有锁）
        self.singData.play_song_lock.acquire()
        try:
            if self.singData.SongMenuList.empty():
                return
            mlist = self.singData.SongMenuList.get()
            self.singData.SongNowName = mlist
            self.singData.is_singing = 1
            # 暂停背景音乐
            self.obs.control_video("背景音乐", VideoControl.PAUSE.value)
        except Exception:
            self.log.exception("获取队列异常")
            self.singData.is_singing = 2
            self.singData.SongNowName = {}
            return
        finally:
            # ⚠️ 关键：释放锁，播放期间不持有锁
            self.singData.play_song_lock.release()

        # 2. 播放歌曲（此时锁已释放，停止指令可立即执行）
        try:
            self.play_song(mlist["songname"], mlist["mp3_path"],
                           mlist["username"], mlist["query"])
        except Exception:
            self.log.exception("播放过程异常")

        # 3. 播放结束后，再次获取锁更新状态
        self.singData.play_song_lock.acquire()
        try:
            # 如果播放被停止，is_singing 可能已被 stop_playing 置为 2
            if self.singData.is_singing == 1:
                self.singData.is_singing = 2
            self.singData.SongNowName = {}

            # 队列空则恢复背景音乐
            if self.singData.SongMenuList.qsize() == 0:
                self.obs.control_video("背景音乐", VideoControl.PLAY.value)
        except Exception:
            self.log.exception("更新播放结束状态异常")
        finally:
            self.singData.play_song_lock.release()

    def check_sing(self):
        """点歌请求队列轮询"""
        if not self.singData.SongQueueList.empty():
            song_json = self.singData.SongQueueList.get()
            self.log.info(f"收到点歌: {song_json}")
            Thread(target=self.singTry,
                   args=(song_json["prompt"], song_json["username"])).start()

    # -------------------- HTTP 接口 --------------------
    def http_sing(self, songname, username):
        self.log.info(f'HTTP点歌: "{username}" 点播《{songname}》')
        self.singData.SongQueueList.put({"prompt": songname, "username": username})

    def http_songlist(self, _):
        jsonstr = []
        if self.singData.SongNowName:
            jsonstr.append({
                "songname": f"'{self.singData.SongNowName['username']}'点播《{self.singData.SongNowName['songname']}》"
            })
        for i in range(self.singData.SongMenuList.qsize()):
            data = self.singData.SongMenuList.queue[i]
            jsonstr.append({"songname": f"'{data['username']}'点播《{data['songname']}》"})
        return f'({"status": "成功","content": {json.dumps(jsonstr)}})'

    # -------------------- 指令识别 --------------------
    def msg_deal(self, traceid, query, uid, user_name):
        if self._handle_stop_command(traceid, query, user_name):
            return True
        if self._handle_sing_command(traceid, query, user_name):
            return True
        return False

    def _handle_stop_command(self, traceid, query, user_name):
        stop_keywords = ["停下", "停止"]
        for kw in stop_keywords:
            if kw in query:
                self.log.info(f"[{traceid}] 停止指令: {query}")
                Thread(target=self.stopTry).start()
                return True
        return False

    def _handle_sing_command(self, traceid, query, user_name):
        keywords = ["唱一下", "唱一首", "唱歌", "点歌", "点播"]
        matched = StringUtil.has_string_reg_list(f"^{keywords}", query)
        if matched is not None:
            idx = StringUtil.is_index_contain_string(keywords, query)
            song_query = query[idx:].strip()
            if song_query:
                self.log.info(f"[{traceid}] 点歌: {song_query}")
                self.singData.SongQueueList.put({
                    "traceid": traceid,
                    "prompt": song_query,
                    "username": user_name
                })
                return True
        return False

    # -------------------- 工具函数 --------------------
    def exist_song_queues(self, queues, name):
        if self.singData.SongNowName and self.singData.SongNowName.get("songname") == name:
            return True
        for i in range(queues.qsize()):
            if queues.queue[i]["songname"] == name:
                return True
        return False

    def _find_mp3_file(self, song_dir, songname):
        if not os.path.isdir(song_dir):
            return None
        exact = os.path.join(song_dir, f"{songname}.mp3")
        if os.path.isfile(exact):
            return exact
        mp3s = glob.glob(os.path.join(song_dir, "*.mp3"))
        return mp3s[0] if mp3s else None