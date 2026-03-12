# 入口操作类：所有功能从这里发起
from func.log.default_log import DefaultLog
from func.config.default_config import defaultConfig
from func.llm.llm_core import LLmCore
from func.sing.sing_core import SingCore
from func.draw.draw_core import DrawCore
from func.nsfw.nsfw_core import NsfwCore
from func.image.image_core import ImageCore
from func.search.search_core import SearchCore
from func.dance.dance_core import DanceCore
from func.cmd.cmd_core import CmdCore
from func.vtuber.action_oper import ActionOper
from func.qwen_vision.qwen_vision_core import QwenVisionCore

from func.obs.obs_init import ObsInit
from func.tools.string_util import StringUtil
from func.tools.singleton_mode import singleton
import time
import uuid

@singleton
class EntranceCore:
    # 设置控制台日志
    log = DefaultLog().getLogger()

    cmdCore = CmdCore()  # 命令操作

    # ============= LLM参数 =====================
    llmCore = LLmCore()  # llm核心
    # ============================================

    # ============= 绘画参数 =====================
    drawCore = DrawCore()
    # ============================================

    # ============= 搜图参数 =====================
    imageCore = ImageCore()
    # ============================================

    # ============= 搜文参数 =====================
    searchCore = SearchCore()
    # ============================================

    # ============= 唱歌参数 =====================
    singCore = SingCore()  # 唱歌核心
    # ============================================

    # ============= 跳舞、表情视频 ================
    danceCore = DanceCore()
    # ============================================

    # ============= 鉴黄 =====================
    nsfwCore = NsfwCore()
    # ============================================

    actionOper = ActionOper()  # 动作核心

    def __init__(self):
        self.obs = ObsInit().get_ws()
        self.qwen_vision = QwenVisionCore()
        self.last_activity_time = time.time()          # 新增：记录最后活动时间
        self.config = defaultConfig().get_config()

    def msg_deal(self, traceid, query, uid, user_name):
        # 更新最后活动时间
        self.last_activity_time = time.time()
        """
        处理弹幕消息
        """
        # traceid = str(uuid.uuid4())
        # 过滤特殊字符
        query = self.nsfwCore.str_filter(query)
        self.log.info(f"[{traceid}]弹幕捕获：[{user_name}]:{query}")  # 打印弹幕信息

        # ---------- 新增：视觉模块触发 ----------
        if self.qwen_vision.enabled:
            self.qwen_vision.check_and_trigger(query)

        # 命令执行
        if self.cmdCore.cmd(traceid, query, uid, user_name):
            return

        # 说话不执行任务
        text = ["\\"]
        num = StringUtil.is_index_contain_string(text, query)  # 判断是不是需要搜索
        if num > 0:
            return

        # 跳舞表情
        if self.danceCore.msg_deal_emotevideo(traceid, query, uid, user_name):
            return

        # 搜索引擎查询
        if self.searchCore.msg_deal(traceid, query, uid, user_name):
            return

        # 搜索图片
        if self.imageCore.msg_deal(traceid, query, uid, user_name):
            return

        # 绘画
        if self.drawCore.msg_deal(traceid, query, uid, user_name):
            return

        # 唱歌
        if self.singCore.msg_deal(traceid, query, uid, user_name):
            return

        # 跳舞
        if self.danceCore.msg_deal_dance(traceid, query, uid, user_name):
            return

        # 换装
        if self.actionOper.msg_deal_clothes(traceid, query, uid, user_name):
            return

        # 切换场景
        if self.actionOper.msg_deal_scene(traceid, query, uid, user_name):
            return

        # 聊天入口处理
        self.llmCore.msg_deal(traceid, query, uid, user_name)

    def check_idle(self):
        """检查空闲超时，若超时则触发自动说话"""
        response_cfg = self.config.get('response', {})
        idle_minutes = response_cfg.get('idle_minutes', 10)
        idle_message = response_cfg.get('idle_message', "主人10分钟没跟你说话了，主人不理你了")

        if time.time() - self.last_activity_time > idle_minutes * 60:
            # 触发自动提醒，并更新最后活动时间避免重复触发
            self.last_activity_time = time.time()
            self.log.info(f"空闲超时触发自动说话: {idle_message}")
            self.llmCore.add_system_message(idle_message, username="主人", uid=0)