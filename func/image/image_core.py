# 搜图功能
from func.tools.singleton_mode import singleton
from func.tools.string_util import StringUtil

from func.image import search_image_util
from func.log.default_log import DefaultLog
from func.obs.obs_init import ObsInit
from func.tts.tts_core import TTsCore
from func.nsfw.nsfw_core import NsfwCore

from func.gobal.data import ImageData
from func.gobal.data import CommonData

import time
import requests
import base64
import random
from PIL import Image
from io import BytesIO
from requests.exceptions import RequestException, Timeout, ConnectionError

@singleton
class ImageCore:
    # 设置控制台日志
    log = DefaultLog().getLogger()

    imageData = ImageData()  # 绘图数据
    commonData = CommonData()  # 公共数据

    ttsCore = TTsCore()  # 语音核心
    nsfwCore = NsfwCore()  # 鉴黄

    def __init__(self):
        # OBS WebSocket 连接失败时跳过，不影响程序运行
        try:
            self.obs = ObsInit().get_ws()
        except Exception:
            self.log.warning("OBS WebSocket 连接失败，将跳过 OBS 相关操作")
            self.obs = None

    # 搜图任务
    def check_img_search(self):
        if not self.imageData.SearchImgList.empty() and self.imageData.is_SearchImg == 2:
            self.imageData.is_SearchImg = 1
            img_search_json = self.imageData.SearchImgList.get()
            self.output_img_thead(img_search_json)
            self.imageData.is_SearchImg = 2

    # 输出图片到虚拟摄像头
    def searchimg_output(self, img_search_json):
        try:
            prompt = img_search_json["prompt"]
            username = img_search_json["username"]
            imgUrl = self.baidu_search_img(prompt)
            self.log.info(f"搜图内容:{{'prompt':{prompt}, 'username':{username}, 'imgUrl':{imgUrl}}}")
            if imgUrl is not None:
                image = self.output_search_img(imgUrl, prompt, username)
                if image is not None:
                    timestamp = int(time.time())
                    path = f"{self.imageData.physical_save_folder}{prompt}_{username}_{timestamp}.jpg"
                    # 保存图片，若目录不存在则跳过保存及OBS显示
                    try:
                        image.convert("RGB").save(path, "JPEG")
                        if self.obs is not None:
                            self.obs.show_image("绘画图片", path)
                    except FileNotFoundError:
                        self.log.error(f"保存图片失败，目录不存在: {path}")
                    except Exception as e:
                        self.log.error(f"保存图片失败: {e}")
                    return 1
            return 0
        except Exception:
            self.log.exception("【searchimg_output】发生了异常：")
            return 0

    # 搜索引擎-搜图任务
    def output_img_thead(self, img_search_json):
        prompt = img_search_json["prompt"]
        username = img_search_json["username"]
        try:
            if self.obs is not None:
                self.obs.show_text("状态提示", f"{self.commonData.Ai_Name}在搜图《{prompt}》")
            status = self.searchimg_output({"prompt": prompt, "username": username})
            if self.obs is not None:
                self.obs.show_text("状态提示", "")
            if status == 1:
                self.ttsCore.tts_say(f"回复{username}：我给你搜了一张图《{prompt}》")
            else:
                self.ttsCore.tts_say(f"回复{username}：搜索图片《{prompt}》失败")
        except Exception:
            self.log.exception("【output_img_thead】发生了异常：")
        finally:
            self.log.info(f"‘{username}’搜图《{prompt}》结束")

    # 图片转换字节流（含所有容错）
    def output_search_img(self, imgUrl, prompt, username):
        # 1. 下载图片
        try:
            response = requests.get(imgUrl, timeout=(5, 60))
            response.raise_for_status()
            img_data = response.content
        except Exception as e:
            self.log.error(f"下载图片失败: {e}")
            return None

        imgb64 = base64.b64encode(img_data)

        # 2. 鉴黄调用（任何异常都视为“通过”）
        try:
            status, nsfw = self.nsfwCore.nsfw_fun(imgb64, prompt, username, 5, "搜图", 0.6)
        except Exception as e:
            self.log.error(f"鉴黄服务不可用，跳过鉴黄: {e}")
            status, nsfw = 1, 0.0

        # 3. 处理鉴黄结果
        if status == -1:
            self.log.info(f"回复{username}：搜图鉴黄失败《{prompt}》-nsfw:{nsfw}，禁止执行")
            self.ttsCore.tts_say(f"回复{username}：搜图鉴黄失败《{prompt}》，禁止执行")
            return None
        if status == 0:
            self.log.info(f"回复{username}：搜图发现黄图《{prompt}》-nsfw:{nsfw}，禁止执行")
            self.ttsCore.tts_say(f"回复{username}：搜图发现黄图《{prompt}》，禁止执行")
            return None

        self.log.info(f"回复{username}：搜图为绿色图片《{prompt}》-nsfw:{nsfw}，输出显示")

        # 4. 图片解码与缩放
        try:
            img = Image.open(BytesIO(img_data))
            img = img.resize((self.imageData.width, self.imageData.height), Image.LANCZOS)
            return img
        except Exception as e:
            self.log.error(f"图片处理失败: {e}")
            return None

    # 百度搜图
    def baidu_search_img(self, query):
        imageNum = self.imageData.imageNum
        # 第一次搜图
        img_json = {"query": query, "width": self.imageData.width, "height": self.imageData.height}
        images = search_image_util.baidu_get_image_url_regx(img_json, imageNum)
        count = len(images)
        self.log.info(f"1.搜图《{query}》数量：{count}")

        # 第二次搜图（放宽高度限制）
        if count < imageNum:
            img_json = {"query": query, "width": self.imageData.width, "height": 0}
            sec = search_image_util.baidu_get_image_url_regx(img_json, imageNum)
            count += len(sec)
            images += sec
            self.log.info(f"2.搜图《{query}》数量：{len(sec)}")

        if count > 0:
            return images[random.randrange(0, count)]
        return None

    # 搜图入口处理
    def msg_deal(self, traceid, query, uid, user_name):
        text = ["搜图", "搜个图", "搜图片", "搜一下图片"]
        is_contain = StringUtil.has_string_reg_list(f"^{text}", query)
        if is_contain is not None:
            num = StringUtil.is_index_contain_string(text, query)
            queryExtract = query[num:].strip()
            self.log.info(f"[{traceid}]搜索图：" + queryExtract)
            if queryExtract == "":
                return True
            img_search_json = {"traceid": traceid, "prompt": queryExtract, "username": user_name}
            self.imageData.SearchImgList.put(img_search_json)
            return True
        return False