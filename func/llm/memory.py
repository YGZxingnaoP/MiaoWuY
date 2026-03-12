import os
import threading
import datetime
import re
import math
import jieba
from collections import Counter
from typing import List, Dict, Optional, Callable, Tuple

class BM25:
    """简单的 BM25 实现，用于计算查询与文档的相关性"""
    def __init__(self, corpus: List[str], tokenizer: Callable[[str], List[str]]):
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.doc_freqs = []      # 每个文档的词频 Counter
        self.idf = {}             # 词的逆文档频率
        self.doc_len = []         # 每个文档的长度（词数）
        self.avgdl = 0.0
        self.k1 = 1.5
        self.b = 0.75
        self._initialize()

    def _initialize(self):
        df = {}
        for doc in self.corpus:
            tokens = self.tokenizer(doc)
            self.doc_len.append(len(tokens))
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            for token in freq:
                df[token] = df.get(token, 0) + 1
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        for token, freq in df.items():
            self.idf[token] = math.log((len(self.corpus) - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query: str) -> List[float]:
        query_tokens = self.tokenizer(query)
        scores = []
        for i in range(len(self.corpus)):
            score = 0.0
            doc_freq = self.doc_freqs[i]
            doc_len = self.doc_len[i]
            for token in query_tokens:
                if token in doc_freq:
                    tf = doc_freq[token]
                    idf = self.idf.get(token, 0.0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * numerator / denominator
            scores.append(score)
        return scores

class MemoryManager:
    # ========== 【新增】jieba 检索黑名单（高频无意义词） ==========
    STOPWORDS = {
        # 已有词
        "主人", "喵呜", "今天", "昨天",

        # 称呼/人称代词
        "你", "我", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
        "自己", "别人", "大家", "人家",

        # 指示代词
        "这", "那", "这里", "那里", "这边", "那边", "这个", "那个", "这些", "那些",

        # 常见虚词（助词、介词、连词等）
        "的", "了", "在", "是", "有", "和", "与", "或", "而", "且", "也", "就", "都",
        "还", "又", "再", "便", "然后", "但是", "可是", "不过", "因为", "所以", "如果",
        "虽然", "即使", "无论", "不仅", "而且", "以及", "于是", "因此", "否则",

        # 时间词（过于宽泛的）
        "现在", "刚才", "之前", "之后", "将来", "未来", "过去", "同时",

        # 语气词/拟声词
        "啊", "哦", "嗯", "呢", "吧", "吗", "呀", "哇", "哎", "哟", "啦", "呗",
        "呵", "哈", "嘿", "唉", "唔", "噢", "喔",

        # 其他高频常见词（无实际意义）
        "一个", "一种", "这个", "那个", "这些", "那些", "这样", "那样", "这么", "那么",
        "怎么", "什么", "为什么", "如何", "怎样", "哪里", "哪儿", "谁"
    }   # 可根据需要扩展
    # =========================================================
    _shared_memories: List[str] = []
    _shared_bm25: Optional[BM25] = None
    _class_lock = threading.Lock()      # 类级别的锁，保护共享资源
    _dirty = False

    def __init__(self, uid: str, long_term_dir: str = "chatrecords",
                 max_pending_rounds: int = 10, short_term_rounds: int = 3,
                 summary_generator: Optional[Callable[[str], str]] = None):
        """
        记忆管理器
        :param uid: 用户ID，用于区分不同用户的记忆文件
        :param long_term_dir: 长期记忆存储目录
        :param max_pending_rounds: 触发总结的对话轮数
        :param short_term_rounds: 短期记忆保留的对话轮数（用于回复）
        :param summary_generator: 用于生成摘要的函数，接受对话文本，返回摘要字符串
        """
        self.uid = str(uid)
        self.long_term_dir = long_term_dir
        self.max_pending_rounds = max_pending_rounds
        self.short_term_rounds = short_term_rounds
        self.summary_generator = summary_generator
         

        # 确保目录存在
        os.makedirs(long_term_dir, exist_ok=True)
        self.long_term_file = os.path.join(long_term_dir, "shared_memory.txt")

        # 初始化类变量（只加载一次）
        with MemoryManager._class_lock:
            if not MemoryManager._shared_memories:   # 首次实例化时加载
                self._load_shared_memory()
            # 每个实例自己的 pending_dialogues（仍按用户隔离）
            self.pending_dialogues: List[Dict[str, str]] = []  # 每轮格式 {"user": "xxx", "assistant": "xxx"}

        # 线程锁
        self.lock = threading.Lock()

    def _tokenize(self, text: str) -> List[str]:
        """分词函数"""
        return list(jieba.cut(text))
        

    def _build_index(self):
        """根据当前长期记忆列表重建 BM25 索引"""
        if self.long_term_memories:
            self.bm25 = BM25(self.long_term_memories, self._tokenize)
        else:
            self.bm25 = None

    def _load_long_term_memory(self):
        """从文件加载长期记忆"""
        if os.path.exists(self.long_term_file):
            with open(self.long_term_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.long_term_memories.append(line)

    def _load_shared_memory(self):
        """从共享文件加载记忆到类变量，并构建 BM25 索引"""
        if os.path.exists(self.long_term_file):
            with open(self.long_term_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        MemoryManager._shared_memories.append(line)
        # 构建索引
        if MemoryManager._shared_memories:
            MemoryManager._shared_bm25 = BM25(MemoryManager._shared_memories, self._tokenize)
        MemoryManager._dirty = False

    def _save_long_term_memory(self, summary: str):
        """保存一条新的长期记忆到文件"""
        now = datetime.datetime.now()
        date_str = f"{now.year}年{now.month}月{now.day}日"
        line = f"\n{date_str} - {summary}\n\n"
        with MemoryManager._class_lock:
            with open(self.long_term_file, "a", encoding="utf-8") as f:
                f.write(f"\n{date_str} - {summary}\n\n")
            # 更新共享列表
            MemoryManager._shared_memories.append(summary)
            # 标记索引需要重建
            MemoryManager._dirty = True

    def _ensure_index(self):
        """确保 BM25 索引最新（检索前调用）"""
        with MemoryManager._class_lock:
            if MemoryManager._dirty and MemoryManager._shared_memories:
                MemoryManager._shared_bm25 = BM25(MemoryManager._shared_memories, self._tokenize)
                MemoryManager._dirty = False

    def add_user_message(self, message: str):
        """记录用户消息（暂存，等待assistant回复）"""
        self.pending_user_message = message

    def add_assistant_message(self, message: str):
        """记录助手回复，形成完整一轮对话，并处理总结触发"""
        if not hasattr(self, 'pending_user_message'):
            raise RuntimeError("没有待处理的用户消息，请先调用add_user_message")

        # 构建一轮对话
        round_data = {
            "user": self.pending_user_message,
            "assistant": message
        }
        with self.lock:
            self.pending_dialogues.append(round_data)

        # 检查是否需要触发总结（达到max_pending_rounds轮）
        if len(self.pending_dialogues) >= self.max_pending_rounds:
            # 取前max_pending_rounds-1轮用于总结，保留最后一轮
            with self.lock:
                dialogues_to_summarize = self.pending_dialogues[:-1]
                self.pending_dialogues = [self.pending_dialogues[-1]]
            if self.summary_generator and dialogues_to_summarize:
                threading.Thread(target=self._generate_and_save_summary,
                                 args=(dialogues_to_summarize,)).start()

        # 清除暂存
        del self.pending_user_message

    def _generate_and_save_summary(self, dialogues: List[Dict[str, str]]):
        """生成摘要并保存（在后台线程运行）"""
        # 将对话列表格式化为文本
        dialogue_text = ""
        for d in dialogues:
            dialogue_text += f"用户：{d['user']}\n"
            dialogue_text += f"助手：{d['assistant']}\n"

        # 调用摘要生成函数
        summary = self.summary_generator(dialogue_text)
        if summary:
            # 限制长度不超过40字
            if len(summary) > 220:
                summary = summary[:220] + "…"
            self._save_long_term_memory(summary)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        根据用户查询检索最相关的长期记忆
        :param query: 用户消息
        :param top_k: 返回最多 top_k 条记忆
        :return: 相关记忆文本列表（按相关性降序）
        """
        self._ensure_index()
        if not MemoryManager._shared_bm25 or not MemoryManager._shared_memories:
            return []
        scores = MemoryManager._shared_bm25.get_scores(query)
        # 获取得分大于 0 的索引并按得分排序
        indexed_scores = [(i, score) for i, score in enumerate(scores) if score > 0]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in indexed_scores[:top_k]]
        return [MemoryManager._shared_memories[i] for i in top_indices]

    def build_messages(self, current_user_message: str, include_long_term: bool = True) -> List[Dict[str, str]]:
        messages = []
        # 短期记忆（最近几轮对话）
        with self.lock:
            recent_dialogues = self.pending_dialogues[-self.short_term_rounds:] if self.pending_dialogues else []

        for round_data in recent_dialogues:
            messages.append({"role": "user", "content": round_data["user"]})
            messages.append({"role": "assistant", "content": round_data["assistant"]})

        # 插入长期记忆作为背景（使用特殊的 user 消息格式）
        if include_long_term:
            relevant_memories = self.retrieve(current_user_message, top_k=3)
            if relevant_memories:
                # 将多条记忆合并为一段文本
                memory_text = "；".join(relevant_memories)
                # 在对话历史开头插入一条背景记忆消息，角色为 user，但内容用特殊标记
                messages.insert(0, {"role": "user", "content": f"[喵呜的记忆：{memory_text}]"})

        # 添加当前用户消息
        messages.append({"role": "user", "content": current_user_message})
        return messages