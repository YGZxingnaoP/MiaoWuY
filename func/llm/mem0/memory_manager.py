import os
import threading
import datetime
import json
from typing import List, Dict, Optional, Callable, Any
from mem0 import Memory

# 全局 mem0 客户端（单例）
_mem0_client = None
_mem0_lock = threading.Lock()

def get_mem0_client(config: Optional[Dict] = None):
    """获取或创建全局 mem0 客户端（线程安全）"""
    global _mem0_client
    if _mem0_client is None:
        with _mem0_lock:
            if _mem0_client is None:
                if config is None:
                    # 默认配置：使用 Chroma 本地持久化 + HuggingFace 嵌入模型
                    config = {
                        "vector_store": {
                            "provider": "chroma",
                            "config": {
                                "collection_name": "mem0",
                                "path": "./mem0_data",
                            }
                        },
                        "embedder": {
                            "provider": "huggingface",
                            "config": {
                                "model": "./mem0model",
                            }
                        },
                        "llm": {
                            "provider": "ollama",          # 使用 ollama
                            "config": {
                                "model": "qwen2.5:1.5b",      # 与你 pull 的模型一致
                                "ollama_base_url": "http://localhost:11434"  # ollama 默认地址
                            }
                        }
                    }
                _mem0_client = Memory.from_config(config)
    return _mem0_client


class Mem0Manager:
    """
    基于 mem0 的长期记忆管理器
    - 每 max_pending_rounds 轮对话生成摘要并存入 mem0
    - 构建消息时注入短期记忆（最近 short_term_rounds 轮）和检索到的长期记忆
    """
    _global_pending_dialogues: List[Dict[str, str]] = []   # 元素格式: {"user_id": str, "user": str, "assistant": str}
    _global_lock = threading.Lock()
    _GLOBAL_MAX_PENDING = 10   # 全局触发阈值，可根据需要改为可配置

    def __init__(self, uid: str,
                 max_pending_rounds: int = 10,
                 short_term_rounds: int = 3,
                 summary_generator: Optional[Callable[[str], str]] = None,
                 mem0_config: Optional[Dict] = None,
                 shared_user_id: Optional[str] = None):
        self.uid = str(uid)
        self.shared_user_id = shared_user_id
        self.max_pending_rounds = max_pending_rounds
        self.short_term_rounds = short_term_rounds
        self.summary_generator = summary_generator

        # mem0 客户端（全局单例）
        self.mem0 = get_mem0_client(mem0_config)

        # 短期对话存储（未总结的轮次）
        self.pending_dialogues: List[Dict[str, str]] = []   # 每轮格式 {"user":..., "assistant":...}
        self.lock = threading.Lock()
        self.pending_user_message = None   # 暂存用户消息

    def add_user_message(self, message: str, username: str):
        """记录用户消息，等待 assistant 回复"""
        with self.lock:
            self.pending_user_message = message
            self.pending_username = username

    def add_assistant_message(self, message: str):
        with self.lock:
            if not hasattr(self, 'pending_user_message') or self.pending_user_message is None:
                raise RuntimeError("没有待处理的用户消息，请先调用 add_user_message")
        
            username = self.pending_username
            round_data = {
                "user": self.pending_user_message,
                "assistant": message,
                "username": username,
            }
            # 1. 加入当前用户的短期记忆列表（用于构建最近几轮上下文）
            with self.lock:
                self.pending_dialogues.append(round_data)

            # 2. 加入全局队列（附带用户ID）
            global_round = {
                "user_id": self.uid,
                "username": username,
                "user": self.pending_user_message,
                "assistant": message,
            }
            with self.__class__._global_lock:
                self.__class__._global_pending_dialogues.append(global_round)
                # 检查全局队列长度是否达到阈值
                if len(self.__class__._global_pending_dialogues) >= self.__class__._GLOBAL_MAX_PENDING:
                    # 取出前 max-1 轮用于总结，保留最后一轮继续积累
                    to_summarize = self.__class__._global_pending_dialogues[:-1]
                    self.__class__._global_pending_dialogues = [self.__class__._global_pending_dialogues[-1]]
                    if self.summary_generator and to_summarize:
                        threading.Thread(target=self._generate_and_save_global_summary,
                                         args=(to_summarize,)).start()

        # 3. 注释掉原有的用户独立触发总结的代码（否则会同时触发两种总结）
        # if len(self.pending_dialogues) >= self.max_pending_rounds:
        #     with self.lock:
        #         dialogues_to_summarize = self.pending_dialogues[:-1]
        #         self.pending_dialogues = [self.pending_dialogues[-1]]
        #     if self.summary_generator and dialogues_to_summarize:
        #         threading.Thread(target=self._generate_and_save_summary,
        #                          args=(dialogues_to_summarize,)).start()

        del self.pending_user_message
        del self.pending_username

    def _get_mem0_user_id(self):
        """返回实际用于 mem0 的用户 ID"""
        return self.shared_user_id if self.shared_user_id is not None else self.uid

    def _generate_and_save_summary(self, dialogues: List[Dict[str, str]]):
        """生成摘要并存入 mem0（后台线程执行）"""
        dialogue_text = ""
        for d in dialogues:
            dialogue_text += f"用户：{d['user']}\n"
            dialogue_text += f"助手：{d['assistant']}\n"

        summary = self.summary_generator(dialogue_text)
        if summary:
            safe_summary = summary.replace('\n', ' ').replace('\r', ' ').replace('"', "'")
            if len(summary) > 300:
                safe_summary = escaped_summary[:300] + "…"
            metadata = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "conversation_summary"
            }
            user_id = self._get_mem0_user_id()
            self.mem0.add(summary, user_id=user_id, metadata=metadata)
        print(f"[MEM0] 生成的摘要: {summary}")  # 确认是中文

    def _generate_and_save_global_summary(self, dialogues: List[Dict[str, str]]):
        """全局对话摘要生成，dialogues 包含 user_id, user, assistant"""
        dialogue_text = ""
        for d in dialogues:
            # 加入用户ID，便于摘要理解对话来源
            dialogue_text += f"用户({d['username']})：{d['user']}\n"
            dialogue_text += f"助手：{d['assistant']}\n"

        summary = self.summary_generator(dialogue_text)
        if summary:
            safe_summary = summary.replace('\n', ' ').replace('\r', ' ').replace('"', "'")
            if len(safe_summary) > 300:
                safe_summary = safe_summary[:300] + "…"
            # 使用共享用户ID存储（即构造时传入的 shared_user_id）
            user_id = self.shared_user_id if self.shared_user_id else "global"
            metadata = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "global_conversation_summary"
            }
            self.mem0.add(safe_summary, user_id=user_id, metadata=metadata)
            print(f"[MEM0] 全局摘要已生成: {summary[:300]}...")

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """从 mem0 检索相关记忆，返回记忆文本列表"""
        if not query:
            return []
        user_id = self._get_mem0_user_id()
        print(f"[MEM0] 检索查询: {query}, user_id: {user_id}")
        response = self.mem0.search(query, user_id=user_id, limit=top_k)  # response 是字典
        results = response.get("results", [])  # 提取列表
        memories = [r["memory"] for r in results if "memory" in r]
        return memories

    def build_messages(self, current_user_message: str, username: str, include_long_term: bool = True) -> List[Dict[str, str]]:
        """
        构建发送给 LLM 的消息列表
        - 短期记忆（最近 short_term_rounds 轮）
        - 可选长期记忆（检索 top_k=3 条，合并后插入开头）
        - 当前用户消息
        """
        messages = []

        # 短期记忆
        with self.lock:
            recent_dialogues = self.pending_dialogues[-self.short_term_rounds:] if self.pending_dialogues else []
        for round_data in recent_dialogues:
            messages.append({"role": "user", "content": f"{username}：{round_data['user']}"})
            messages.append({"role": "assistant", "content": round_data["assistant"]})

        # 长期记忆（检索后插入开头）
        if include_long_term:
            relevant = self.retrieve(current_user_message, top_k=3)
            if relevant:
                memory_text = "；".join(relevant)
                print(f"[MEM0 DEBUG] 准备插入记忆: {memory_text}")
                messages.insert(0, {"role": "user", "content": f"[喵呜的记忆：{memory_text}]"})
                print(f"[MEM0 DEBUG] 插入后 messages 前2条: {messages[:2]}")

        messages.append({"role": "user", "content": current_user_message})
        return messages