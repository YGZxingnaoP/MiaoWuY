#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将每个 PCM 文件单独转换为对应的 npz 文件（保存在相同文件夹）
用法：python pcm_to_individual.py --folder "speaker_embeddings"
"""

import argparse
import numpy as np
import os
import sys
import glob

try:
    from resemblyzer import VoiceEncoder
except ImportError:
    print("请安装 resemblyzer: pip install resemblyzer")
    sys.exit(1)

def extract_embedding_from_pcm(pcm_path, sample_rate=16000):
    """从 PCM 文件提取说话人 embedding"""
    with open(pcm_path, 'rb') as f:
        pcm_bytes = f.read()
    wav = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    encoder = VoiceEncoder()
    embed = encoder.embed_utterance(wav)
    return embed

def save_individual_embedding(embed, output_path, speaker_name):
    """保存单个 embedding 到 npz 文件（字典格式）"""
    # 创建一个字典，键为说话人名称，值为 embedding
    embeddings_dict = {speaker_name: embed}
    np.savez(output_path, embeddings=embeddings_dict)
    print(f"已保存: {output_path} (说话人: {speaker_name})")

def main():
    parser = argparse.ArgumentParser(description="批量 PCM 转单个 npz 文件（每个 PCM 生成一个对应的 npz）")
    parser.add_argument("--folder", default="speaker_embeddings", help="存放 PCM 文件的文件夹（默认 speaker_embeddings）")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"错误：文件夹 {args.folder} 不存在")
        sys.exit(1)

    # 查找所有 .pcm 文件（不区分大小写）
    pcm_files = glob.glob(os.path.join(args.folder, "*.pcm")) + \
                glob.glob(os.path.join(args.folder, "*.PCM"))
    if not pcm_files:
        print(f"在 {args.folder} 中未找到任何 .pcm 文件")
        sys.exit(1)

    for pcm_file in pcm_files:
        # 提取文件名（不含扩展名）作为说话人名称
        base_name = os.path.splitext(os.path.basename(pcm_file))[0]
        # 输出文件路径：同文件夹下，相同文件名，扩展名为 .npz
        output_file = os.path.join(args.folder, base_name + ".npz")
        print(f"正在处理: {base_name} ({pcm_file})")
        try:
            embed = extract_embedding_from_pcm(pcm_file)
            save_individual_embedding(embed, output_file, base_name)
        except Exception as e:
            print(f"  处理失败: {e}")

if __name__ == "__main__":
    main()