
import json
import pandas as pd
import numpy as np
import unidecode
import re

# 定义清理文本的函数
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()  # 强制转为小写
    text = unidecode.unidecode(text)  # 处理乱码
    text = re.sub(r"http\S+", "", text)  # 去除链接
    text = re.sub(r"[^a-z0-9\s,.!?']", "", text)  # 只保留小写字母、数字和基本标点
    text = re.sub(r"\s+", " ", text).strip()  # 合并空格

    text =  text if text else ""

    return text if len(text.split())>4 else ""  # 用"empty"替代空字符串

def clean_filename(filename):
    # 去掉空格和符号（保留字母、数字和点）
    new_filename = ''.join(e for e in filename if e.isalnum() or e in ['.', '_'])
    return new_filename

# 分割列表
def chunk_list(lst, k):
    """
    将列表 lst 切分成长度为 k 的子列表。

    :param lst: 要切分的列表
    :param k: 每个子列表的期望长度
    :return: 包含长度为 k 的子列表的列表
    """
    return [lst[i:i + k] for i in range(0, len(lst), k)]

# 定义截断文本的函数
def truncate_text(text, max_length):
    if len(text) <= max_length:
        return text
    # 按句号截断
    sentences = text.split(".")
    truncated_text = ""
    for sentence in sentences:
        if len(truncated_text) + len(sentence) + 1 <= max_length:
            truncated_text += sentence + "."
        else:
            break
    return truncated_text.strip()

def json_load(file_path, encoding='utf-8'):
    with open(file_path, encoding=encoding) as json_file:
        return json.load(json_file)

def json_write(file, file_path, encoding = 'utf-8'):
    with open(file_path, 'w', encoding = encoding) as json_file:
        json_file.write(json.dumps(file, ensure_ascii=False))

def write_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in data:
            json_record = json.dumps(item, ensure_ascii=False)
            file.write(json_record + '\n')

# 用jsonl保存的文件因为是生成器格式，所以可以减少内存消耗
def read_jsonl(jsonl_file_path):
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid line: {line}. Error: {e}")
                continue


def stratified_sampling_by_probability(df, prob_col, N, bins = np.arange(0, 3.1, 0.3)):
    """
    根据probability列对DataFrame进行分层抽样，确保每个bin中的样本数量尽可能相同。

    参数:
    - df: 输入的pd.DataFrame，要求包含指定的`prob_col`列。
    - prob_col: 指定用于分层抽样的列名。
    - N: 抽样的总样本量。

    返回:
    - 抽样后的pd.DataFrame。
    """


    # 创建cuts列，表示每个probability所属的区间
    df['cuts'] = pd.cut(df[prob_col], bins)

    # 统计每个bin的样本数量
    bin_counts = df.groupby('cuts', observed=False).size()

    # 计算每个bin的目标样本数
    num_bins = len(bin_counts)
    base_samples_per_bin = N // num_bins  # 每个bin的基础样本数
    remainder = N % num_bins  # 需要额外分配的样本数

    # 分配样本数：基础样本数 + 额外分配的样本数（前remainder个bin多分配1个样本）
    sample_counts = pd.Series([base_samples_per_bin] * num_bins, index=bin_counts.index)
    sample_counts.iloc[:remainder] += 1

    # 处理空bin的情况：如果某个bin没有数据，则将其样本数分配给其他bin
    for bin_name in sample_counts.index:
        if bin_counts.get(bin_name, 0) == 0:  # 如果该bin为空
            sample_counts[bin_name] = 0  # 将其样本数设为0
            # 将多余的样本数均匀分配给其他非空bin
            non_empty_bins = sample_counts[sample_counts > 0].index
            extra_samples = sample_counts[bin_name]
            sample_counts[non_empty_bins] += extra_samples // len(non_empty_bins)
            sample_counts[non_empty_bins].iloc[:extra_samples % len(non_empty_bins)] += 1

    # 对每个非空bin进行抽样
    sampled_dfs = []
    for bin_name, n_samples in sample_counts.items():
        if n_samples > 0:  # 只对有样本需求的bin进行抽样
            bin_data = df[df['cuts'] == bin_name]
            if len(bin_data) >= n_samples:  # 如果bin中有足够的样本
                sampled_dfs.append(bin_data.sample(n=n_samples, replace=False))
            else:  # 如果bin中样本不足，则允许重复抽样
                sampled_dfs.append(bin_data.sample(n=n_samples, replace=True))

    # 合并所有抽样结果
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)

    return sampled_df