import pandas as pd
import os
from loguru import logger
import unidecode
import re
from glob import glob
import tldextract
import gzip
import shutil

def compress_existing_file(input_path, output_path):
    """将普通文件压缩为 gzip 格式"""
    with open(input_path, 'rb') as f_in:  # 读取原始文件（二进制模式）
        with gzip.open(output_path, 'wb') as f_out:  # 写入压缩文件
            shutil.copyfileobj(f_in, f_out)  # 高效复制数据

def extract_domain(url):
    # 提取主域名（自动识别公共后缀如 .com、.co.uk）
    extracted = tldextract.extract(url)
    # 组合主域名和后缀（如 "example" + "com" → "example.com"）
    return extracted.fqdn

def split_list(lst, parts):
    """
    将列表均匀分割为n个子列表

    参数:
        lst: 要分割的原始列表
        parts: 要分割的子列表数量
    返回:
        包含parts个子列表的列表
    """
    # 计算列表长度
    total = len(lst)

    # 处理n大于列表长度的情况
    if parts >= total:
        return [[item] for item in lst] + [[] for _ in range(parts - total)]

    # 计算基本长度和余数
    base_size = total // parts
    remainder = total % parts

    result = []
    start = 0

    for i in range(parts):
        # 前remainder个子列表多一个元素
        end = start + base_size + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end

    return result

def get_all_files(input_dir: str, suffix: str = '.csv', recursive=False):
    """
    获取指定目录下所有符合后缀条件的文件绝对路径

    参数:
        input_dir: 查找的根目录路径。若传入的是文件路径，则直接返回该文件的绝对路径
        suffix: 目标文件的后缀名（含小数点），默认为 '.csv'。例如：'.txt'、'.json'
        recursive: 是否递归查找所有子目录。为True时会遍历input_dir下的所有层级子目录，
                   为False时仅查找input_dir的直接子文件

    返回:
        list[str]: 符合条件的文件绝对路径列表。若目录不存在或无符合条件的文件，返回空列表

    示例:
        >>> get_all_files('/data', '.csv')  # 仅查找/data目录下的.csv文件
        ['/data/file1.csv', '/data/file2.csv']

        >>> get_all_files('/data', '.txt', recursive=True)  # 递归查找所有.txt文件
        ['/data/doc1.txt', '/data/subdir/doc2.txt']

        >>> get_all_files('/data/report.csv')  # 传入文件路径时直接返回该文件
        ['/data/report.csv']
    """
    if not os.path.exists(input_dir):
        logger.error('input dir: {}, not exists!', input_dir)
        return []
    if os.path.isfile(input_dir):
        return [os.path.abspath(input_dir)]

    pattern = os.path.join(input_dir, f'*{suffix}')
    if recursive:
        pattern = os.path.join(input_dir, '**', f'*{suffix}')
    files = glob(pattern, recursive=recursive)
    return [os.path.abspath(file) for file in files]

# 定义清理文本的函数
def clean_text(text: str):
    if pd.isna(text):
        return ""
    text = str(text).lower()  # 强制转为小写
    text = unidecode.unidecode(text)  # 处理乱码
    text = re.sub(r"http\S+", "", text)  # 去除链接
    text = re.sub(r"[^a-z0-9\s,.!?']", "", text)  # 只保留小写字母、数字和基本标点
    text = re.sub(r"\s+", " ", text).strip()  # 合并空格

    text =  text if text else ""

    return text if len(text.split())>4 else ""  # 用"empty"替代空字符串
