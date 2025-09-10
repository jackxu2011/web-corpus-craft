import pandas as pd
import os
from tqdm import tqdm
from loguru import logger
from pandas import DataFrame
import unidecode
import re
from glob import glob
import zstandard as zstd
import io
import tldextract
import gzip
import shutil

base_root = '/work'
base_home='/work/group1/jack'

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

def read_file(file: str, format: str = 'csv', is_zstd: bool = False) -> DataFrame:
    """
    读取不同格式的文件，支持zstd压缩

    参数:
        file: 文件路径
        format: 文件格式，支持 'csv', 'jsonl', 'parquet'
        zstd: 是否为zstd压缩文件

    返回:
        pandas DataFrame
    """
    # 检查文件是否存在
    if not os.path.isfile(file):
        raise ValueError(f'输入文件不存在: {file}')

    # 根据文件格式读取数据
    if format == "csv":
        if is_zstd:
            df = pd.read_csv(file, compression='zstd')
        else:
            df = pd.read_csv(file)
    elif format == "jsonl":
        if is_zstd:
            df = pd.read_json(file, lines=True, compression='zstd')
        else:
            df = pd.read_json(file, lines=True)
    elif format == "parquet":
        df = pd.read_parquet(file)
    else:
        raise ValueError("仅支持 'csv', 'jsonl' 和 'parquet' 格式")
    return df

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


def append_to_csv(file_path, new_data, index=False, **kwargs):
    """
    向CSV文件追加数据

    参数:
        file_path (str): CSV文件路径
        new_data (pd.DataFrame): 要追加的数据
        index (bool): 是否写入索引，默认为False
        **kwargs: 传递给to_csv的其他参数
    """
    try:
        # 检查文件是否存在
        file_exists = os.path.isfile(file_path)

        # 如果文件存在，不写入表头；否则写入表头
        new_data.to_csv(
            file_path,
            mode='a',
            header=not file_exists,
            index=index,** kwargs
        )
        logger.info(f"成功向CSV文件追加了 {len(new_data)} 行数据")
        return True
    except Exception as e:
        logger.error(f"追加到CSV时出错: {str(e)}")
        return False

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

# pandas 去除重复行
def drop_duplicates(input_df: DataFrame, columns=['text']):
  # 1. 查看重复行数量（可选）
  logger.info(f"原始数据行数: {len(input_df)}")
  logger.info(f"重复行数量: {input_df.duplicated(subset=columns).sum()}")

  # 2. 去除重复行
  # 默认保留第一次出现的行，删除后续重复行
  df_cleaned = input_df.drop_duplicates(subset=columns)

  # 3. 查看处理后的结果
  logger.info(f"去重后的数据行数: {len(df_cleaned)}")
  return df_cleaned
