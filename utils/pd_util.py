import pandas as pd
import os
from tqdm import tqdm
from loguru import logger
from pandas import DataFrame
from glob import glob
import zstandard as zstd
import hashlib

def generate_record_hash(record, cols=None, length=8):
    """生成记录的哈希值（用于快速判断重复）"""
    if cols:
        record = record[cols]
    str_repr = '|'.join(map(str, record.values)).encode('utf-8')
    blake_hash = hashlib.blake2b(str_repr, digest_size=length).digest()
    return blake_hash.hex()

def read_file(file: str, format: str = 'csv', compression: str = 'infer') -> DataFrame:
    """
    读取不同格式的文件，支持zstd压缩

    参数:
        file: 文件路径
        format: 文件格式，支持 'csv', 'json', 'jsonl', 'parquet'
        zstd: 是否为zstd压缩文件

    返回:
        pandas DataFrame
    """
    # 检查文件是否存在
    if not os.path.isfile(file):
        raise ValueError(f'输入文件不存在: {file}')

    # 根据文件格式读取数据
    if format == "csv":
        df = pd.read_csv(file, compression=compression)
    elif format == "jsonl":
        df = pd.read_json(file, lines=True, compression=compression)
    elif format == "json":
            df = pd.read_json(file, compression=compressione)
    elif format == "parquet":
        df = pd.read_parquet(file)
    else:
        raise ValueError("仅支持 'csv', 'jsonl', 'json' 和 'parquet' 格式")
    return df

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

def str_length_filter(df: DataFrame, min: int = 20, max: int = 100_000, text_key: str = 'text'):
    logger.info(f"原始数据行数: {len(df)}")
      # 过滤过短/过长文本
    mask = (df[text_key].str.len() <= max) & (df[text_key].str.len() >= min)

    logger.info(f"过滤短/长文本行数: {len(df) - mask.sum()}")
    logger.info(f"过滤后行数: {mask.sum()}")
    return df[mask]
