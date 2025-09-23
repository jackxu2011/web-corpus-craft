import sys
import os

# 将项目根目录添加到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
from tqdm import tqdm
import argparse
import glob
from loguru import logger
from pd_util import drop_duplicates

def merge_csv(input_dir, output_file, columns = [], text_key='text'):
  # 获取当前目录下所有 .csv 文件的路径
  file_paths = glob.glob(os.path.join(input_dir,"*.csv"))  # 可根据实际路径修改
  metric = {
    'total': 0,
    'duplicated': 0,
    'remain': 0
  }
  # 读取每个文件并存入列表
  dataframes = []
  for file in tqdm(file_paths, desc='read files'):

      try:
        df = pd.read_csv(file, on_bad_lines='skip')
        if not columns:
          dataframes.append(df)
        else:
          dataframes.append(df[columns])
      except Exception as e:
        logger.error(f'failed deal file: {file}, 发生未知错误：{str(e)}')

  # 合并所有 DataFrame
  combined_df = pd.concat(dataframes, ignore_index=True)

  metric['total'] = len(combined_df)
  metric['duplicated'] = combined_df.duplicated().sum()

  logger.info(f"原始数据行数: {metric['total']}")
  logger.info(f"重复行数量: {metric['duplicated'] }")
  # 3. 去除重复行
  # 默认保留第一次出现的行，删除后续重复行
  df_cleaned = combined_df.drop_duplicates()

  metric['remain'] = len(df_cleaned)

  logger.info('begin save')
  df_cleaned.to_csv(output_file, index=False)
  logger.info(f"save {metric['remain']} to file:{output_file}")
  return metric

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--columns", type=str, default=[], nargs='*')
    args = parser.parse_args()
    merge_csv(args.input_dir, args.output_file, args.columns)
