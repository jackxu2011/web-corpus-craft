import pandas as pd
import os
from tqdm import tqdm
import argparse
import data_util
import tldextract
from loguru import logger
import sys

def filter_by_score(input_dir, output_file, low=0, up=5, score_key='traffic_relevance_score', columns=['text']):
  files = data_util.get_all_files(input_dir)
  df_array = []
  columns.insert(0, score_key)
  for file in tqdm(files, desc='deal files'):
    df = pd.read_csv(file)
    logger.info(f'{file} has {len(df)} rows')
    # 将列转换为数值类型，无法转换的设为NaN
    df[score_key] = pd.to_numeric(df[score_key], errors='coerce')

    # 过滤掉转换后为NaN的行（即非数值的行）
    df = df.dropna(subset=[score_key])
    if up is not None:
      df = df[(df[score_key]<=up)]
    if low is not None:
      df = df[(df[score_key]>=low)]
    df = data_util.drop_duplicates(df, columns)
    df_array.append(df[columns])

  combine_df = pd.concat(df_array, ignore_index=True)

  result = data_util.drop_duplicates(combine_df, columns)

  result.to_csv(output_file, index=False)
  logger.info(f'分数在[{low},{up}]之间的数据有{len(result)}条！')

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--low", type=float, default=0)
    parser.add_argument("--up", type=float, default=5)
    parser.add_argument("--score_key", type=str, default='traffic_relevance_score')
    parser.add_argument("--columns", type=str, default=['text'], nargs='*')
    args = parser.parse_args()
    # 输入文件夹不存在则退出
    if not os.path.exists(args.input_dir):
        logger.error(f"文件夹 {args.input_dir} 不存在，退出...")
        sys.exit(1)

    filter_by_score(args.input_dir, args.output_file, args.low, args.up, args.score_key, args.columns)
