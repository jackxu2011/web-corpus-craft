import pandas as pd
import os
from tqdm import tqdm
import argparse
import data_util
import pd_util
import tldextract
from loguru import logger
import sys

def filter_by_len(input_dir, output_file, low=6000, text_key='text'):
  columns=[text_key]
  files = data_util.get_all_files(input_dir)
  output_dir = os.path.dirname(output_file)
  os.makedirs(output_dir, exist_ok=True)
  df_array = []
  for file in tqdm(files, desc='deal files'):
    df = pd.read_csv(file)
    logger.info(f'{file} has {len(df)} rows')

    df = pd_util.str_length_filter(df, text_key=text_key)

    # 过滤掉转换后为NaN的行（即非数值的行）
    df = df.dropna(subset=[text_key])
    df = df[(df[text_key].str.len()>=low)]

    df = pd_util.drop_duplicates(df, columns)
    df_array.append(df[columns])

  combine_df = pd.concat(df_array, ignore_index=True)

  result = pd_util.drop_duplicates(combine_df, columns)

  result.to_csv(output_file, index=False)
  logger.info(f'文本长度在{low}以上的数据有{len(result)}条！')

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--low", type=float, default=6000)
    parser.add_argument("--text_key", type=str, default='text')
    args = parser.parse_args()
    # 输入文件夹不存在则退出
    if not os.path.exists(args.input_dir):
        logger.error(f"文件夹 {args.input_dir} 不存在，退出...")
        sys.exit(1)

    filter_by_len(args.input_dir, args.output_file, args.low, args.text_key)
