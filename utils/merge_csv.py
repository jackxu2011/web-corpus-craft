import pandas as pd
import os
from tqdm import tqdm
import argparse
import glob
from loguru import logger

def merge_csv(input_dir, output_file, columns = []):
  # 获取当前目录下所有 .csv 文件的路径
  file_paths = glob.glob(os.path.join(input_dir,"*.csv"))  # 可根据实际路径修改

  # 读取每个文件并存入列表
  dataframes = []
  for file in tqdm(file_paths, desc='read files'):
      df = pd.read_csv(file)
      if not columns:
        dataframes.append(df)
      else:
        dataframes.append(df[columns])

  # 合并所有 DataFrame
  combined_df = pd.concat(dataframes, ignore_index=True)

  logger.info(f"原始数据行数: {len(combined_df)}")
  logger.info(f"重复行数量: {combined_df.duplicated().sum()}")
  # 3. 去除重复行
  # 默认保留第一次出现的行，删除后续重复行
  df_cleaned = combined_df.drop_duplicates()

  df_cleaned['text'] = df_cleaned.text.apply(lambda x: x.replace('\n', ' '))

  logger.info(f"save {len(df_cleaned)} to file:{output_file}")
  df_cleaned.to_csv(output_file, index=False)

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--columns", type=str, default=[], nargs='*')
    args = parser.parse_args()
    merge_csv(args.input_dir, args.output_file, args.columns)
