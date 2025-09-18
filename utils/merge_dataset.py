import pandas as pd
import os
from tqdm import tqdm
import argparse
import glob
from loguru import logger

def merge_datasets(input_dir, output_file, text_key = 'text'):
  # 获取当前目录下所有 .csv 文件的路径
  file_paths = glob.glob(os.path.join(input_dir, "*.csv"))  # 可根据实际路径修改

  # 读取每个文件并存入列表
  dataframes = []
  for file in tqdm(file_paths, desc='读取文件'):
      df = pd.read_csv(file)
      dataframes.append(df[[text_key]])

  # 合并所有 DataFrame
  logger.info('begin merge dataframes')
  combined_df = pd.concat(dataframes, ignore_index=True)

  logger.info('begin clean text')
  combined_df[text_key] = combined_df[text_key].apply(data_util.clean_text)
  combined_df = combined_df[(combined_df[text_key] != "")]
  logger.info(combined_df.info())
  #去除重复行
  df_cleaned = pd_util.drop_duplicates(combined_df)

  df_cleaned.to_csv(output_file, index=False)
  logger.info(f"合并后的数据集数据: {len(df_cleaned)}")

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--text_key", type=str, default='text')
    args = parser.parse_args()
    merge_datasets(args.input_dir, args.output_file, args.text_key)
