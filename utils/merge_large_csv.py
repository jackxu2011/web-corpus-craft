import pandas as pd
import os
from tqdm import tqdm
import argparse
import glob
from loguru import logger
from pd_util import append_to_csv

def merge_csv(input_dir, output_file, columns = []):
  # 获取当前目录下所有 .csv 文件的路径
  file_paths = glob.glob(os.path.join(input_dir,"*.csv"))  # 可根据实际路径修改
  total_rows = 0
  # 读取每个文件并存入列表
  for file in tqdm(file_paths, desc='read files'):
      chunk_iter = pd.read_csv(file, chunksize=100000, on_bad_lines='skip')
      try:
        for chunk in chunk_iter:
          total_rows += len(chunk)
          if not columns:
            append_to_csv(output_file, chunk)
          else:
            append_to_csv(output_file, chunk[columns])
      except Exception as e:
        logger.error(f'failed deal file: {file}, 发生未知错误：{str(e)}')

  logger.info(f"finally save {total_rows} to file:{output_file}")

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--columns", type=str, default=[], nargs='*')
    args = parser.parse_args()
    merge_csv(args.input_dir, args.output_file, args.columns)
