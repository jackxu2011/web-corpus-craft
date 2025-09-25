import pandas as pd
import os
from tqdm import tqdm
import argparse
import data_util
import tldextract
from loguru import logger
import sys

def extract_domain(input_dir: str, output_dir: str):
    files = data_util.get_all_files(input_dir, suffix='.zst', recursive=True)
    df_array=[]
    for file in tqdm(files, desc="deal files"):
        file_name = os.path.splitext(os.path.basename(file))[0]
        output_file_dir = os.path.dirname(file).replace(input_dir, output_dir)
        os.makedirs(output_file_dir, exist_ok=True)
        output_file = f'{output_file_dir}/{file_name}.csv'
        if os.path.exists(output_file):
            logger.info(f'{file} have been dealed')
            continue
        try:
            df = pd.read_json(file, compression='zstd', lines=True)
        except Exception as e:
            logger.error(f"读取文件 {file} 时发生错误: {str(e)}")
            continue

        logger.info(f'{file} load {len(df)} lines')
        if len(df) <= 0:
            continue
        df['domain'] = df.url.apply(data_util.extract_domain)
        group_domain = df.groupby('domain').size().reset_index(name='count')
        logger.info(f'{file} has {len(group_domain)} domain')
        group_domain.to_csv(output_file, index=False)

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    # 输入文件夹不存在则退出
    if not os.path.exists(args.input_dir):
        logger.error(f"文件夹 {args.input_dir} 不存在，退出...")
        sys.exit(1)

    # 文件夹不存在则创建
    if not os.path.exists(args.output_dir):
        logger.warning(f"文件夹 {args.output_dir} 不存在，自动创建...")
        os.makedirs(args.output_dir, exist_ok=True)

    extract_domain(args.input_dir, args.output_dir)
