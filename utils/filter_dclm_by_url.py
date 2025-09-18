import pandas as pd
import os
from tqdm import tqdm
import argparse
import data_util
import pd_util
import tldextract
from loguru import logger
import sys

domains = pd.read_csv(os.path.join(data_util.base_home,'domain.csv'))['domain'].tolist()

def is_in_domains(url: str):
    domain = tldextract.extract(url).fqdn
    return domain in domains

def filter_urls(input_dir: str, output_dir: str):
    files = data_util.get_all_files(input_dir, suffix='.zstd')
    df_array=[]
    for file in tqdm(files, desc="deal files"):
        df = pd.read_json(file, compression='zstd', lines=True)
        file_name = os.path.splitext(os.path.basename(file))[0]
        logger.info(f'{file} load {len(df)} lines')
        if len(df) <= 0:
            continue
        df['remain'] = df.url.apply(is_in_domains)
        df = df[df['remain']]
        logger.info(f'{file} remain {len(df)} lines')
        pd_util.append_to_csv(os.path.join(output_dir, f'{file_name}.csv'), df[['url','text']])

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

    filter_urls(args.input_dir, args.output_dir)
