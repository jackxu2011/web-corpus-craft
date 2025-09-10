import pandas as pd
import os
from tqdm import tqdm
import argparse
import data_util
import tldextract
from loguru import logger
import sys

def sample(input_dir: str, output_file: str, sample: int, format: str = 'csv', zstd: bool = False):
    file_paths = data_util.get_all_files(input_dir)

    batch_sample = min(sample//len(file_paths) + 20, sample + 20)

    dataframes = []
    for file in tqdm(file_paths, desc="处理文件"):
        df = data_util.read_file(file, format, zstd)
        df = df.sample(n=batch_sample)
        df['text'] = df.text.apply(data_util.clean_text)
        df = df[(df['text'] != "")]
        dataframes.append(df.dropna(subset=['text']))

    combined_df = pd.concat(dataframes, ignore_index=True)
    result = combined_df.sample(n=sample)

    logger.info(result.info())
    result['text'].to_csv(f'{output_file}', index=False)

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("sample", type=int)
    parser.add_argument("--format", type=str, default='csv')
    parser.add_argument("--zstd", type=bool, default=False)
    args = parser.parse_args()
    # 输入文件夹不存在则退出
    if not os.path.exists(args.input_dir):
        logger.error(f"文件夹 {args.input_dir} 不存在，退出...")
        sys.exit(1)

    sample(args.input_dir, args.output_file, args.sample, args.format, args.zstd)
