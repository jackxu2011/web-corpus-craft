import pandas as pd
import os
from tqdm import tqdm
import argparse
from utils import data_util
import tldextract

domains = df.read_csv('domain.csv')['domain'].tolist()

def filter_url(url: str):
    domain = tldextract.extract(url).fqdn
    return domain in domains

def split_csv(input_dir: str, output_file: str):
    files = data_util.get_all_files(input_dir, suffix='.zstd')
    df_array=[]
    for file in tqdm(files, desc="deal files"):
        df = pd.read_csv(file)
        df['remain'] = df.url.apply(filter_url)
        df = df[df['remain']]
        df_array.append(df[['url','text']])
    result = pd.concat(df_array, , ignore_index=True)



# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--prefix", type=str, default="data")
    args = parser.parse_args()
    split_csv(args.input_file, args.output_dir, args.prefix, args.size, args.format)
