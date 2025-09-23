import sys
import os

# 将项目根目录添加到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
import hashlib
from tqdm import tqdm
from loguru import logger
import argparse
from data_util import extract_domain
from data_util import get_all_files

def domain_statistic_for_file(
    input_file,
    output_file,
    chunksize=5_000_000
):
    """
    对大型CSV文件进行全局去重，同时统计块内重复和跨块重复

    参数:
        input_file: 源CSV文件路径
        output_file: 最终去重结果输出路径
        chunksize: 分块大小（行数）
    """
    # 验证输入文件
    if not os.path.isfile(input_file):
        logger.error(f'输入文件不存在: {input_file}')
        return

    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"开始处理文件: {input_file}")

    # 分块读取并处理
    total_urls = 0
    domain_groups = []
    with tqdm(desc=f"统计domain") as pbar:
        for chunk in pd.read_csv(input_file, encoding='utf-8', chunksize=chunksize):
            chunk_len = len(chunk)
            total_urls += chunk_len
            chunk['domain'] = chunk.url.apply(extract_domain)
            grouped_domain = chunk.groupby('domain')['count'].sum().reset_index()
            domain_groups.append(grouped_domain)
            pbar.update(chunk_len)
            pbar.set_postfix({
                "本次处理": f"{chunk_len:,}",
                "新增url": f"{len(grouped_domain):,}"
            })

    logger.info(f'共有url记录： {total_urls} 条')
    cat_df = pd.concat(domain_groups, ignore_index=True)
    df_sum = cat_df.groupby('domain')['count'].sum().reset_index()

    # 排序
    logger.info(f'开始排序')
    sorted_url = df_sum.sort_values(by='count', ascending=False)

    logger.info(f'共有唯一domain数量：{len(df_sum)} 条')

    sorted_url.to_csv(output_file, index=False, encoding='utf-8')

    # 输出最终统计
    logger.info("\n===== 统计结果 =====")
    logger.info(f"处理总数: {total_urls:,} 行")
    logger.info(f"唯一domain数: {len(df_sum):,}")
    logger.info(f"结果已保存至: {output_file}")

def domain_statistic_for_dir(
    input_dir,
    output_file
):
    logger.info('get file list')
    files = get_all_files(input_dir, recursive=True)
    dfs = []
    for index, file in tqdm(enumerate(files), total=len(files), desc='deal files'):
        df = pd.read_csv(file)
        dfs.append(df)
        if (index+1)%1000 == 0:
            cat_df = pd.concat(dfs, ignore_index=True)
            df_sum = cat_df.groupby('domain')['count'].sum().reset_index()
            dfs=[df_sum]
            logger.info(f'have domain: {len(df_sum)}')

    if len(dfs) > 1:
        concated_df = pd.concat(dfs, ignore_index=True)
    else:
        concated_df = dfs[0]
    df_sum = concated_df.groupby('domain')['count'].sum().reset_index()
    logger.info(df_sum.head())
    logger.info(f'共有domain数据：{len(df_sum)}')
    # 排序
    sorted_url = df_sum.sort_values(by='count', ascending=False)

    # 保存处理后的结果到新的 CSV 文件
    logger.info(f'保存数据')
    sorted_url.to_csv(f'{output_file}', index=False)  # index=False 表示不保存索引列
    logger.info(sorted_url.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--chunksize", type=int, default=5_000_000)
    args = parser.parse_args()
    if os.path.isfile(args.input_dir):
        domain_statistic_for_file(
            input_file=args.input_dir,
            output_file=args.output_file,
            chunksize=args.chunksize
        )
    else:
        domain_statistic_for_dir(
            input_dir=args.input_dir,
            output_file=args.output_file
        )
