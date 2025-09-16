import pandas as pd
import os
import hashlib
from tqdm import tqdm
from loguru import logger
import argparse
from data_util import extract_domain

def domain_statistic(
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--chunksize", type=int, default=5_000_000)
    args = parser.parse_args()
    domain_statistic(
        input_file=args.input_file,
        output_file=args.output_file,
        chunksize=args.chunksize
    )
