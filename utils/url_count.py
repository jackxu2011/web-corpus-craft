import pandas as pd
import os
import hashlib
from tqdm import tqdm
from loguru import logger
import argparse

def url_count(
    input_file,
    output_file,
    chunksize=10_000_000
):
    """
    对大型CSV文件进行全局去重，同时统计块内重复和跨块重复

    参数:
        input_file: 源CSV文件路径
        output_file: 最终去重结果输出路径
        prob: fasttext最小可信度
        prob_key: prob分数所在列
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
    url_groups = []
    with tqdm(desc=f"统计url") as pbar:
        for chunk in pd.read_csv(input_file, usecols=['url'], encoding='utf-8', chunksize=chunksize):
            total_urls += len(chunk)
            grouped_url = chunk.groupby('url').size().reset_index(name='count')
            url_groups.append(grouped_url)
            pbar.update(1)
            pbar.set_postfix({
                "本次处理": f"{len(chunk):,}",
                "新增url": f"{len(grouped_url):,}"
            })

    logger.info(f'共有url记录： {total_urls} 条')
    cat_df = pd.concat(url_groups, ignore_index=True)
    df_sum = cat_df.groupby('url')['count'].sum().reset_index()

    # 排序
    logger.info(f'开始排序')
    sorted_url = df_sum.sort_values(by='count', ascending=False)

    logger.info(f'共有唯一url数据：{len(df_sum)} 条')

    sorted_url.to_csv(output_file, index=False, encoding='utf-8')

    # 输出最终统计
    logger.info("\n===== 统计结果 =====")
    logger.info(f"处理总数: {total_urls:,} 行")
    logger.info(f"唯一url数: {len(df_sum):.2%}")
    logger.info(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--chunksize", type=int, default=10_000_000)
    args = parser.parse_args()
    url_count(
        input_file=args.input_file,
        output_file=args.output_file,
        chunksize=args.chunksize
    )
