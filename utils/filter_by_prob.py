import pandas as pd
import os
import hashlib
from tqdm import tqdm
from loguru import logger
import argparse

def filter_by_prob(
    input_file,
    output_file,
    prob=0.9,
    prob_key='prob',
    chunksize=1_000_000
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

    # 初始化变量
    total_processed = 0           # 总处理记录数
    total_remain = 0              # 总保留数
    first_write = True

    logger.info(f"开始处理文件: {input_file}")

    # 分块读取并处理
    with tqdm(desc=f"过滤prob大于{prob}的记录") as pbar:
        for chunk in pd.read_csv(input_file, chunksize=chunksize):
            chunk_size = len(chunk)
            total_processed += chunk_size

            # 1. 过滤prob大于目录值的记录
            mask = chunk[prob_key] >= prob
            new_remain = mask.sum()
            total_remain += new_remain

            # 2. 写入过滤出来的数据
            remain_chunk = chunk[mask]
            remain_chunk.to_csv(
                output_file,
                mode="w" if first_write else "a",
                header=first_write,
                index=False
            )
            first_write = False

            # 3. 更新进度条和日志
            pbar.update(chunk_size)
            pbar.set_postfix({
                "总处理": f"{total_processed:,}",
                "总remain": f"{total_remain:,}",
                f"评分大于{prob}占比": f"{total_remain/total_processed:.2%}"
            })
            logger.debug(
                f"块处理完成 | 块大小: {chunk_size:,} | "
                f"本块新增: {new_remain:,}"
            )

    # 输出最终统计
    logger.info("\n===== 完整去重统计结果 =====")
    logger.info(f"原始记录总数: {total_processed:,} 行")
    logger.info(f"总记录数: {total_remain:,} 行")
    logger.info(f"评分大于{prob}占比: {total_remain/total_processed:.2%}")
    logger.info(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--prob", type=float, default=0.9)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()
    filter_by_prob(
        input_file=args.input_file,
        output_file=args.output_file,
        prob=args.prob,
        chunksize=args.chunksize
    )
