import pandas as pd
import os
import hashlib
from tqdm import tqdm
from loguru import logger
import argparse

def generate_record_hash(record, cols=None):
    """生成记录的哈希值（用于快速判断重复）"""
    if cols:
        record = record[cols]
    str_repr = '|'.join(map(str, record.values)).encode('utf-8')
    return hashlib.md5(str_repr).hexdigest()

def global_dedup(
    input_file,
    output_path,
    chunksize=1_000_000,  # 每块行数（根据内存调整）
    dedup_cols=['text']
):
    """
    对大型CSV文件进行全局去重，同时统计块内重复和跨块重复

    参数:
        input_file: 源CSV文件路径
        output_path: 最终去重结果输出路径
        chunksize: 分块大小（行数）
        dedup_cols: 用于去重的列名列表
    """
    # 验证输入文件
    if not os.path.isfile(input_file):
        logger.error(f'输入文件不存在: {input_file}')
        return

    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 初始化变量
    global_hash_set = set()       # 全局哈希集合（跨块去重）
    total_processed = 0           # 总处理记录数
    total_duplicates = 0          # 总重复数（块内+跨块合并）
    total_unique = 0              # 最终唯一记录数
    first_write = True

    logger.info(f"开始处理文件: {input_file}")

    # 分块读取并处理
    with tqdm(desc="全局去重进度") as pbar:
        for chunk in pd.read_csv(input_file, chunksize=chunksize):
            chunk_size = len(chunk)
            total_processed += chunk_size

            # 1. 计算当前块所有记录的哈希
            chunk["_hash"] = chunk.apply(
                lambda row: generate_record_hash(row, dedup_cols), axis=1
            )

            # 2. 先去除块内重复（同一块中重复出现的记录）
            # 保留每个哈希在块内的第一条记录
            chunk_unique_in_block = chunk.drop_duplicates(
                subset=["_hash"], keep="first"
            )
            intra_dups = chunk_size - len(chunk_unique_in_block)

            # 3. 再去除跨块重复（与之前块重复的记录）
            mask = ~chunk_unique_in_block["_hash"].isin(global_hash_set)
            new_unique = mask.sum()
            inter_dups = len(chunk_unique_in_block) - new_unique

            block_total_dups = intra_dups + inter_dups
            total_duplicates += block_total_dups

            # 4. 更新统计和全局哈希集合
            total_unique += new_unique
            global_hash_set.update(
                chunk_unique_in_block.loc[mask, "_hash"].tolist()
            )

            # 5. 写入去重后的数据
            final_unique_chunk = chunk_unique_in_block[mask].drop(columns=["_hash"])
            final_unique_chunk.to_csv(
                output_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False
            )
            first_write = False

            # 6. 更新进度条和日志
            pbar.update(chunk_size)
            pbar.set_postfix({
                "总处理": f"{total_processed:,}",
                "总唯一": f"{total_unique:,}",
                "重复率": f"{total_duplicates/total_processed:.2%}"
            })
            logger.info(
                f"块处理完成 | 块大小: {chunk_size:,} | "
                f"本块重复: {block_total_dups:,} | "
                f"本块新增唯一: {new_unique:,}"
            )

    # 输出最终统计
    logger.info("\n===== 完整去重统计结果 =====")
    logger.info(f"原始记录总数: {total_processed:,} 行")
    logger.info(f"重复记录总数: {total_duplicates:,} 行")
    logger.info(f"去重后记录数: {total_unique:,} 行")
    logger.info(f"总重复率: {total_duplicates/total_processed:.2%}")
    logger.info(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--dedup_cols", type=str, default=['text'], nargs='*')
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()
    global_dedup(
        input_file=args.input_file,
        output_path=args.output_file,
        dedup_cols=args.dedup_cols,
        chunksize=args.chunksize
    )
