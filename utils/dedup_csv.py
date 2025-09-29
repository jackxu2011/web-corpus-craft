import sys
from pathlib import Path

# 将项目根目录添加到系统路径
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, parent_dir)

import os
import pandas as pd
from loguru import logger
import argparse
from pd_util import drop_duplicates

def dedup(
    input_file,
    output_file,
    dedup_cols=['text']
):
    """
    对CSV文件进行全局去重

    参数:
        input_file: 源CSV文件路径
        output_file: 最终去重结果输出路径
        dedup_cols: 用于去重的列名列表
    """
    # 验证输入文件
    if not os.path.isfile(input_file):
        logger.error(f'输入文件不存在: {input_file}')
        return
    if not output_file:
        output_file = input_file

    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"开始处理文件: {input_file}")
    df = pd.read_csv(input_file)
    cleaned_df = drop_duplicates(df, dedup_cols)
    cleaned_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--dedup_cols", type=str, default=['text'], nargs='*')
    args = parser.parse_args()
    dedup(
        input_file=args.input_file,
        output_file=args.output_file,
        dedup_cols=args.dedup_cols
    )
