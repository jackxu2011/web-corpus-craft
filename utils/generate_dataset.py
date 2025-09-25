import pandas as pd
import os
from tqdm import tqdm
import argparse
import gzip
from loguru import logger

def generate_dataset(input_file: str,
              output_dir: str,
              prefix: str = "data",
              chunk_size: int= 150000,
              gziped: bool=False):
    """
    将CSV文件分割成多个小文件，每个文件包含指定数量的行

    参数:
        input_file (str): 源CSV文件路径
        output_dir: 输出目录
        chunk_size: 每个文件的行数，默认150000
        prefix: 输出文件前缀
    """
    try:
        if not os.path.isfile(input_file):
            logger.error(f'input file:{input_file} not exists!')
            return False
        # 获取文件名和扩展名
        file_name, file_ext = os.path.splitext(os.path.basename(input_file))
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 分块读取CSV文件
        chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)

        # 计数器，用于命名输出文件
        chunk_num = 1
        output_ext = 'jsonl.gz' if gziped else 'jsonl'
        for chunk in tqdm(chunk_iter, desc='write to split files'):
            # 生成输出文件路径

            output_file = os.path.join(output_dir, f"{prefix}_{chunk_num:04d}.{output_ext}")
            chunk.to_json(output_file, orient="records", lines=True, force_ascii=False)
            logger.info(f"已生成: {output_file}，包含 {len(chunk)} 行数据")
            chunk_num += 1

        logger.info(f"分割完成！共生成 {chunk_num - 1} 个文件，保存在 {output_dir} 目录下")
        return True

    except Exception as e:
        logger.error(f"分割文件时出错: {str(e)}")
        return False

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--prefix", type=str, default="data")
    parser.add_argument("--size", type=int, default=150000, help="Size of rows for splited files")
    parser.add_argument("--gziped" type=bool default=False)
    args = parser.parse_args()
    generate_dataset(args.input_file, args.output_dir, args.prefix, args.size, args.gziped)
