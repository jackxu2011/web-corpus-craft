import pandas as pd
import os
from tqdm import tqdm
import argparse
import gzip

def split_csv(input_file: str,
              output_dir: str | None = None,
              prefix: str = "data",
              chunk_size: int= 50000,
              file_format="jsonl",
              need_gz: bool = False):
    """
    将CSV文件分割成多个小文件，每个文件包含指定数量的行

    参数:
        input_file (str): 源CSV文件路径
        output_dir: 输出目录
        chunk_size: 每个文件的行数，默认50000
        prefix: 输出文件前缀
        file_format: 保存格式，支持"jsonl","csv"或"parquet"（parquet更高效）
    """
    try:
        # 获取文件名和扩展名
        file_name, file_ext = os.path.splitext(os.path.basename(input_file))
        if not output_dir:
            output_dir = os.path.dirname(input_file)
        else:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)

        # 分块读取CSV文件
        chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)

        # 计数器，用于命名输出文件
        chunk_num = 1

        for chunk in tqdm(chunk_iter, desc='write to split files'):
            # 生成输出文件路径
            output_file = os.path.join(output_dir, f"{prefix}_{chunk_num:04d}.{file_format}")
            if need_gz:
                output_file = f'{output_file}.gz'
                with gzip.open(output_file, 'wt', encoding='utf-8') as f:
                    # 保存当前块到新文件
                    if file_format == "csv":
                        chunk.to_csv(f, index=False)
                    elif file_format == "parquet":
                        chunk.to_parquet(f, index=False)
                    elif file_format == "jsonl":
                        chunk.to_json(f,orient="records", lines=True, force_ascii=False)
                    else:
                        raise ValueError("仅支持 'csv', 'jsonl' 和 'parquet' 格式")
            else:
                # 保存当前块到新文件
                if file_format == "csv":
                    chunk.to_csv(output_file, index=False)
                elif file_format == "parquet":
                    chunk.to_parquet(output_file, index=False)
                elif file_format == "jsonl":
                    chunk.to_json(output_file,orient="records", lines=True, force_ascii=False)
                else:
                    raise ValueError("仅支持 'csv', 'jsonl' 和 'parquet' 格式")
            print(f"已生成: {output_file}，包含 {len(chunk)} 行数据")
            chunk_num += 1

        print(f"分割完成！共生成 {chunk_num - 1} 个文件，保存在 {output_dir} 目录下")
        return True

    except Exception as e:
        print(f"分割文件时出错: {str(e)}")
        return False

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="data")
    parser.add_argument("--size", type=int, default=50000, help="Size of rows for splited files")
    parser.add_argument("--format", type=str, default="csv", help="Save format for splited files")
    parser.add_argument("--need_gz", type=bool, default=False, help="Should gzip the file")
    args = parser.parse_args()
    split_csv(args.input_file, args.output_dir, args.prefix, args.size, args.format, args.need_gz)
