import pd_util
import data_util
from loguru import logger
import pandas as pd
from tqdm import tqdm
import argparse
import os
import sys
from rbloom import Bloom
from hashlib import sha256
from pickle import dumps
import tempfile
import gc
import time

def hash_func(obj):
    h = sha256(dumps(obj)).digest()
    return int.from_bytes(h[:16], "big", signed=True)

def read_jsonl_file(json_path, text_key="text"):
    """加载文本"""
    df = pd.DataFrame()
    try:
        df = pd.read_json(json_path, compression='zstd', lines=True)
        if text_key in df.columns:
            mask = (df[text_key].str.len() < 100_000) & (df[text_key].str.len() > 20)
            logger.info(f'remove long or short text lines: {len(df) - mask.sum()}')
            df = df[mask]
    except Exception as e:
        logger.error(f"读取文件 {json_path} 时发生错误: {str(e)}")
        with open('logs/failed_file.log', 'a', encoding='utf-8') as f:
            f.write(f'{json_path}\n')
    return df

def save_dataframe(df, output_path):
    """保存 DataFrame 到文件（在主进程中）"""
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # 保存数据
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)

        # 获取行数信息
        row_count = len(df)

        logger.info(f"成功保存: {output_path}，行数: {row_count}")
        return True
    except Exception as e:
        logger.error(f"保存失败 {output_path}: {str(e)}")
        return False

class Deduplicator:
    def __init__(self, output_dir, target_rows=100_000, dedup_cols=None,
                 initial_capacity=100_000_000, error_rate=0.001):
        self.output_dir = output_dir
        self.target_rows = target_rows
        self.dedup_cols = dedup_cols or ['url']
        self.initial_capacity = initial_capacity
        self.error_rate = error_rate
        self.total_processed = 0
        self.total_duplicates = 0
        self.total_unique = 0
        self.output_file_num = 0
        self.shard_size = 10

        # 创建 Bloom Filter 文件路径
        self.bloom_filter_file = os.path.join(tempfile.gettempdir(), 'dedup_bloom_filter.bloom')

        if os.path.exists(self.bloom_filter_file):
            self.global_bloom_filter = self._load_bloom_filter()
        else:
            # 初始化 Bloom Filter
            self.global_bloom_filter = Bloom(initial_capacity, error_rate, hash_func)
            self._save_bloom_filter()

    def _save_bloom_filter(self):
        """保存 Bloom Filter 到临时文件"""
        try:
            self.global_bloom_filter.save(self.bloom_filter_file)
            logger.info(f"保存 Bloom Filter 到文件")
        except Exception as e:
            logger.error(f"保存 Bloom Filter 失败: {e}")

    def _load_bloom_filter(self):
        """从临时文件加载 Bloom Filter"""
        try:
            bloom_filter = Bloom.load(self.bloom_filter_file, hash_func)
            logger.info(f"加载 Bloom Filter 从文件")
            return bloom_filter
        except Exception as e:
            logger.warning(f"加载 Bloom Filter 失败，使用当前实例: {e}")
            return None

    def _merge_bloom_filter(self):
        try:
            old_filter = self._load_bloom_filter()
            self.global_bloom_filter.update(old_filter)
            self._save_bloom_filter()
        except Exception as e:
            logger.warning(f"加载 Bloom Filter 失败，使用当前实例: {e}")
            return None

    def get_output_path(self):
        """生成输出文件路径"""
        shard_num = self.output_file_num % self.shard_size + 1
        file_num = self.output_file_num // self.shard_size + 1
        self.output_file_num += 1
        output_dir = os.path.join(self.output_dir, f"shard_{shard_num:02d}")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"shard_{file_num:05d}.jsonl.zst")

    def process_single_file(self, file_path):
        """在主进程中处理单个文件"""
        file_name = os.path.basename(file_path)

        try:
            df = read_jsonl_file(file_path)
            if len(df) == 0:
                logger.info(f"{file_name} 是空文件，跳过")
                return 0, 0, 0, pd.DataFrame()

            # 计算哈希
            df['_hash'] = df.apply(
                lambda row: pd_util.generate_record_hash(row, self.dedup_cols), axis=1
            )

            # 块内去重
            unique_in_block = df.drop_duplicates(subset=["_hash"], keep="first")
            intra_dups = len(df) - len(unique_in_block)

            # 使用全局 Bloom Filter 过滤
            unique_hashes = unique_in_block['_hash'].tolist()
            unique_mask = []
            new_hashes = []

            for hash_val in unique_hashes:
                if hash_val in self.global_bloom_filter:
                    unique_mask.append(False)
                else:
                    unique_mask.append(True)
                    new_hashes.append(hash_val)

            # 批量更新 Bloom Filter
            if new_hashes:
                self.global_bloom_filter.update(new_hashes)
                # 定期保存 Bloom Filter
                if len(new_hashes) > 1000:
                    self._merge_bloom_filter()

            # 应用过滤器
            mask_series = pd.Series(unique_mask, index=unique_in_block.index)
            unique = unique_in_block[mask_series]
            unique_size = mask_series.sum()
            inter_dups = len(unique_in_block) - unique_size

            processed_count = len(df)
            block_total_dups = intra_dups + inter_dups

            logger.info(f'处理文件{file_path}, 行数：{processed_count:,}, 块内重复{intra_dups}, 块间重复{inter_dups}')

            # 清理临时列
            if '_hash' in unique.columns:
                unique = unique.drop(columns=["_hash"])

            # 强制垃圾回收
            del df, unique_in_block, mask_series
            gc.collect()

            return processed_count, block_total_dups, unique_size, unique

        except Exception as e:
            logger.error(f"处理文件 {file_name} 时发生错误: {str(e)}")
            return 0, 0, 0, pd.DataFrame()

    def process_files_in_main_process(self, files):
        """在主进程中处理所有文件和保存操作"""
        logger.info(f"开始处理 {len(files)} 个文件")

        current_data = []
        current_count = 0

        # 在主进程中逐个处理文件
        for file_path in tqdm(files, desc="处理文件"):
            processed_count, dup_count, unique_count, unique_df = self.process_single_file(file_path)

            if processed_count > 0:
                self.total_processed += processed_count
                self.total_duplicates += dup_count
                self.total_unique += unique_count

                # 如果有唯一数据，处理数据分片
                if unique_count > 0 and len(unique_df) > 0:
                    # 处理数据分片
                    unique_size = len(unique_df)
                    if (unique_size + current_count) < self.target_rows:
                        current_data.append(unique_df)
                        current_count += unique_size
                    else:
                        need_rows = self.target_rows - current_count
                        # 拆分当前块
                        part1 = unique_df.iloc[:need_rows]
                        part2 = unique_df.iloc[need_rows:]

                        if len(part1) > 0:
                            current_data.append(part1)
                            # 在主进程中保存文件
                            output_path = self.get_output_path()
                            save_dataframe(pd.concat(current_data), output_path)

                        # 清理内存
                        current_data = [part2] if len(part2) > 0 else []
                        current_count = len(part2) if len(part2) > 0 else 0

                        # 强制垃圾回收
                        gc.collect()

                # 定期保存 Bloom Filter
                if self.total_processed % 1000000 == 0:
                    self._save_bloom_filter()

        # 保存剩余数据
        if current_count > 0:
            output_path = self.get_output_path()
            save_dataframe(pd.concat(current_data), output_path)

        # 最终保存 Bloom Filter
        self._save_bloom_filter()

    def statistic(self):
        """输出统计信息"""
        logger.info("-" * 10 + "处理结果")
        logger.info("-" * 5 + f"处理总条数: {self.total_processed:,}")
        logger.info("-" * 5 + f"重复条数: {self.total_duplicates:,}")
        logger.info("-" * 5 + f"总唯一条数: {self.total_unique:,}")
        logger.info("-" * 5 + f"总文件数: {self.output_file_num}")
        if self.total_processed > 0:
            logger.info("-" * 5 + f"重复率: {self.total_duplicates/self.total_processed:.2%}")

def main(args):
    # 输入文件夹不存在则退出
    if not os.path.exists(args.input_dir):
        logger.warning(f"文件夹 {args.input_dir} 不存在，退出...")
        sys.exit(1)

    # 文件夹不存在则创建
    if not os.path.exists(args.output_dir):
        logger.warning(f"文件夹 {args.output_dir} 不存在，自动创建...")
        os.makedirs(args.output_dir, exist_ok=True)

    files = data_util.get_all_files(args.input_dir, suffix='.jsonl.zstd', recursive=True)
    logger.info(f'总文件数: {len(files)}')

    # 创建去重器
    dedup = Deduplicator(
        output_dir=args.output_dir,
        target_rows=args.target_rows,
        dedup_cols=args.dedup_cols,
        initial_capacity=args.initial_capacity,
        error_rate=args.error_rate
    )

    # 在主进程中处理所有文件和保存操作
    start_time = time.time()
    dedup.process_files_in_main_process(files)
    end_time = time.time()

    logger.info(f"总处理时间: {end_time - start_time:.2f} 秒")

    # 输出统计信息
    dedup.statistic()

    # 清理临时文件
    # try:
    #     os.remove(dedup.bloom_filter_file)
    #     logger.info("清理临时文件完成")
    # except Exception as e:
    #     logger.warning(f"清理临时文件失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="输入目录")
    parser.add_argument("output_dir", type=str, help="输出目录")
    parser.add_argument("--dedup_cols", type=str, default=['url'], nargs='*', help="去重列")
    parser.add_argument("--target_rows", type=int, default=100_000, help="目标行数")
    parser.add_argument("--initial_capacity", type=int, default=2_000_000_000, help="Bloom Filter 初始容量")
    parser.add_argument("--error_rate", type=float, default=0.001, help="Bloom Filter 误判率")
    args = parser.parse_args()
    main(args)
