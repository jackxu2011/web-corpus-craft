import pd_util
import data_util
from loguru import logger
import pandas as pd
from tqdm import tqdm
import argparse
import os
import sys
import time
import requests
from requests.exceptions import RequestException, Timeout

# -------------------------- 全局配置 --------------------------
# Bloom 服务默认地址（可通过命令行参数覆盖）
DEFAULT_BLOOM_SERVICE_URL = "http://localhost:8000"
# HTTP 请求超时时间（秒）
HTTP_TIMEOUT = 30
# 批量请求大小（避免单次请求体过大，默认1万条/批）
BATCH_SIZE = 30000
# 服务调用重试次数（临时网络错误时重试）
RETRY_TIMES = 2


# -------------------------- 工具函数 --------------------------
def save_dataframe(df, output_path):
    """保存 DataFrame（保留原逻辑）"""
    try:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        df.to_json(output_path, orient="records", lines=True, force_ascii=False, mode='a')
        logger.info(f"保存成功: {output_path}（{len(df)} 行）")
        return True
    except Exception as e:
        logger.error(f"保存失败 {output_path}: {str(e)}")
        return False

def send_bloom_request(url, data, retry=RETRY_TIMES):
    """发送 Bloom 服务请求（带重试逻辑）"""
    for attempt in range(retry + 1):
        try:
            response = requests.post(
                url,
                json=data,
                timeout=HTTP_TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()  # 触发 4xx/5xx 错误
            return response.json()
        except Timeout:
            logger.warning(f"请求 {url} 超时（第 {attempt+1}/{retry+1} 次）")
        except RequestException as e:
            logger.warning(f"请求 {url} 失败（第 {attempt+1}/{retry+1} 次）: {str(e)}")
    # 重试耗尽仍失败，抛出异常
    raise RuntimeError(f"请求 {url} 失败（已重试 {retry} 次）")


# -------------------------- 去重核心类 --------------------------
class Deduplicator:
    def __init__(self, output_dir, target_rows=100_000, dedup_cols=None, chunksize=10_000,
                 bloom_service_url=DEFAULT_BLOOM_SERVICE_URL, text_key='text'):
        self.output_dir = output_dir
        self.target_rows = target_rows
        self.dedup_cols = dedup_cols or ['url', 'text']
        self.bloom_service_url = bloom_service_url.rstrip("/")  # 统一 URL 格式（去掉末尾/）
        self.text_key = text_key
        self.chunksize=chunksize

        # 统计信息
        self.total_processed = 0
        self.total_duplicates = 0
        self.total_unique = 0
        self.total_error_len = 0
        self.output_file_num = 0
        self.shard_size = 10

        # 初始化：检查 Bloom 服务可用性
        self._check_bloom_service()

    def _check_bloom_service(self):
        """检查 Bloom 服务是否正常（调用 /status 接口）"""
        try:
            response = requests.get(f"{self.bloom_service_url}/status", timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            status = response.json()
            logger.info(f"Bloom 服务连接成功！状态：容量={status['shard_capacity'] * status['shard_count'] }, 元素数={status['total_element_count']}")
        except RequestException as e:
            logger.error(f"Bloom 服务连接失败: {str(e)}")
            sys.exit(1)  # 服务不可用，退出程序

    def _bloom_check(self, elements):
        """调用 Bloom 服务 /check 接口，批量检查元素是否存在"""
        if not elements:
            return {}
        # 分批次检查（避免请求体过大）
        url = f"{self.bloom_service_url}/check"
        data = {"elements": elements}
        response_json = send_bloom_request(url, data)
        return response_json["results"]

    def _bloom_add(self, elements):
        """调用 Bloom 服务 /add 接口，批量添加元素"""
        if not elements:
            return

        # 分批次添加
        url = f"{self.bloom_service_url}/add"
        data = {"elements": elements}
        send_bloom_request(url, data)
        logger.debug(f"向 Bloom 服务添加 {len(elements)} 个元素")

    def get_output_path(self):
        """生成输出路径（保留原逻辑）"""
        shard_num = self.output_file_num % self.shard_size + 1
        file_num = self.output_file_num // self.shard_size + 1
        self.output_file_num += 1
        output_dir = os.path.join(self.output_dir, f"shard_{shard_num:02d}")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"shard_{file_num:05d}.jsonl.zst")

    def process_single_file(self, file_path):
        """处理单个文件（核心：Bloom 操作替换为服务调用）"""
        file_name = os.path.basename(file_path)
        try:
            for chunk in pd.read_json(file_path, compression='zstd', lines=True, chunksize=self.chunksize):
                processed_count = len(chunk)

                if processed_count == 0:
                    logger.info(f"{file_name} 是空文件，跳过")
                    return

                if not self.text_key in chunk.columns:
                    logger.info(f"{file_name} 没有 {self.text_key} 列， 跳过")
                    return

                self.total_processed += processed_count

                mask = (chunk[self.text_key].str.len() < 100_000) & (chunk[self.text_key].str.len() > 20)

                error_len = processed_count - mask.sum()

                self.total_error_len += error_len

                logger.info(f"文件 {file_name}：过滤短/长文本 {error_len} 行")
                df = chunk[mask]

                # 1. 计算哈希（保留原逻辑）
                df['_hash'] = df.apply(
                    lambda row: pd_util.generate_record_hash(row, self.dedup_cols), axis=1
                )

                # 2. 块内去重（保留原逻辑）
                unique_in_block = df.drop_duplicates(subset=["_hash"], keep="first")
                intra_dups = len(df) - len(unique_in_block)
                unique_hashes = unique_in_block['_hash'].tolist()  # 待检查的哈希列表

                # 3. 调用 Bloom 服务检查元素是否存在（替换本地判断）
                logger.debug(f"检查 {file_name} 的 {len(unique_hashes)} 个块内唯一哈希")
                exists_results = self._bloom_check(unique_hashes)

                # 4. 生成掩码（区分新元素/重复元素）
                unique_mask = []
                new_hashes = []
                for hash_val in unique_hashes:
                    if exists_results.get(hash_val, True):  # 默认视为重复（避免服务异常）
                        unique_mask.append(False)
                    else:
                        unique_mask.append(True)
                        new_hashes.append(hash_val)

                # 5. 调用 Bloom 服务添加新元素（替换本地 update）
                if new_hashes:
                    self._bloom_add(new_hashes)

                # 6. 应用掩码，获取最终唯一数据
                mask_series = pd.Series(unique_mask, index=unique_in_block.index)
                unique = unique_in_block[mask_series]
                unique_size = mask_series.sum()
                inter_dups = len(unique_in_block) - unique_size

                # 7. 更新统计
                block_total_dups = intra_dups + inter_dups
                self.total_duplicates += block_total_dups
                self.total_unique += unique_size
                logger.info(f"处理 {file_path}: 总行数={processed_count:,}, 块内重复={intra_dups}, 跨块重复={inter_dups}")

                # 8. 清理临时列
                if '_hash' in unique.columns:
                    unique = unique.drop(columns=["_hash"])
                yield unique
        except Exception as e:
            logger.error(f"处理文件 {file_name} 错误: {str(e)}")

    def process_files_in_main_process(self, files):
        """主进程批量处理文件（保留原逻辑）"""
        logger.info(f"开始处理 {len(files)} 个文件，Bloom 服务地址：{self.bloom_service_url}")
        output_path = self.get_output_path()
        current_count = 0

        for file_path in tqdm(files, desc="处理文件进度"):
            for unique_df in self.process_single_file(file_path):
                # 处理数据分片
                if len(unique_df) > 0:
                    unique_size = len(unique_df)
                    save_dataframe(unique_df, output_path)
                    if current_count + unique_size < self.target_rows:
                        current_count += unique_size
                    else:
                        # 重新生成path
                        output_path = self.get_output_path()
                        # 重置累积数据
                        current_count = 0
        # 最终记录 Bloom 服务状态
        self._check_bloom_service()

    def statistic(self):
        """输出统计信息（保留原逻辑）"""
        logger.info("\n" + "-"*50)
        logger.info("                    去重处理结果")
        logger.info("-"*50)
        logger.info(f"处理总条数: {self.total_processed:,}")
        logger.info(f"长度异常条数: {self.total_error_len:,}")
        logger.info(f"重复条数: {self.total_duplicates:,}")
        logger.info(f"总唯一条数: {self.total_unique:,}")
        logger.info(f"输出文件数: {self.output_file_num}")
        if self.total_processed > 0:
            logger.info(f"重复率: {self.total_duplicates/self.total_processed:.2%}")
        logger.info("-"*50 + "\n")

# -------------------------- 主函数 --------------------------
def main(args):
    # 目录检查
    if not os.path.exists(args.input_dir):
        logger.warning(f"输入目录不存在：{args.input_dir}，退出")
        sys.exit(1)
    if not os.path.exists(args.output_dir):
        logger.warning(f"输出目录不存在：{args.output_dir}，自动创建")
        os.makedirs(args.output_dir, exist_ok=True)

    # 获取文件列表
    files = data_util.get_all_files(args.input_dir, suffix='.jsonl.zstd', recursive=True)
    logger.info(f"发现 {len(files)} 个待处理文件")
    if not files:
        logger.info("无待处理文件，退出")
        sys.exit(0)

    # 创建去重器（传入 Bloom 服务地址）
    dedup = Deduplicator(
        output_dir=args.output_dir,
        target_rows=args.target_rows,
        dedup_cols=args.dedup_cols,
        bloom_service_url=args.bloom_service_url
    )

    # 执行去重
    start_time = time.time()
    dedup.process_files_in_main_process(files)
    end_time = time.time()

    # 输出统计
    dedup.statistic()
    logger.info(f"总处理时间: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    # 命令行参数解析（新增 Bloom 服务地址参数）
    parser = argparse.ArgumentParser(description="调用远程 Bloom 服务的 JSONL 去重程序")
    parser.add_argument("input_dir", type=str, help="输入文件目录")
    parser.add_argument("output_dir", type=str, help="输出文件目录")
    parser.add_argument("--dedup_cols", type=str, default=['url'], nargs='*', help="去重列")
    parser.add_argument("--target_rows", type=int, default=100_000, help="每个输出文件目标行数")
    parser.add_argument("--bloom-service-url", type=str, default=DEFAULT_BLOOM_SERVICE_URL,
                        help=f"Bloom Filter 服务地址（默认：{DEFAULT_BLOOM_SERVICE_URL}）")
    args = parser.parse_args()

    main(args)
