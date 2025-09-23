from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import JsonlReader, WarcReader, ParquetReader, CSVReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from glob import glob
import argparse
import os
from loguru import logger
import csv

csv.field_size_limit(1024 * 1024 * 10)

def get_format_from_path(path):
    """
    根据路径中包含的关键字判断文件格式（不严格依赖后缀）

    Args:
        path: 文件路径（字符串）

    Returns:
        str: 匹配的格式，默认返回'csv'
    """
    # 统一转为小写，实现大小写不敏感匹配
    path_lower = str(path).lower()

    # 定义关键字与格式的映射，按优先级排序（前面的优先匹配）
    format_keywords = [
        ('jsonl', 'jsonl'),
        ('parquet', 'parquet'),
        ('warc', 'warc'),
        # ('tsv', 'tsv'),
        # ('json', 'json')
    ]

    # 遍历关键字列表，返回第一个匹配的格式
    for keyword, fmt in format_keywords:
        if keyword in path_lower:
            return fmt

    # 无匹配时默认返回'csv'
    return 'csv'

def filtering(input_folder: str, output_folder: str, glob_pattern: str, id_key: str = 'id', format: str = None) -> None:
    folder_name = os.path.basename(input_folder.rstrip('/'))
    logger.info(f"开始处理文件夹: {folder_name}")
    files = glob(os.path.join(input_folder, glob_pattern) , recursive= True)
    n_job = len(files)

    if n_job == 0:
        logger.warning(f"文件夹 {folder_name} 中未找到文件，跳过处理")
        return

    logger.info(f"文件夹 {folder_name} 包含 {n_job} 个文件，开始处理")

    removed_output_folder = os.path.join(output_folder, 'removed')

    if not format:
        format = get_format_from_path(glob_pattern)

    # 定义格式与阅读器类的映射关系
    reader_map = {
        'jsonl': JsonlReader,
        'parquet': ParquetReader,
        'warc': WarcReader,
        'csv': CSVReader  # 默认格式
    }

    # 获取对应的阅读器类（默认使用CSVReader）
    ReaderClass = reader_map.get(format.lower(), CSVReader)
    INPUT_READER = ReaderClass(
            input_folder,
            glob_pattern=glob_pattern,
            id_key=id_key
        )

    main_processing_executor = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            LanguageFilter(
                languages=['en'],
                exclusion_writer=JsonlWriter(
                    os.path.join(removed_output_folder, folder_name, '1_non_english'),
                    output_filename="${language}/${rank}.jsonl.gz",
                    # folder structure: language/file
                )
            ),
            GopherRepetitionFilter(
                exclusion_writer=JsonlWriter(os.path.join(removed_output_folder, folder_name, '2_GopherRepetition'))
            ),
            GopherQualityFilter(
                exclusion_writer=JsonlWriter(os.path.join(removed_output_folder, folder_name, '3_GopherQuality'))
            ),
            C4QualityFilter(
                filter_no_terminal_punct=False,
                exclusion_writer=JsonlWriter(os.path.join(removed_output_folder, folder_name, '4_C4Quality')),
            ),
            FineWebQualityFilter(
                exclusion_writer=JsonlWriter(os.path.join(removed_output_folder, folder_name, '5_FineWebQuality'))
            ),
            JsonlWriter(os.path.join(output_folder, 'filtering', folder_name)),
        ],
        tasks=n_job,
        workers=min(
            n_job,
            os.cpu_count()
        ),
        logging_dir=os.path.join(output_folder, 'logs/filtering', folder_name),
    )

    try:
        result = main_processing_executor.run()
        logger.info(f"文件夹 {folder_name} 处理完成: {result}")
    except Exception as e:
        logger.error(f"处理文件夹 {folder_name} 时出错: {str(e)}")

def minhash(input_folder, output_folder):
    folder_name = os.path.basename(input_folder.rstrip('/'))
    logger.info(f"minhash 开始处理文件夹: {folder_name}")
    # you can also change ngrams or the number of buckets and their size here
    minhash_config = MinhashConfig(
        hash_config=HashConfig(
            hash_fc="sha1",  # better precision -> fewer false positives (collisions)
            precision=64,
        ),
        num_buckets=14,
        hashes_per_bucket=8,
        n_grams=5,
    )

    files = glob(os.path.join(output_folder, 'filtering', folder_name, '*.jsonl.gz') , recursive= True)
    n_job = len(files)

    if n_job == 0:
        logger.warning(f"文件夹 {folder_name} 中未找到文件，跳过处理")
        return

    logger.info(f"文件夹 {folder_name} 包含 {n_job} 个文件，开始处理")

    logging_dir = os.path.join(output_folder, 'logs/minhash', folder_name)

    INPUT_READER = JsonlReader(os.path.join(output_folder, 'filtering', folder_name), glob_pattern="*.jsonl.gz")
    logger.info(f"开始处理 {input_folder}，包含 {n_job} 个任务")

    # 创建该文件夹专用的临时路径
    MINHASH_BASE_PATH = os.path.join(output_folder, "tmp", folder_name)
    os.makedirs(MINHASH_BASE_PATH, exist_ok=True)

    # Stage 1: 计算MinHash签名
    stage1 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(
                output_folder=f"{MINHASH_BASE_PATH}/signatures", config=minhash_config
            ),
        ],
        tasks=n_job,
        workers=min(
            n_job,
            os.cpu_count()
        ),
        logging_dir=f"{logging_dir}/signatures",
    )

    # Stage 2: 在每个桶中查找匹配项
    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{MINHASH_BASE_PATH}/signatures",
                output_folder=f"{MINHASH_BASE_PATH}/buckets",
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        workers=min(
            n_job,
            os.cpu_count()
        ),
        logging_dir=f"{logging_dir}/buckets",
        depends=stage1,
    )

    # Stage 3: 创建重复项集群
    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{MINHASH_BASE_PATH}/buckets",
                output_folder=f"{MINHASH_BASE_PATH}/remove_ids",
                config=minhash_config,
            ),
        ],
        tasks=1,
        workers=1,
        logging_dir=f"{logging_dir}/clusters",
        depends=stage2,
    )

    # Stage 4: 过滤掉重复项并输出去重后的数据
    stage4 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            MinhashDedupFilter(
                input_folder=f"{MINHASH_BASE_PATH}/remove_ids",
                exclusion_writer=JsonlWriter(os.path.join(output_folder, "removed", folder_name, '6_Minhash')),
            ),
            JsonlWriter(os.path.join(output_folder, 'minhash', folder_name)),
        ],
        tasks=n_job,
        workers=min(
            n_job,
            os.cpu_count()
        ),
        logging_dir=f"{logging_dir}/filtering",
        depends=stage3,
    )

    # 执行并打印结果
    result = stage4.run()
    logger.info(f"文件夹 {input_folder} 处理完成，结果: {result}")

def batch_process_folders(root_input_folder: str, output_root_folder: str, glob_pattern: str, format: str, id_key: str = 'id'):
    """批量处理根目录下的所有子文件夹"""
    # 获取根目录下的所有子文件夹
    subfolders = [f.path for f in os.scandir(root_input_folder) if f.is_dir()]

    if not subfolders:
        logger.warning(f"在 {root_input_folder} 中未找到任何子文件夹")
        filtering(input_folder = root_input_folder,
                  output_folder = output_root_folder,
                  glob_pattern = glob_pattern,
                  id_key= id_key,
                  format=format)
        minhash(root_input_folder, output_root_folder)
    else:

        logger.info(f"发现 {len(subfolders)} 个子文件夹，开始批量处理")

        # 依次处理每个子文件夹
        for i, subfolder in enumerate(subfolders, 1):
            logger.info(f"\n===== 处理第 {i}/{len(subfolders)} 个文件夹 =====")
            filtering(subfolder, output_root_folder, glob_pattern, id_key)
            minhash(subfolder, output_root_folder)

    logger.info("所有文件夹处理完毕")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="fineweb pipeline")
    parser.add_argument("input_dir", type=str, help="输入文件目录")
    parser.add_argument("output_dir", type=str, help="输出文件目录")
    parser.add_argument("--glob_pattern", type=str, default='**/*.jsonl.zst', help="glob查找文件的pattern")
    parser.add_argument("--id_key", type=str, default="warc_record_id", help="每条记录对应的id列")
    parser.add_argument("--format", type=str, default="jsonl", help="每条记录对应的id列")
    args = parser.parse_args()

    # 目录检查
    if not os.path.exists(args.input_dir):
        logger.warning(f"输入目录不存在：{args.input_dir}，退出")
        sys.exit(1)
    if not os.path.exists(args.output_dir):
        logger.warning(f"输出目录不存在：{args.output_dir}，自动创建")
        os.makedirs(args.output_dir, exist_ok=True)

    # 开始批量处理
    batch_process_folders(
        root_input_folder=args.input_dir,
        output_root_folder=args.output_dir,
        glob_pattern=args.glob_pattern,
        id_key=args.id_key,
        format=args.format
    )
