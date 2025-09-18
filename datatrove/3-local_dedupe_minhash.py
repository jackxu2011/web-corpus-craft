import argparse
import json
import os
import time
from glob import glob
from loguru import logger

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    HashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter


def process_single_folder(input_folder, output_folder, folder_name, logging_dir):
    """处理单个文件夹的去重逻辑"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "removed"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "result"), exist_ok=True)

    # 配置MinHash参数
    minhash_config = MinhashConfig(
        hash_config=HashConfig(
            precision=64
        ),
        num_buckets=14,
        hashes_per_bucket=8,
    )

    # 计算输入文件夹中的任务数量
    n_job = len(glob(f"{input_folder}/*.jsonl.gz"))
    if n_job == 0:
        logger.warning(f"文件夹 {input_folder} 中未找到任何jsonl.gz文件，跳过处理")
        return

    INPUT_READER = JsonlReader(input_folder, glob_pattern="*.jsonl.gz", text_key="text")
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
        logging_dir=f"{logging_dir}/{folder_name}/signatures",
        skip_completed=False,
    )
    stage1.run()

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
        logging_dir=f"{logging_dir}/{folder_name}/buckets",
        skip_completed=False,
    )
    stage2.run()

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
        logging_dir=f"{logging_dir}/{folder_name}/clusters",
        skip_completed=False,
    )
    stage3.run()

    # Stage 4: 过滤掉重复项并输出去重后的数据
    stage4 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            MinhashDedupFilter(
                input_folder=f"{MINHASH_BASE_PATH}/remove_ids",
                exclusion_writer=JsonlWriter(os.path.join(output_folder, "removed", folder_name)),
            ),
            JsonlWriter(os.path.join(output_folder, 'result', folder_name)),
        ],
        tasks=n_job,
        workers=min(
            n_job,
            os.cpu_count()
        ),
        logging_dir=f"{logging_dir}/{folder_name}/filter",
        skip_completed=False,
    )

    # 执行并打印结果
    result = stage4.run()
    logger.info(f"文件夹 {input_folder} 处理完成，结果: {result}")



def parse_args():
    parser = argparse.ArgumentParser(description="批量处理文件夹的去重脚本")
    return parser.parse_args()

def batch_process_folders(root_input_folder, output_root_folder, logging_dir):
    """批量处理根目录下的所有子文件夹"""
    # 获取根目录下的所有子文件夹
    subfolders = [f.path for f in os.scandir(root_input_folder) if f.is_dir()]

    if not subfolders:
        logger.warning(f"在 {root_input_folder} 中未找到任何子文件夹")
        file_name = os.path.basename(root_input_folder.rstrip('/'))
        process_single_folder(root_input_folder, output_root_folder, file_name,logging_dir)
    else:
        logger.info(f"发现 {len(subfolders)} 个子文件夹，开始批量处理")
        # 依次处理每个子文件夹
        for i, subfolder in enumerate(subfolders, 1):
            logger.info(f"\n===== 处理第 {i}/{len(subfolders)} 个文件夹 =====")
            file_name = os.path.basename(subfolder.rstrip('/'))
            process_single_folder(subfolder, output_root_folder, file_name,logging_dir)

    logger.info("所有文件夹处理完毕")

if __name__ == '__main__':
    args = parse_args()

    os.environ['INPUT_ROOT_DIR'] = "data/datatrove/quality/result/source"
    os.environ['OUTPUT_ROOT_DIR'] = "data/datatrove/dedup"
    os.environ['LOGGING_ROOT_DIR'] = "data/datatrove/dedup/logs"

    # 从环境变量获取参数
    required_env_vars = [
        'INPUT_ROOT_DIR',     # 输入根目录（包含多个子文件夹）
        'OUTPUT_ROOT_DIR',    # 输出根目录
        'LOGGING_ROOT_DIR'    # 日志根目录
    ]

    # 检查必要的环境变量是否存在
    missing_vars = [var for var in required_env_vars if var not in os.environ]
    if missing_vars:
        raise EnvironmentError(f"缺少必要的环境变量: {', '.join(missing_vars)}")

    # 从环境变量读取参数
    input_root_dir = os.environ['INPUT_ROOT_DIR']
    output_root_dir = os.environ['OUTPUT_ROOT_DIR']
    logging_root_dir = os.environ['LOGGING_ROOT_DIR']


    # 确保日志目录存在
    os.makedirs(logging_root_dir, exist_ok=True)

    # 开始批量处理
    batch_process_folders(
        root_input_folder=input_root_dir,
        output_root_folder=output_root_dir,
        logging_dir=logging_root_dir
    )
