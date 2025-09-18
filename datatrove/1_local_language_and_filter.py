import os
import time
import json
from glob import glob
import sys
from typing import List

from loguru import logger
import zstandard #pip install zstandard
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import LanguageFilter, LambdaFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter


def get_env_var(var_name: str, required: bool = True, default=None):
    """获取环境变量，支持设置必填项和默认值"""
    value = os.getenv(var_name)
    if required and value is None:
        logger.error(f"环境变量 {var_name} 未设置，请先配置")
        sys.exit(1)
    return value if value is not None else default


def process(input_folder: str, output_folder: str, file_name: str, logging_dir: str) -> None:
    """处理JSONL文件，筛选英文文本并写入新文件"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)  # 确保日志目录存在

    # 获取输入文件夹中的jsonl文件数量
    jsonl_files = glob(os.path.join(input_folder, "**", "*.jsonl.zst"), recursive=True)# 启用递归模式
    n_job = len(jsonl_files)

    if n_job == 0:
        logger.warning(f"在 {input_folder} 中未找到任何*.jsonl.zst文件，跳过处理")
        return

    logger.info(f"处理 {input_folder}，包含 {n_job} 个文件")

    # 构建数据处理管道
    pipeline = [
        JsonlReader(
            input_folder,  # 确保 input_folder 是正确的父目录
            glob_pattern="**/*.jsonl.zst",  # 关键：用** 递归子文件夹匹配文件
            text_key="text",
            id_key="warc_record_id",
            compression="zstd",
            recursive=True  # 强制开启递归
        ),
        # 语言过滤：只保留英文
        LanguageFilter(backend="ft176", label_only=True),
        LambdaFilter(filter_function=lambda doc: doc.metadata["language"] in ['en']),
        JsonlWriter(
            os.path.join(output_folder, file_name),
        ),
    ]

    # 执行管道
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=n_job,
        # 根据CPU核心数自动调整工作进程数，避免资源耗尽
        workers=min(
            n_job,
            os.cpu_count()
        ),
        logging_dir=logging_dir,
        skip_completed=False,
    )

    logger.info(f"处理结果: {executor.run()}")

def batch_process_folders(root_input_folder, output_root_folder, logging_dir):
    """批量处理根目录下的所有子文件夹"""
    # 获取根目录下的所有子文件夹
    subfolders = [f.path for f in os.scandir(root_input_folder) if f.is_dir()]
    logger.info(subfolders)
    if not subfolders:
        logger.warning(f"在 {root_input_folder} 中未找到任何子文件夹")
        file_name = os.path.basename(root_input_folder.rstrip('/'))
        process(root_input_folder, output_root_folder, file_name, logging_dir)
    else:
        logger.info(f"发现 {len(subfolders)} 个子文件夹，开始批量处理")
        # 依次处理每个子文件夹
        for i, subfolder in enumerate(subfolders, 1):
            logger.info(f"\n===== 处理第 {i}/{len(subfolders)} 个文件夹 =====")
            file_name = os.path.basename(subfolder.rstrip('/'))
            process(subfolder, output_root_folder, file_name, logging_dir)

    logger.info("所有文件夹处理完毕")

if __name__ == '__main__':
    os.environ['INPUT_ROOT_DIR'] = "data/datatrove/source"
    os.environ['OUTPUT_ROOT_DIR'] = "data/datatrove/lang"
    os.environ['LOGGING_ROOT_DIR'] = "data/datatrove/lang/logs"

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
