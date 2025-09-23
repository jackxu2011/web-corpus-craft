import os
import time
from glob import glob
from loguru import logger

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import (
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    C4QualityFilter,
)

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

def process_single_folder(input_folder, glob_pattern, output_folder, logging_dir):
    """处理单个文件夹的函数"""
    # 获取当前处理的文件夹名称（用于输出路径）
    folder_name = os.path.basename(input_folder.rstrip('/'))
    logger.info(f"开始处理文件夹: {folder_name}")

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'remove'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'result'), exist_ok=True)

    # 检测输入文件夹中的JSONL文件数量
    jsonl_files = glob(os.path.join(input_folder,"**", glob_pattern) , recursive=True)
    n_job = len(jsonl_files)

    if n_job == 0:
        logger.warning(f"文件夹 {folder_name} 中未找到JSONL文件，跳过处理")
        return

    logger.info(f"文件夹 {folder_name} 包含 {n_job} 个JSONL文件，开始处理")

    INPUT_READER = JsonlReader(input_folder, glob_pattern="*.jsonl.gz", text_key="text")

    main_processing_executor = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            GopherRepetitionFilter(
                exclusion_writer=JsonlWriter(os.path.join(output_folder, 'remove', folder_name, '1_GopherRepetition'))
            ),
            GopherQualityFilter(
                exclusion_writer=JsonlWriter(os.path.join(output_folder, 'remove', folder_name,'2_GopherQuality'))
            ),
            C4QualityFilter(
                filter_no_terminal_punct=False,
                exclusion_writer=JsonlWriter(os.path.join(output_folder, 'remove', folder_name,'3_C4Quality')),
            ),
            FineWebQualityFilter(
                exclusion_writer=JsonlWriter(os.path.join(output_folder, 'remove', folder_name,'4_FineWebQuality'))
            ),
            JsonlWriter(os.path.join(output_folder, 'result', folder_name)),
        ],
        tasks=n_job,
        workers=min(
            n_job,
            os.cpu_count()
        ),
        logging_dir=logging_dir,
        skip_completed=False,
    )

    try:
        result = main_processing_executor.run()
        logger.info(f"文件夹 {folder_name} 处理完成: {result}")
    except Exception as e:
        logger.error(f"处理文件夹 {folder_name} 时出错: {str(e)}")


def batch_process_folders(root_input_folder, output_root_folder, logging_dir):
    """批量处理根目录下的所有子文件夹"""
    # 获取根目录下的所有子文件夹
    subfolders = [f.path for f in os.scandir(root_input_folder) if f.is_dir()]

    if not subfolders:
        logger.warning(f"在 {root_input_folder} 中未找到任何子文件夹")
        process_single_folder(root_input_folder, output_root_folder, logging_dir)
    else:

        logger.info(f"发现 {len(subfolders)} 个子文件夹，开始批量处理")

        # 依次处理每个子文件夹
        for i, subfolder in enumerate(subfolders, 1):
            logger.info(f"\n===== 处理第 {i}/{len(subfolders)} 个文件夹 =====")
            process_single_folder(subfolder, output_root_folder, logging_dir)

    logger.info("所有文件夹处理完毕")


if __name__ == '__main__':

    os.environ['INPUT_ROOT_DIR'] = "data/datatrove/lang/source"
    os.environ['OUTPUT_ROOT_DIR'] = "data/datatrove/quality"
    os.environ['LOGGING_ROOT_DIR'] = "data/datatrove/quality/logs"

    # 从环境变量获取参数
    root_input_folder = os.environ.get('INPUT_ROOT_DIR')
    output_root_folder = os.environ.get('OUTPUT_ROOT_DIR')
    logging_dir = os.environ.get('LOGGING_ROOT_DIR', './logs')

    # 检查必要的环境变量是否已设置
    required_vars = ['INPUT_ROOT_DIR', 'OUTPUT_ROOT_DIR']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]

    # 确保日志目录存在
    os.makedirs(logging_dir, exist_ok=True)

    # 开始批量处理
    batch_process_folders(
        root_input_folder=root_input_folder,
        output_root_folder=output_root_folder,
        logging_dir=logging_dir
    )
