import sys
import os
import time
import json
import argparse
import fasttext
import random
import pandas as pd
from tqdm import tqdm
from glob import glob
from loguru import logger
from concurrent.futures import ProcessPoolExecutor

# 全局变量，用于子进程中存储模型（每个进程一个实例）
global_model = None

WORKERS = 20             # 并行进程数（按机器 CPU / IO 能力调整）
INITIAL_BATCH_SIZE = 10000  # 基准批次大小（会根据内存自适应）
INFERENCE_THRESHOLD = 0.90  # 只写入 label=__label__1 且 score >= 阈值
WRITE_ONLY_POSITIVE = True  # True: 只写正样本；False: 写入全部预测
MEMORY_LIMIT_GB = 98       # 容器/机器的内存上限（用于自适应）
MEMORY_SAFETY_THRESHOLD = 0.80
MEMORY_CRITICAL_PERCENT = 0.90

# ---------- 辅助函数 ----------
def get_container_memory_usage():
    """尝试读取 cgroup 内存，否则回退到 psutil"""
    try:
        with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
            current_usage = int(f.read().strip())
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
            memory_limit = int(f.read().strip())
        return current_usage, memory_limit
    except Exception:
        proc = psutil.Process()
        return proc.memory_info().rss, MEMORY_LIMIT_GB * 1024**3

def calculate_adaptive_batch_size(base_batch_size=INITIAL_BATCH_SIZE):
    """根据当前内存使用百分比自适应批次大小"""
    try:
        current_usage, memory_limit = get_container_memory_usage()
        usage_percent = (current_usage / memory_limit) if memory_limit > 0 else 0
        # 映射策略
        if usage_percent > MEMORY_CRITICAL_PERCENT:
            return max(1000, base_batch_size // 8)
        elif usage_percent > MEMORY_SAFETY_THRESHOLD:
            return max(5000, base_batch_size // 4)
        elif usage_percent > 0.70:
            return max(10000, base_batch_size // 2)
        else:
            return base_batch_size
    except Exception:
        return base_batch_size

# 初始化子进程，加载模型（每个进程只加载一次）
def init_worker(model_path):
    global global_model
    # 获取当前进程的 PID 和父进程 PID
    pid = os.getpid()
    ppid = os.getppid()
    try:
        logger.info(f"[PID:{pid} | PPID:{ppid}] 子进程初始化，加载模型: {model_path}")
        global_model = fasttext.load_model(model_path)
    except Exception as e:
        logger.error(f"[PID:{pid}]子进程加载模型失败: {str(e)}")
        raise

# 加载文本（修正原代码中的变量名错误）
def load_texts_from_jsonl(file_path, text_key="text"):
    compression = 'zstd' if file_path.endswith('zstd') else 'infer'
    try:
        chunksize = calculate_adaptive_batch_size()
        # 读取压缩的 jsonl 文件
        for chunk in pd.read_json(file_path, lines=True, chunksize=chunksize, compression=compression):
            try:
                if text_key in chunk.columns:
                    # 替换换行符
                    chunk.loc[:, text_key] = chunk[text_key].apply(lambda x: x.replace('\n', ' '))
                    chunk = chunk[(chunk[text_key].str.len()>20)|(chunk[text_key].str.len()<100000)]
                    yield chunk
                else:
                    logger.info(f"{file_path} 没有 {text_key} 列， 跳过")
                    return
            except Exception as e:
                logger.error(f"处理 {file_path}, chunk 时发生错误: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"读取文件 {file_path} 时发生错误: {str(e)}")
        # 记录失败文件
        with open('logs/failed_file.log', 'a', encoding='utf-8') as f:
            f.write(f'{file_path}\n')
    return


# 使用 fasttext 多线程预测
def run_inference(model, texts_df, text_key='text', ext_column=['url']):
    texts = texts_df[text_key].tolist()
    logger.info(f"开始推理，样本数: {len(texts)}")
    start = time.time()

    # 调用多线程预测接口
    labels, probs = model.predict(texts, k=1)
    result_list = []
    for i, label in enumerate(labels):
        if label[0] == '__label__1' and probs[i][0] > INFERENCE_THRESHOLD:
            result = {
                'prob': probs[i][0],
                text_key: texts[i]
            }
            for col in ext_column:
              result[col] = texts_df.iloc[i][col]
            result_list.append(result)
    # 合并结果并过滤
    df = pd.DataFrame(result_list, columns=['prob', text_key] + ext_column)
    end = time.time()
    return end - start, df

# 追加数据到 CSV
def append_to_csv(file_path, new_data, index=False, **kwargs):
    try:
        file_exists = os.path.exists(file_path)
        new_data.to_csv(
            file_path,
            mode='a',
            header=not file_exists,
            index=index,** kwargs
        )
        logger.info(f"成功追加 {len(new_data)} 行到 {file_path}")
        return True
    except Exception as e:
        logger.error(f"追加到 CSV 失败: {str(e)}")
        return False


# 单个文件处理逻辑（供子进程调用）
def process_file(args_tuple):
    file_path, input_dir, output_dir, text_key, ext_column = args_tuple
    global global_model  # 使用子进程初始化的模型
    # 获取当前进程的 PID
    pid = os.getpid()
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    file_dir = os.path.dirname(file_path)

    logger.info(f"[PID:{pid}] 开始处理文件: {file_name}（路径: {file_path}）")

    output_file_dir = file_dir.replace(input_dir, output_dir, 1)

    if not os.path.exists(output_file_dir):
        logger.warning(f"输出目录 {output_file_dir} 不存在，自动创建")
        os.makedirs(output_file_dir, exist_ok=True)

    output_path = f'{output_file_dir}/{file_name}.csv'

    metrics = {
        "total_samples": 0,
        "true_samples": 0,
        "total_time": 0,
        "pid": pid
    }

    if os.path.isfile(output_path):
        logger.info(f"[PID:{pid}] 文件: {file_name}（路径: {file_path} 已处理, 跳过")
        return metrics

    for texts_df in load_texts_from_jsonl(file_path, text_key=text_key):
        length = len(texts_df)
        if length > 0:
            # 使用全局模型进行推理
            total_time, result_df = run_inference(global_model, texts_df, text_key=text_key, ext_column=ext_column)
            # 保存结果
            if len(result_df) > 0:
                append_to_csv(output_path, result_df, index=False)

            # 计算指标
            avg_time = total_time / length if length > 0 else 0
            speed = length / total_time if total_time > 0 else 0
            metrics["total_samples"] = metrics["total_samples"] + length
            metrics["true_samples"] = metrics["true_samples"] + len(result_df)
            metrics["total_time"] =  metrics["total_time"] + total_time

    logger.info(f"[PID:{pid}] \n========== 处理报告 [{file_name}] ==========")
    logger.info(f"[PID:{pid}] 总样本数: {metrics['total_samples']}")
    logger.info(f"[PID:{pid}] 符合条件样本数: {metrics['true_samples']}")
    logger.info(f"[PID:{pid}] 总耗时: {metrics['total_time']:.2f}s")
    logger.info("==========================================")
    return metrics

# 查找所有 zstd 文件
def find_all_files(input_dir, file_ext="gz", recursive=True):
    if os.path.isfile(input_dir):
        return [input_dir] if input_dir.endswith(file_ext) else []
    pattern = os.path.join(input_dir, "**", f"*.{file_ext}")
    zstd_files = glob(pattern, recursive=recursive)
    return [os.path.abspath(file) for file in zstd_files]

if __name__ == "__main__":

    # 主进程 PID（可选打印，用于区分主/子进程）
    main_pid = os.getpid()
    logger.info(f"主进程启动，PID: {main_pid}")

    # 解决 Windows 多进程问题
    if os.name == 'nt':
        import multiprocessing
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="fasttext 模型路径")
    parser.add_argument("input_dir", type=str, help="输入目录或文件")
    parser.add_argument("--file_ext", type=str, default="gz", help="文件后缀名")
    parser.add_argument("--output_dir", type=str, default="result/dclm", help="输出目录")
    parser.add_argument("--processes", type=int, default=WORKERS, help="进程数（默认 CPU 核心数）")
    parser.add_argument("--text_key", type=str, default="text", help="jsonl 中文本字段的键名")
    parser.add_argument("--ext_column", type=str, default=['url'], nargs='*', help="需要保留的除了text_key外的其它信息")

    args = parser.parse_args()

    # 创建输出目录
    if not os.path.exists(args.output_dir):
        logger.warning(f"输出目录 {args.output_dir} 不存在，自动创建")
        os.makedirs(args.output_dir, exist_ok=True)

    # 查找所有待处理文件
    files = find_all_files(args.input_dir, args.file_ext)
    if not files:
        logger.error(f"未找到任何 .{args.file_ext} 文件，退出")
        sys.exit(1)
    logger.info(f"共找到 {len(files)} 个 .{args.file_ext} 文件待处理")

    random.shuffle(files)

    # 准备进程池参数（每个文件的处理参数）
    process_args = [
        (file, os.path.abspath(args.input_dir), args.output_dir, args.text_key, args.ext_column)
        for file in files
    ]

    logger.info(f"共启动 {args.processes} 个进程处理文件")
    # 多进程处理
    with ProcessPoolExecutor(
        max_workers=args.processes,
        initializer=init_worker,  # 子进程初始化函数
        initargs=(args.model_path, )  # 初始化参数（模型路径）
    ) as executor:
        # 并行处理所有文件，用 tqdm 显示进度
        global_metrics = list(tqdm(
            executor.map(process_file, process_args),
            total=len(process_args),
            desc="处理文件进度"
        ))

    # 汇总全局指标
    sum_total = sum(m["total_samples"] for m in global_metrics)
    sum_true = sum(m["true_samples"] for m in global_metrics)
    sum_time = sum(m["total_time"] for m in global_metrics)

    avg_time_per_sample = sum_time / sum_total if sum_total > 0 else 0
    speed_total = sum_total / sum_time if sum_time > 0 else 0

    logger.info("\n========== 全局汇总报告 ==========")
    logger.info(f"总处理样本数: {sum_total}")
    logger.info(f"总符合条件样本数: {sum_true}")
    logger.info(f"总耗时 (s): {sum_time:.2f}")
    logger.info(f"全局单样本平均耗时 (s): {avg_time_per_sample:.4f}")
    logger.info(f"全局处理速度 (样本/秒): {speed_total:.2f}")
    logger.info(f"使用进程数: {args.processes}")
    logger.info("==================================")
