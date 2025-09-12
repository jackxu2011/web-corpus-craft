import sys
sys.path.insert(0, "/work/fastText/build/lib.linux-x86_64-cpython-310")

import os
import time
import json
import argparse
import fasttext
import pandas as pd
from tqdm import tqdm
from glob import glob
import logging
import threading
import queue

# 配置日志（实时控制台输出，格式简洁）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info(f"fasttext loaded from: {fasttext.__file__}")

QUEUE_MAX_SIZE = 100           # 队列最大容量（防止内存溢出）
READ_THREAD_NUM = 10         # 处理线程数量（可根据CPU核心数调整，提高并发效率）
PROCESS_THREAD_NUM = 4         # 处理线程数量（可根据CPU核心数调整，提高并发效率）

# -------------------------- 1. 读取线程：读取文件→解析key-value→存入read_queue ----------
def read_worker(files, read_queue: queue.Queue, worker_id: int):
    logger.info(f'thread_{worker_id} deal {len(files)} files')
    for file in tqdm(files, desc=f'thread_{worker_id} read files'):
        file_name = os.path.basename(file)
        df = load_texts_from_jsonl(file)
        if len(df) > 0:
            read_queue.put((file_name, df))
        else:
            logger.info(f"{file_name} 是空的，跳过")

# -------------------------- 2. 处理线程：从read_queue取数据→处理→存入result_queue --------------------------
def process_worker(read_queue: queue.Queue, result_queue: queue.Queue, model, worker_id: int):
    logger.info(f"【处理线程{worker_id}】启动，等待数据...")
    while True:
        # 从读取队列取数据（无数据时阻塞）
        signal, df = read_queue.get()

        # 若收到“结束信号”，退出循环（需标记队列任务完成）
        if signal == "end":
            logger.info(f"【处理线程{worker_id}】收到结束信号，准备退出")
            read_queue.task_done()  # 标记任务完成（供join()等待）
            result_queue.put(("end", None))
            break

        # 执行处理逻辑
        try:
            result = run_inference_mt(model, df)
            # 将处理结果存入结果队列（格式："处理后key=处理后value"，可自定义）
            result_queue.put((signal, result))
        except Exception as e:
            logger.error(f"【处理线程{worker_id}】处理失败{signal}：{str(e)}")

        # 标记当前任务完成（必须调用，否则read_queue.join()会永久阻塞）
        read_queue.task_done()

# -------------------------- 3. 保存线程：从result_queue取结果→写入输出文件 --------------------------
def save_worker(output_dir: str, result_queue: queue.Queue, process_thread_num: int):

    end_count = 0  # 计数结束信号，达到处理线程数时退出
    global_metrics = []
    while True:
        # 从结果队列取数据（无数据时阻塞）
        signal, data = result_queue.get()

        # 若收到“结束信号”，计数+1，达到阈值则退出
        if signal == "end":
            end_count += 1
            result_queue.task_done()
            if end_count == process_thread_num:
                logger.info(f"【保存线程】收到所有处理线程的结束信号，退出")
                break
            continue
        try:
            file_name = os.path.splitext(signal)[0]
            global_metrics.append(save_data(signal, os.path.join(output_dir, f'{file_name}.csv'), data))
        except Exception as e:
            logger.error(f"保存数据{file_name}失败：{str(e)}")
        result_queue.task_done()  # 标记任务完成

    # 初始化各指标的总和为0
    sum_total_samples = 0
    sum_true_samples = 0
    sum_total_time = 0

    # 遍历数组并累加各指标
    for metrics in global_metrics:
        sum_total_samples += metrics["total_samples"]
        sum_true_samples += metrics["true_samples"]
        sum_total_time += metrics["total_time"]

    avg_time = sum_total_time / sum_total_samples
    speed = sum_total_samples / sum_total_time
    logger.info("\n========== Global Report ==========")
    logger.info(f"Total samples:         {sum_total_samples}")
    logger.info(f"0.9_True samples:         {sum_true_samples}")
    logger.info(f"Total time (s):        {sum_total_time:.2f}")
    logger.info(f"Average per sample:    {avg_time:.4f} seconds")
    logger.info(f"Samples per second:    {speed:.2f}")
    logger.info("======================================")

def split_list_into_n(lst, n):
    """
    将列表均匀分割为n个子列表

    参数:
        lst: 要分割的原始列表
        n: 要分割的子列表数量

    返回:
        包含n个子列表的列表
    """
    # 计算列表长度
    total = len(lst)

    # 处理n大于列表长度的情况
    if n >= total:
        return [[item] for item in lst] + [[] for _ in range(n - total)]

    # 计算基本长度和余数
    base_size = total // n
    remainder = total % n

    result = []
    start = 0

    for i in range(n):
        # 前remainder个子列表多一个元素
        end = start + base_size + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end

    return result

# 加载文本
def load_texts_from_json(json_path, text_key="text", max_lines=None):
    texts = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                if text_key in data:
                    texts.append(data[text_key].replace('\n', ' '))
            except json.JSONDecodeError:
                continue
            if max_lines and len(texts) >= max_lines:
                break
    return texts

# 加载文本
def load_texts_from_jsonl(json_path, text_key="text", max_lines=None):
    df = pd.DataFrame()
    try:
        df = pd.read_json(json_path, compression='zstd', lines=True, on_bad_lines="skip")
        if text_key in df.columns:
            df = df[(df[text_key].str.len() < 100000) & (df[text_key].str.len() > 100)]
            df.loc[:, text_key] = df[text_key].apply(lambda x: x.replace('\n', ' '))
    except Exception as e:
        logger.error(f"读取文件 {file} 时发生错误: {str(e)}")
        with open('logs/failed_file.log', 'a', encoding='utf-8') as f:
            f.write(f'{json_path}\n')
    return df

# 使用 fasttext 内部多线程函数 predict_mt
def run_inference_mt(model, input_df, text_key='text'):
    # model.eval()
    logger.info(f"[INFO] Loaded {len(input_df)} samples for inference.")

    texts = input_df[text_key].tolist()
    logger.info(f"[INFO] Start inference...")
    start = time.time()

    labels, probs = model.predict_mt(texts, k=1)  # 不再传 num_threads
    result=[]
    for i, label in enumerate(labels):
        # if label[0] == '__label__1' and probs[i][0] > 0.9:
        #     result.append({
        #         'prob': probs[i][0],
        #         'text': texts[i]
        #     })
        result.append({
            'prob': probs[i][0],
            'label': label[0]
        })
    df = pd.DataFrame(result, columns=['prob','label'])
    concated_df = pd.concat([input_df, df], axis=1)
    concated_df = concated_df[(concated_df['label'] == '__label__1') & (concated_df['prob'] > 0.9)]
    end = time.time()

    return end - start, len(df), concated_df[['prob', 'url', text_key]]

def append_to_csv(file_path, new_data, index=False, **kwargs):
    """
    向CSV文件追加数据

    参数:
        file_path (str): CSV文件路径
        new_data (pd.DataFrame): 要追加的数据
        index (bool): 是否写入索引，默认为False
        **kwargs: 传递给to_csv的其他参数
    """
    try:
        # 检查文件是否存在
        file_exists = os.path.isfile(file_path)

        # 如果文件存在，不写入表头；否则写入表头
        new_data.to_csv(
            file_path,
            mode='a',
            header=not file_exists,
            index=index,** kwargs
        )
        logger.info(f"成功向CSV文件追加了 {len(new_data)} 行数据")
        return True
    except Exception as e:
        logger.error(f"追加到CSV时出错: {str(e)}")
        return False

def save_data(input_file, output_file, data):

    total_time, length, result_df = data
    append_to_csv(output_file, result_df, index=False)
    avg_time = total_time / length
    speed = length / total_time

    metrics = {
        "total_samples": length,
        "true_samples": len(result_df),
        "total_time": total_time
    }

    logger.info("\n========== Inference Report ==========")
    logger.info(f"file:         {input_file}")
    logger.info(f"Total samples:         {length}")
    logger.info(f"0.9_True samples:         {len(result_df)}")
    logger.info(f"Total time (s):        {total_time:.2f}")
    logger.info(f"Average per sample:    {avg_time:.4f} seconds")
    logger.info(f"Samples per second:    {speed:.2f}")
    logger.info("======================================")
    return metrics

def load_model(model_path, num_threads=20):

    logger.info(f"[INFO] Using FastText model: {model_path}")
    model = fasttext.load_model(args.model_path)
        # 优先尝试设置线程数
    if hasattr(model, "setNumThreads"):
        model.setNumThreads(num_threads)
        logger.info(f"[INFO] Set num_threads to {num_threads} via setNumThreads().")
    else:
        logger.info(f"[WARN] setNumThreads() not available, model will use default thread setting.")
    return model

def find_zstd_files(input_dir, recursive=True):
    if os.path.isfile(input_dir):
        return [input_dir]
    """查找指定目录下所有以.zstd结尾的文件"""
    pattern = os.path.join(input_dir, "**", "*.zstd")
    zstd_files = glob(pattern, recursive=recursive)
    return [os.path.abspath(file) for file in zstd_files]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("input_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="result/dclm")
    parser.add_argument("--max_lines", type=int, default=None)
    parser.add_argument("--threads", type=int, default=20, help="Number of threads for FastText (real multi-thread test)")
    args = parser.parse_args()

    # 输入文件夹不存在则退出
    if not os.path.exists(args.input_dir):
        logger.warning(f"文件夹 {args.input_dir} 不存在，退出...")
        sys.exit(1)

    # 文件夹不存在则创建
    if not os.path.exists(args.output_dir):
        logger.warning(f"文件夹 {args.output_dir} 不存在，自动创建...")
        os.makedirs(args.output_dir, exist_ok=True)

    # 1. 初始化线程安全队列（maxsize=QUEUE_MAX_SIZE，防止队列无限膨胀）
    read_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
    result_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)

    files = find_zstd_files(args.input_dir)

    files_parts = split_list_into_n(files, READ_THREAD_NUM)

    # 2. 创建并启动读取线程
    read_threads = []
    for worker_id in range(1, READ_THREAD_NUM + 1):
        read_thread = threading.Thread(
            target=read_worker,
            args=(files_parts[worker_id-1], read_queue, worker_id),
            daemon=True  # 守护线程：主线程退出时自动关闭（可选，此处建议设为True）
        )
        read_thread.start()
        read_threads.append(read_thread)

    # 3. 创建并启动多个处理线程（数量由PROCESS_THREAD_NUM控制）
    process_threads = []
    for worker_id in range(1, PROCESS_THREAD_NUM + 1):
        model = load_model(args.model_path, args.threads)
        t = threading.Thread(
            target=process_worker,
            args=(read_queue, result_queue, model, worker_id),
            daemon=True
        )
        t.start()
        process_threads.append(t)

    # 4. 创建并启动保存线程
    save_thread = threading.Thread(
        target=save_worker,
        args=(args.output_dir, result_queue, PROCESS_THREAD_NUM),
        daemon=True
    )
    save_thread.start()

    # 5. 等待读取线程完成（所有数据存入read_queue）
    for t in read_threads:
        t.join()
    print("【主线程】读取线程已完成")

    for _ in range(PROCESS_THREAD_NUM):
        read_queue.put(("end", None))  # ("end", ...) 表示处理线程可退出

    # 6. 等待read_queue中所有数据被处理线程消费完毕（需调用task_done()配合）
    read_queue.join()
    print("【主线程】所有数据已处理完成")

    # 7. 等待处理线程完成
    for t in process_threads:
        t.join()

    # 8. 等待result_queue中所有结果被保存线程消费完毕
    result_queue.join()
    print("【主线程】所有结果已保存完成")

    # 9. 等待保存线程退出
    save_thread.join()
    print("【主线程】所有线程已退出，程序结束")
