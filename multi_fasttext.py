import sys
sys.path.insert(0, "/mnt/data-token-cpfs/group-web/fastText/build/lib.linux-x86_64-cpython-310")

import os
import time
import json
import argparse
import fasttext
import pandas as pd
from tqdm import tqdm
from glob import glob
import logging

# 配置日志（实时控制台输出，格式简洁）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info(f"fasttext loaded from: {fasttext.__file__}")

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
        df = pd.read_json(json_path, compression='zstd', lines=True)
        if text_key in df.columns:
            df[text_key] = df[text_key].apply(lambda x: x.replace('\n', ' '))
    except Exception as e:
        logger.error(f"读取文件 {file} 时发生错误: {str(e)}")
        with open('logs/failed_file.log', 'a', encoding='utf-8') as f:
            f.write(f'{json_path}\n')
    return df

# 使用 fasttext 内部多线程函数 predict_mt
def run_inference_mt(model, file, text_key='text'):
    # model.eval()
    texts_df = load_texts_from_jsonl(file, text_key)
    logger.info(f"[INFO] Loaded {len(texts_df)} samples for inference.")

    if len(texts_df) <= 0:
        return 0,0,texts_df

    texts = texts_df[text_key].tolist()
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
    concated_df = pd.concat([texts_df, df], axis=1)
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

def main(model, input_json, output_dir='data/result', num_threads=20):

    logger.info(f"[INFO] Loading texts from: {input_json}")

    file_name = os.path.splitext(os.path.basename(input_json))[0]

    total_time, length, result_df = run_inference_mt(model, input_json)
    if length > 0:
        append_to_csv(f'{output_dir}/{file_name}.csv', result_df, index=False)
        avg_time = total_time / length
        speed = length / total_time

        metrics = {
            "total_samples": length,
            "true_samples": len(result_df),
            "total_time": total_time
        }

        logger.info("\n========== Inference Report ==========")
        logger.info(f"file:         {file_name}")
        logger.info(f"Total samples:         {length}")
        logger.info(f"0.9_True samples:         {len(result_df)}")
        logger.info(f"Total time (s):        {total_time:.2f}")
        logger.info(f"Average per sample:    {avg_time:.4f} seconds")
        logger.info(f"Samples per second:    {speed:.2f}")
        logger.info(f"Threads used:          {num_threads}")
        logger.info("======================================")
    else:
        metrics = {
            "total_samples": 0,
            "true_samples": 0,
            "total_time": 0
        }
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
    model = load_model(args.model_path, args.threads)
    files = find_zstd_files(args.input_dir)

    # 文件夹不存在则创建
    if not os.path.exists(args.output_dir):
        logger.warning(f"文件夹 {args.output_dir} 不存在，自动创建...")
        os.makedirs(args.output_dir, exist_ok=True)

    global_metrics = []
    for file in tqdm(files, desc='处理dclm文件'):
        global_metrics.append(main(model, file, args.output_dir))

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
