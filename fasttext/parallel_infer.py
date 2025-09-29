#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "/mnt/data-token-cpfs/group-web/fastText/build/lib.linux-x86_64-cpython-310")

import os
import io
import json
import time
import gc
import math
import gzip
import fasttext
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from threading import Lock
import psutil
from typing import List
from loguru import logger

# ------------------ CONFIG ------------------
# 请按实际环境修改下面配置
MODEL_PATH = "/mnt/data-token-cpfs/group-web/group3/dataset/model/fasttext_medical_model_r3_1.bin"
DATA_ROOT = "/mnt/data-token-cpfs/group-web/cc_dump/shard_2"   # root 目录，递归查找 .gz 文件
OUTPUT_ROOT = "/mnt/si001117d1p1/default/group3/predict_process/v2_95/cc_dump_shard2"  # 输出放这里（会按相对路径创建）
PROCESSED_RECORD = os.path.join(OUTPUT_ROOT, "processed_files.txt")  # 全局已处理记录（主进程合并）
LOG_DIR = os.path.join(OUTPUT_ROOT, "log")  # 日志目录
LOG_PATH = os.path.join(LOG_DIR, "fasttext_predict.log")
WORKERS = 20             # 并行进程数（按机器 CPU / IO 能力调整）
INITIAL_BATCH_SIZE = 20000  # 基准批次大小（会根据内存自适应）
INFERENCE_THRESHOLD = 0.95  # 只写入 label=__label__1 且 score >= 阈值
WRITE_ONLY_POSITIVE = True  # True: 只写正样本；False: 写入全部预测
MEMORY_LIMIT_GB = 120       # 容器/机器的内存上限（用于自适应）
MEMORY_SAFETY_THRESHOLD = 0.80
MEMORY_CRITICAL_PERCENT = 0.90
FASTTEXT_NUM_THREADS = 0    # 传给 predict_mt 的 workers（0 = use default / auto）[不用predictmt了]
# --------------------------------------------

# Logging 配置
def setup_logging(log_file=None):
    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")
    if log_file:
        logger.add(log_file, rotation="500 MB", retention="30 days", level="INFO",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", enqueue=True)
    return logger

setup_logging(log_file=LOG_PATH)

# 线程锁：用于主进程更新全局进度文件（合并时使用）
file_lock = Lock()

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

def calculate_adaptive_batch_size(base_batch_size):
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

def find_input_files(root_dir: str, file_suffixes=("gz",)) -> List[Path]:
    """递归查找所有 *.gz 文件"""
    files = []
    root = Path(root_dir)
    for p in root.rglob("*"):
        if p.is_file() and any(str(p.name).endswith(f".{ext}") for ext in file_suffixes):
            files.append(p)
    return sorted(files)

def load_processed_files(path: str) -> set:
    """读取已处理文件集合（主进程）"""
    s = set()
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s.add(line.strip())
    return s

def merge_worker_processed_files(output_dir: str, worker_record_files: List[str], global_record_path: str):
    """将 worker 写的 processed list 合并到全局 processed_files.txt"""
    seen = load_processed_files(global_record_path)
    with open(global_record_path, "a", encoding="utf-8") as g:
        for rec in worker_record_files:
            try:
                with open(rec, "r", encoding="utf-8") as r:
                    for line in r:
                        fp = line.strip()
                        if fp and fp not in seen:
                            g.write(fp + "\n")
                            seen.add(fp)
            except Exception as e:
                logger.warning(f"合并 worker 记录文件 {rec} 时出错: {e}")
    # 删除 worker 的临时记录文件
    for rec in worker_record_files:
        try:
            os.remove(rec)
        except Exception:
            pass

# ---------- 子进程函数 ----------
def process_file_list_worker(args):
    """
    每个 worker 的入口函数：加载模型 -> 逐文件流式处理 -> 写入输出（按文件）
    args: dict 包含 keys:
        - model_path
        - file_list (List[str]) : 待处理文件路径列表
        - root_dirs, output_root, batch_size, threshold, write_only_positive, worker_id
    返回: (worker_id, processed_count, positive_count, worker_record_path)
    """
    worker_id = args["worker_id"]
    model_path = args["model_path"]
    file_list = args["file_list"]
    root_dir = args["root_dir"]
    output_root = args["output_root"]
    base_batch = args["batch_size"]
    threshold = args["threshold"]
    write_only_positive = args["write_only_positive"]
    num_threads = args.get("ft_threads", FASTTEXT_NUM_THREADS)

    processed_count = 0
    positive_count = 0
    # 将worker记录文件放到log目录下
    log_dir = os.path.dirname(args.get("log_path", LOG_PATH))
    worker_record_path = os.path.join(log_dir, f"processed_files_worker_{worker_id}.txt")

    # 使用全局logger
    logger.info(f"[worker-{worker_id}] 启动，加载模型: {model_path}")
    try:
        model = fasttext.load_model(model_path)
    except Exception as e:
        logger.error(f"[worker-{worker_id}] 加载模型失败: {e}")
        return worker_id, processed_count, positive_count, worker_record_path

    # 打开 worker 的 processed 记录文件（追加）
    try:
        rec_fh = open(worker_record_path, "a", encoding="utf-8")
    except Exception as e:
        logger.error(f"[worker-{worker_id}] 无法打开记录文件 {worker_record_path}: {e}")
        rec_fh = None

    # 逐文件处理
    for idx, file_path in enumerate(file_list):
        try:
            rel = os.path.relpath(file_path, root_dir)
            out_rel = rel.rsplit(".", 1)[0] + "_pred.jsonl"  # 去掉 .gz 后缀
            out_full = os.path.join(output_root, out_rel)
            os.makedirs(os.path.dirname(out_full), exist_ok=True)

            # 如果输出文件已存在，跳过（幂等）
            if os.path.exists(out_full):
                logger.info(f"[worker-{worker_id}] 输出已存在，跳过 {file_path}")
                if rec_fh:
                    rec_fh.write(str(file_path) + "\n")
                continue

            # 流式读取并批量预测
            batch_texts = []
            batch_lens = []
            batch_uids = []
            local_processed = 0
            local_positive = 0
            total_lines = 0  # 总行数统计
            saved_lines = 0  # 保存的行数统计

            with gzip.open(file_path, "rt", encoding="utf-8") as text_reader, open(out_full, "w", encoding="utf-8") as out_f:
                line_no = 0
                for line in text_reader:
                    line_no += 1
                    total_lines += 1  # 统计总行数
                    try:
                        # 解析JSON格式的TSV数据
                        data = json.loads(line.strip())
                        text = data.get("text", "")
                        warc_record_id = data.get("warc_record_id", "")
                    except Exception:
                        continue
                    if not text:
                        continue
                    text = text.replace("\n", " ").replace("\r", " ").strip()
                    if not text:
                        continue 

                    batch_texts.append(text)
                    batch_lens.append(len(text))
                    batch_uids.append(warc_record_id)

                    # 动态批次大小
                    adaptive_batch = calculate_adaptive_batch_size(base_batch)

                    if len(batch_texts) >= adaptive_batch:
                        try:
                            # labels, probs = model.predict_mt(batch_texts, k=1, workers=num_threads)
                            labels, probs = model.predict(batch_texts, k=1)
                        except Exception:
                            # 回退到逐条预测（非常慢但保证鲁棒）
                            labels = []
                            probs = []
                            for t in batch_texts:
                                try:
                                    p = model.predict(t, k=1)
                                    labels.append([p[0]])
                                    probs.append([p[1]])
                                except Exception:
                                    labels.append(["__label__0"])
                                    probs.append([0.0])

                        # 写入并过滤
                        for t, l, p, i, leng in zip(batch_texts, labels, probs, batch_uids, batch_lens):
                            lbl = l[0] if isinstance(l, (list, tuple)) and l else l
                            # 正确处理FastText返回的概率值（可能是numpy数组）
                            if isinstance(p, (list, tuple)) and p:
                                sc = float(p[0]) if hasattr(p[0], 'item') else float(p[0])
                            else:
                                sc = float(p.item()) if hasattr(p, 'item') else float(p)
                            processed_count += 1
                            local_processed += 1
                            write_flag = True
                            if write_only_positive:
                                if str(lbl) != "__label__1" or sc < threshold or sc > 1.0:
                                    write_flag = False
                            if write_flag:
                                out_obj = {"warc_record_id": i, "text": t, "label": lbl, "score": sc}
                                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                                saved_lines += 1  # 统计保存的行数
                                if str(lbl) == "__label__1":
                                    positive_count += 1
                                    local_positive += 1

                        # 清空 batch
                        batch_texts = []
                        batch_uids = []
                        batch_lens = []
                        gc.collect()

                # 处理剩余批次
                if batch_texts:
                    try:
                        # labels, probs = model.predict_mt(batch_texts, k=1, workers=num_threads)
                        labels, probs = model.predict(batch_texts, k=1)
                    except Exception:
                        labels = []
                        probs = []
                        for t in batch_texts:
                            try:
                                p = model.predict(t, k=1)
                                labels.append([p[0]])
                                probs.append([p[1]])
                            except Exception:
                                labels.append(["__label__0"])
                                probs.append([0.0])

                    for t, l, p, i, leng in zip(batch_texts, labels, probs, batch_uids, batch_lens):
                        lbl = l[0] if isinstance(l, (list, tuple)) and l else l
                        # 正确处理FastText返回的概率值（可能是numpy数组）
                        if isinstance(p, (list, tuple)) and p:
                            sc = float(p[0]) if hasattr(p[0], 'item') else float(p[0])
                        else:
                            sc = float(p.item()) if hasattr(p, 'item') else float(p)
                        processed_count += 1
                        local_processed += 1
                        write_flag = True
                        if write_only_positive:
                            if str(lbl) != "__label__1" or sc < threshold or sc > 1.0:
                                write_flag = False
                        if write_flag:
                            out_obj = {"warc_record_id": i, "text": t, "label": lbl, "score": sc}
                            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                            saved_lines += 1  # 统计保存的行数
                            if str(lbl) == "__label__1":
                                positive_count += 1
                                local_positive += 1
                    batch_texts = []
                    batch_uids = []
                    batch_lens = []
                    gc.collect()

            # 文件处理完成，写 worker 的 processed 记录
            if rec_fh:
                rec_fh.write(str(file_path) + "\n")
                rec_fh.flush()

            # 计算百分比
            process_rate = (local_processed / total_lines * 100) if total_lines > 0 else 0
            save_rate = (saved_lines / total_lines * 100) if total_lines > 0 else 0
            positive_rate = (local_positive / local_processed * 100) if local_processed > 0 else 0
            
            logger.info(f"[worker-{worker_id}] 完成文件 {os.path.basename(file_path)} -> {os.path.basename(out_full)} | "
                       f"总行数: {total_lines} | "
                       f"正样本数: {local_positive} ({positive_rate:.2f}%)")

        except Exception as e:
            logger.error(f"[worker-{worker_id}] 处理文件 {file_path} 出错: {e}")
            # 出错则跳过当前文件，继续下一个

    # 关闭记录文件
    if rec_fh:
        rec_fh.close()

    # 释放模型与 GC
    try:
        del model
    except Exception:
        pass
    gc.collect()
    logger.info(f"[worker-{worker_id}] 退出。总 processed={processed_count}, positive={positive_count}")
    return worker_id, processed_count, positive_count, worker_record_path

# ---------- 主流程 ----------
def chunkify(lst: List[str], n: int) -> List[List[str]]:
    """均匀把 lst 切成 n 份（尽量均匀）"""
    if n <= 0:
        return [lst]
    k, m = divmod(len(lst), n)
    chunks = []
    i = 0
    for j in range(n):
        size = k + (1 if j < m else 0)
        if size > 0:
            chunks.append(lst[i:i+size])
        else:
            chunks.append([])
        i += size
    return chunks

def main():
    start = time.time()
    logger.info("=== FastText 并行推理 ===")
    logger.info(f"MODEL_PATH={MODEL_PATH}")
    logger.info(f"DATA_ROOT={DATA_ROOT}")
    logger.info(f"OUTPUT_ROOT={OUTPUT_ROOT}")
    logger.info(f"WORKERS={WORKERS}  BATCH={INITIAL_BATCH_SIZE}  THRESHOLD={INFERENCE_THRESHOLD}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)  # 创建日志目录

    # 找到所有输入文件
    files = find_input_files(DATA_ROOT)
    logger.info(f"共发现 {len(files)} 输入文件（匹配 *.gz）")

    # 读取全局已处理记录，过滤
    processed_set = load_processed_files(PROCESSED_RECORD) if os.path.exists(PROCESSED_RECORD) else set()
    files_to_process = [str(p) for p in files if str(p) not in processed_set]
    logger.info(f"未处理文件: {len(files_to_process)}")

    if not files_to_process:
        logger.info("没有未处理的文件，退出")
        return

    # 将文件分块分配给 worker（每个 worker 一次拿一大块文件，减少调度开销）
    chunks = chunkify(files_to_process, WORKERS)
    worker_args = []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        worker_args.append({
            "worker_id": i,
            "model_path": MODEL_PATH,
            "file_list": chunk,
            "root_dir": DATA_ROOT,
            "output_root": OUTPUT_ROOT,
            "batch_size": INITIAL_BATCH_SIZE,
            "threshold": INFERENCE_THRESHOLD,
            "write_only_positive": WRITE_ONLY_POSITIVE,
            "ft_threads": FASTTEXT_NUM_THREADS,
            "log_path": LOG_PATH
        })

    # 并行提交
    worker_record_files = []
    total_processed = 0
    total_positive = 0
    completed_files = 0  # 已完成文件计数器

    with ProcessPoolExecutor(max_workers=WORKERS) as exe:
        futures = {exe.submit(process_file_list_worker, args): args["worker_id"] for args in worker_args}
        with tqdm(total=len(files_to_process), desc="总体文件进度", unit="文件") as pbar:
            for fut in as_completed(futures):
                wid = futures[fut]
                try:
                    worker_id, processed_count, positive_count, worker_record = fut.result()
                    total_processed += processed_count
                    total_positive += positive_count
                    if worker_record:
                        worker_record_files.append(worker_record)
                    # 更新已完成文件数：记录文件中行数为该 worker 完成的文件数，读取行数来更新 progress bar
                    # 更快的方式：读取 worker_record 并统计，但可能较慢 -> 这里 read lines
                    try:
                        cnt = 0
                        with open(worker_record, "r", encoding="utf-8") as fh:
                            for _ in fh:
                                cnt += 1
                        pbar.update(cnt)
                        completed_files += cnt
                    except Exception:
                        # 无法读取就用 processed_count 的估算（processed_count 是样本数，不是文件数），我们无法精确映射，所以只更新 1
                        pbar.update(1)
                        completed_files += 1
                    
                    # 每200个文件记录一次进度
                    if completed_files % 200 == 0:
                        logger.info(f"[进度] 已完成 {completed_files}/{len(files_to_process)} 个文件 ({completed_files/len(files_to_process)*100:.1f}%) | 累计处理样本: {total_processed} | 累计正样本: {total_positive}")
                    
                    logger.info(f"[主进程] worker-{worker_id} 完成 processed={processed_count}, positive={positive_count}")
                except Exception as e:
                    logger.error(f"[主进程] worker 异常: {e}")
                    pbar.update(1)

    # 合并 worker 的 processed 文件到全局记录
    if worker_record_files:
        merge_worker_processed_files(OUTPUT_ROOT, worker_record_files, PROCESSED_RECORD)

    elapsed = time.time() - start
    logger.info("=== 任务完成 ===")
    logger.info(f"总样本（近似） processed: {total_processed}")
    logger.info(f"正样本（近似） positive: {total_positive}")
    logger.info(f"总耗时: {elapsed:.2f} 秒，平均样本吞吐（近似）: { (total_processed/elapsed) if elapsed>0 else 0 :.2f} /s")

if __name__ == "__main__":
    main()
