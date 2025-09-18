from zai import ZhipuAiClient  # !pip install zai-sdk
import os
import pandas as pd
import time
import threading
from queue import Queue
import logging
import signal
import sys
from collections import defaultdict
from glob import glob
import argparse
from dotenv import load_dotenv

load_dotenv()

# -------------------------- 基础配置与全局变量初始化 --------------------------
# 配置日志（实时控制台输出，格式简洁）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 线程安全队列（兼容历史逻辑）
result_queue = Queue()
# 全局状态：控制程序运行/终止
is_running = True
# 全局统计：存储所有文件有效评分（0-5分，-1分单独统计错误数）
global_score_stats = defaultdict(int)  # 键：0-5分，值：数量
global_error_count = 0  # 单独统计-1分（错误）的总数量
file_score_details = {}  # 键：文件名，值：{0-5分数量, 错误数}

API_KEY = os.getenv('ZHIPU_API_KEY')  # 替换为你的智谱API密钥
REQUEST_DELAY = 0.5  # 调用间隔（秒），MAX_THREADS*REQUEST_DELAY ≥1 可降低限流风险
PROMPT = """
[角色定位]
交通语料评估专家，判断输入文本是否适合作为交通领域fasttext分类器正样本。

[核心要求]
1. 与交通领域直接相关（交通方式/管理/基建/智能交通/政策服务/衍生问题）
2. 基于现实交通场景（含真实地域/机构/数据/政策，无虚构标识/元素/行为）
3. 排除"交通工具设计与制造"（如车身设计/零部件制造）

[智能交通明确]
自动驾驶(L1-L5级)、V2X、ITS、交通大数据均属智能交通

[评分规则]
5分：2+要素（含[地][数][策][机]任意2个）
4分：1个要素（含[地][数][策][机]任意1个）
3分：信息笼统（无具体地域/数据）
2分：交通关键词（仅含1-2个交通词无要素）
1分：虚构（含游戏/小说/影视内容）
0分：无任何交通相关内容及关键词或仅含歧义 “交通” 关键词（如 “交通部门”），易误导模型；

[输出格式]
仅输出0-5的阿拉伯数字,且只返回一个数字
    """

# -------------------------- 信号处理函数（中断安全退出） --------------------------
def signal_handler(sig, frame):
    global is_running
    logger.warning(f"\n接收到中断信号 {sig}（如Ctrl+C），正在保存当前结果...")
    is_running = False
    # 等待活跃线程结束（最多10秒）
    for thread in threading.enumerate():
        if thread != threading.current_thread() and thread.is_alive():
            thread.join(timeout=10)
    logger.warning("已保存结果，程序即将退出")
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# -------------------------- 1. 文本清理工具函数 --------------------------
def clean_text(text):
    """清理文本空值、换行符，统一格式"""
    if pd.isna(text):
        return ""
    cleaned = str(text).replace('\n', ' ').replace('\r', ' ')
    return ' '.join(cleaned.split())  # 合并连续空格


# -------------------------- 2. 模型评分函数（核心逻辑，错误返回-1） --------------------------
def chat(text, model_name):
    """调用GLM-4-Flash，正常返回0-5分，出错返回-1"""
    try:
        client = ZhipuAiClient(api_key=API_KEY)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": PROMPT
                },
                {
                    "role": "user",
                    "content": text  # 已清理的文本
                }
            ],
            temperature=0.1,
            timeout=30  # 超时保护
        )
        # 提取并验证结果（正常结果为0-5）
        res = clean_text(response.choices[0].message.content)
        if res.isdigit() and 0 <= int(res) <= 5:
            return int(res)
        else:
            logger.error(f"模型返回无效分数：{res}，待评分文本前50字：{text[:50]}...")
            return -1  # 无效结果视为错误，标记-1
    except Exception as e:
        logger.error(f"API调用失败：{str(e)}，待评分文本前50字：{text[:50]}...")
        return -1  # 调用异常标记-1


# -------------------------- 3. 线程工作函数（错误标记-1） --------------------------
def worker(queue, output_file, df, model_name, file_stats):
    """线程任务：评分+错误标记-1，更新DataFrame和统计"""
    global is_running
    while is_running and not queue.empty():
        index, text = queue.get()
        try:
            cleaned_text = clean_text(text)
            if not cleaned_text.strip():
                score = 0
                file_stats["score_counts"][score] += 1
                logger.info(f"索引 {index}：文本为空，评分=0")
            else:
                # 控制API调用频率
                time.sleep(REQUEST_DELAY)
                score = chat(cleaned_text, model_name)
                # 更新统计：区分正常分（0-5）和错误分（-1）
                if score == -1:
                    file_stats["error_count"] += 1
                    logger.error(f"索引 {index}：处理出错，评分=-1")
                else:
                    file_stats["score_counts"][score] += 1
                    logger.info(f"索引 {index}：评分完成，结果={score}")

            # 更新DataFrame评分列
            df.at[index, 'traffic_relevance_score'] = score

            # 每10条记录增量保存（避免数据丢失）
            if index % 20 == 0 and not df.empty:
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
                logger.info(f"临时保存：{os.path.basename(output_file)}（已处理至索引{index}）")

        except Exception as e:
            # 线程内其他异常，同样标记-1
            score = -1
            df.at[index, 'traffic_relevance_score'] = score
            file_stats["error_count"] += 1
            logger.error(f"索引 {index} 发生未知错误：{str(e)}，评分=-1")
        finally:
            queue.task_done()


# -------------------------- 4. 单个文件处理主函数（错误统计-1） --------------------------
def process_single_file(input_file, output_dir, model_name, max_threads):
    """处理单个CSV，返回该文件的正常评分分布（0-5）和错误数（-1）"""
    global is_running
    df = None
    # 初始化当前文件统计：score_counts存储0-5分数量，error_count存储-1分数量
    file_stats = {
        "score_counts": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        "error_count": 0
    }

    try:
        # 检查文件是否存在（排除统计文件）
        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, file_name)
        if "交通评分_" in file_name and file_name.endswith(".csv"):
            logger.info(f"跳过冗余统计文件：{file_name}")
            return file_stats

        if not os.path.exists(input_file):
            logger.error(f"文件不存在：{input_file}，跳过处理")
            return file_stats

        df = None
        score_col = 'traffic_relevance_score'

        if not os.path.exists(output_file):
          # 读取CSV文件
          df = pd.read_csv(input_file, encoding='utf-8-sig')
          # 初始化评分列（若不存在则新增）
          if score_col not in df.columns:
              df[score_col] = pd.NA
              df.to_csv(output_file, index=False, encoding='utf-8-sig')
              logger.info(f"新增评分列：{score_col}（已保存文件结构）")
        else:
          df = pd.read_csv(output_file, encoding='utf-8-sig')

        total_records = len(df)
        logger.info(f"\n=== 开始处理文件：{file_name}（总记录数：{total_records}）===")

        # 筛选待评分记录：空值/错误值（含旧的0分错误标记）/超范围值
        rows_to_score = df[
            df[score_col].isna() |
            (df[score_col] < 0) |
            (df[score_col] > 5) |
            (df[score_col].astype(str).isin(["", "翻译失败", "处理出错"]))
        ]
        total_to_score = len(rows_to_score)
        logger.info(f"待评分记录数：{total_to_score}（已完成：{total_records - total_to_score}）")

        # 无可评分记录时，统计已有有效数据
        if total_to_score == 0 or not is_running:
            logger.info(f"=== {file_name} 无待评分记录，跳过处理 ===")
            # 统计已有的正常分（0-5）和错误分（-1）
            if not df[score_col].isna().all():
                existing_scores = df[score_col].dropna().astype(int)
                for score in existing_scores:
                    if score == -1:
                        file_stats["error_count"] += 1
                    elif 0 <= score <= 5:
                        file_stats["score_counts"][score] += 1
            return file_stats

        # 填充任务队列（仅处理待评分记录）
        task_queue = Queue(maxsize=total_to_score)
        for index, row in rows_to_score.iterrows():
            task_queue.put((index, row.get('text', '')))  # 无text列传空字符串

        # 启动线程（守护线程，主程序退出时自动终止）
        thread_count = min(max_threads, total_to_score)
        threads = []
        for i in range(thread_count):
            thread = threading.Thread(
                target=worker,
                args=(task_queue, output_file, df, model_name, file_stats),
                daemon=True
            )
            thread.start()
            threads.append(thread)
            logger.info(f"启动线程 {i+1}/{thread_count}（守护线程）")
            time.sleep(0.2)

        # 等待任务完成（超时1小时）
        task_queue.join()
        # 回收线程（最多等待10秒）
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=10)
        logger.info(f"=== {file_name} 处理完成（线程已回收）===")

        # 补充统计已存在的有效数据（非待评分记录）
        if not df[score_col].isna().all():
            # 排除已处理的待评分记录，仅统计原有有效记录
            existing_mask = ~df.index.isin(rows_to_score.index)
            existing_scores = df.loc[existing_mask, score_col].dropna().astype(int)
            for score in existing_scores:
                if score == -1:
                    file_stats["error_count"] += 1
                elif 0 <= score <= 5:
                    file_stats["score_counts"][score] += 1

        # 校验统计完整性（总记录数=正常分+错误分）
        total_stat = sum(file_stats["score_counts"].values()) + file_stats["error_count"]
        if total_stat != total_records:
            missing = total_records - total_stat
            file_stats["error_count"] += missing  # 缺失记录视为错误，标记-1
            logger.warning(f"{file_name} 统计补全：缺失 {missing} 条记录，标记为-1（错误）")

        return file_stats

    except Exception as e:
        logger.error(f"处理 {file_name} 时发生严重错误：{str(e)}", exc_info=True)
        return file_stats
    finally:
        # 最终强制保存（无论成败，确保-1标记已写入）
        if df is not None and not df.empty:
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"最终保存：{file_name}（已强制保存所有结果，错误记录标记-1）")

# -------------------------- 5. 获取文件夹下有效CSV（排除统计文件） --------------------------
def get_all_csv_files(input_dir):
    """获取指定文件夹下有效CSV，排除“交通评分_”开头的统计文件"""
    if os.path.isfile(input_dir):
        return [input_dir]

    files = glob(f'{input_dir}/*.csv')

    # 筛选CSV：排除隐藏文件、统计文件
    csv_files = []
    for file in files:
        # 条件：是文件+后缀csv+非隐藏文件+非统计文件
        if (os.path.isfile(file) and
            not file.startswith('.') and
            not file.startswith('交通评分_')):
            csv_files.append(file)

    logger.info(f"在文件夹 {input_dir} 中找到 {len(csv_files)} 个有效CSV文件（已排除统计文件）")
    return csv_files

# -------------------------- 6. 批量处理+控制台统计（含错误-1） --------------------------
def batch_process_csv(input_dir, output_dir, model_name, max_threads):
    """批量处理CSV，控制台打印正常分（0-5）和错误分（-1）统计"""
    # 重置全局统计
    global global_score_stats, global_error_count, file_score_details
    global_score_stats.clear()
    global_error_count = 0
    file_score_details.clear()

    # 获取待处理CSV（排除统计文件）
    csv_files = get_all_csv_files(input_dir)
    if not csv_files:
        logger.warning("未找到任何有效CSV文件（已排除统计文件），程序退出")
        return

    # 逐个处理文件并累计统计
    for file in csv_files:
        if not is_running:
            logger.warning("程序已终止，停止处理剩余文件")
            break
        # 处理单个文件，获取统计
        file_stats = process_single_file(file, output_dir, model_name, max_threads)
        file_name = os.path.basename(file)
        file_score_details[file_name] = file_stats

        # 累加到全局统计
        for score, count in file_stats["score_counts"].items():
            global_score_stats[score] += count
        global_error_count += file_stats["error_count"]

    # -------------------------- 控制台打印最终统计 --------------------------
    logger.info("\n" + "="*80)
    logger.info("                          交通领域文本评分最终统计")
    logger.info("="*80)

    # 计算总记录数（正常分+错误分）
    total_global = sum(global_score_stats.values()) + global_error_count
    if total_global == 0:
        logger.info("无有效评分数据可统计（所有文件为空或处理失败）")
        return

    # 1. 全局统计（含错误-1）
    logger.info(f"\n【1. 全局统计】（总计：{total_global} 条记录）")
    logger.info("-"*80)
    # 打印正常分（0-5，降序）
    logger.info(f"{'评分等级':<12} {'数量':<10} {'占比(%)':<10} {'评分说明'}")
    logger.info("-"*80)
    # 按评分降序排列（5→0）
    sorted_scores = sorted(global_score_stats.keys(), reverse=True)
    for score in sorted_scores:
        count = global_score_stats[score]
        ratio = round(count/total_global*100, 2)
        # 评分说明映射
        score_desc = [
            "完全无关（无交通内容或仅含歧义 “交通” 关键词（如 “交通部门”））",
            "不相关（虚构（含游戏/小说/影视内容））",
            "极少相关（仅1-2个交通关键词，无具体内容）",
            "弱相关（交通为辅助信息，描述笼统）",
            "强相关（交通信息完整，含少量非核心内容）",
            "核心相关（聚焦单一类别，含数据/流程，无冗余）"
        ][score]
        logger.info(f"{score:<12} {count:<10} {ratio:<10} {score_desc}")
    # 单独打印错误分-1
    error_ratio = round(global_error_count/total_global*100, 2)
    # 单独打印错误分-1
    error_ratio = round(global_error_count / total_global * 100, 2)
    logger.info(f"{'-1（错误）':<12} {global_error_count:<10} {error_ratio:<10} 处理出错（API调用失败/无效结果/缺失记录）")
    logger.info("-" * 80)

    # 2. 单个文件统计（逐个展示，含错误-1）
    logger.info(f"\n【2. 单个文件统计】（共 {len(file_score_details)} 个文件）")
    for file_name, file_stats in file_score_details.items():
        # 计算单个文件总记录数（正常分+错误分）
        file_total = sum(file_stats["score_counts"].values()) + file_stats["error_count"]
        if file_total == 0:
            logger.info(f"\n→ {file_name}：无有效记录（文件为空或未处理）")
            continue

        # 打印文件基本信息
        logger.info(f"\n→ {file_name}（总记录数：{file_total}）")
        logger.info("-" * 70)
        logger.info(f"{'评分等级':<12} {'数量':<10} {'占比(%)':<10}")
        logger.info("-" * 70)

        # 打印正常分（0-5，降序）
        sorted_file_scores = sorted(file_stats["score_counts"].keys(), reverse=True)
        for score in sorted_file_scores:
            count = file_stats["score_counts"][score]
            ratio = round(count / file_total * 100, 2)
            logger.info(f"{score:<12} {count:<10} {ratio:<10}")

        # 打印错误分-1
        error_count = file_stats["error_count"]
        error_ratio_file = round(error_count / file_total * 100, 2)
        logger.info(f"{'-1（错误）':<12} {error_count:<10} {error_ratio_file:<10}")
        logger.info("-" * 70)

    # 统计结束提示
    logger.info("\n" + "=" * 80)
    logger.info("所有文件处理及评分统计完成！")
    logger.info(
        f"关键说明：1. 错误记录（评分=-1）可能内容敏感，或接口返回出错")


# -------------------------- 程序入口 --------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--model_name", type=str, default="glm-4-flash-250414")
    parser.add_argument("--threads", type=int, default=30, help="Number of threads to call llm")
    args = parser.parse_args()

    # 输入文件夹不存在退出
    if not os.path.exists(args.input_dir):
        logger.error(f"输入文件夹 {args.input_dir} 不存在，退出...")
        sys.exit(1)

    # 文件夹不存在则创建
    if not os.path.exists(args.output_dir):
        logger.warning(f"文件夹 {args.output_dir} 不存在，自动创建...")
        os.makedirs(args.output_dir, exist_ok=True)

    # 启动批量处理
    try:
        logger.info("=" * 80)
        logger.info("          交通领域文本相关性评分程序（错误标记-1 | 仅控制台统计）")
        logger.info("=" * 80)
        logger.info(f"配置参数：线程数={args.threads} | 调用间隔={REQUEST_DELAY}s | 目标文件夹={args.input_dir}")
        logger.info("操作提示：1. 按Ctrl+C可中断程序，中断前会自动保存已处理结果；2. 错误记录（-1）需优先检查API密钥")
        logger.info("-" * 80 + "\n")
        # 调用批量处理函数
        batch_process_csv(args.input_dir, args.output_dir, args.model_name, args.threads)
    except Exception as e:
        logger.error(f"程序启动失败：{str(e)}", exc_info=True)
        sys.exit(1)