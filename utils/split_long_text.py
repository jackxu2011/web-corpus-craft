import pandas as pd
import os
from tqdm import tqdm
import argparse
from loguru import logger
import re

def clean_english_text(text):
    """预处理英文文本：去除多余空格，统一格式（无换行符场景）"""
    # 去除连续空格（保留单个空格）
    text = re.sub(r"\s+", " ", text.strip())
    return text

def is_abbreviation_dot(word):
    """判断单词末尾的`.`是否为缩写（避免误判为句末标点）"""
    common_abbr = {
        # 称谓
        "Mr.", "Ms.", "Mrs.", "Dr.", "Prof.", "Rev.",
        # 机构/地名
        "U.S.A.", "U.K.", "E.U.", "U.N.", "D.C.",
        # 其他
        "e.g.", "i.e.", "etc.", "viz.", "vs.", "a.m.", "p.m.", "Fig.", "Eq."
    }
    # 匹配 A.B.C. 类缩写（如 U.S.A.）
    abbr_pattern = r"^[A-Z]\.[A-Z]\.([A-Z]\.)*$"
    return word in common_abbr or re.match(abbr_pattern, word)

def split_english_sentences(text):
    """将无换行符文本拆分为完整句子（核心步骤）"""
    sentences = []
    current_sent = []
    words = text.split()  # 按空格分割单词（无换行符，直接全部分词）

    for word in words:
        current_sent.append(word)
        # 检查是否为句末标点（.?!）且非缩写
        if word.endswith(('.', '?', '!')):
            word_without_punc = word.rstrip('.,?!')
            # 排除缩写和单个字母+点（如 A.）
            if not is_abbreviation_dot(word) and not (len(word_without_punc) == 1 and word.endswith('.')):
                sentences.append(" ".join(current_sent))
                current_sent = []

    # 补充最后一个句子
    if current_sent:
        sentences.append(" ".join(current_sent))
    return sentences

def count_english_words(text):
    """统计词数（连字符单词计为1词）"""
    return len(re.findall(r"\b[\w-]+\b", text.lower()))

def split_long_english_text_no_newlines(long_text, min_words=500, max_words=1000):
    """
    分割无换行符的英文长文本
    :param long_text: 无换行符的英文文本
    :param min_words: 最小词数
    :param max_words: 最大词数
    :return: 分割后的片段列表
    """
    # 1. 预处理
    clean_txt = clean_english_text(long_text)
    if not clean_txt:
        return []

    # 2. 先拆分为句子（无换行符，句子是最小语义单元）
    sentences = split_english_sentences(clean_txt)
    result = []
    current_segment = ""  # 当前片段

    for sent in sentences:
        sent_words = count_english_words(sent)
        current_words = count_english_words(current_segment)

        # 情况1：当前片段+句子 ≤ max_words → 加入句子
        if current_words + sent_words <= max_words:
            current_segment += sent + " "
        # 情况2：超过最大词数
        else:
            # 子情况A：当前片段已达标 → 保存并开始新片段
            if current_words >= min_words:
                result.append(current_segment.strip())
                current_segment = sent + " "
            # 子情况B：当前片段未达标 → 强制加入（避免过短）
            else:
                current_segment += sent + " "

    # 3. 处理最后一个片段
    if current_segment.strip():
        current_words = count_english_words(current_segment)
        if current_words < min_words and result:
            result[-1] += " " + current_segment.strip()  # 合并到前一个
        else:
            result.append(current_segment.strip())

    # 4. 最终校验：用逻辑连接词微调过长片段
    final_result = []
    for seg in result:
        seg_words = count_english_words(seg)
        if seg_words > max_words:
            # 按逻辑连接词分割（连接词前是断点）
            logic_conn = r"(?<=\.)\s+(however|therefore|thus|in contrast|for example|in addition|furthermore|on the other hand|in conclusion)"
            sub_segs = re.split(logic_conn, seg, flags=re.IGNORECASE)
            temp = ""
            for i in range(len(sub_segs)):
                part = sub_segs[i].strip()
                if not part:
                    continue
                # 连接词需与后续内容合并
                if part.lower() in ["however", "therefore", "thus", "in contrast", "for example", "in addition", "furthermore", "on the other hand", "in conclusion"]:
                    if i + 1 < len(sub_segs):
                        part += " " + sub_segs[i+1].strip()
                # 控制词数
                if count_english_words(temp + " " + part) <= max_words:
                    temp += " " + part
                else:
                    if temp.strip():
                        final_result.append(temp.strip())
                    temp = part
            if temp.strip():
                final_result.append(temp.strip())
        elif seg_words < min_words and final_result:
            final_result[-1] += " " + seg
        else:
            final_result.append(seg)

    return final_result

# ---------------------- 增强型CSV处理逻辑 ----------------------
def process_csv(input_path: str, output_path: str, text_column: str = "text",
                min_words: int = 500, max_words: int = 1000):
    try:
        # 1. 读取CSV并获取列信息
        df = pd.read_csv(input_path)
        total_rows = len(df)
        original_columns = df.columns.tolist()

        logger.info(f"成功读取CSV文件，共 {total_rows} 行，{len(original_columns)} 列")

        # 检查文本字段是否存在
        if text_column not in original_columns:
            logger.error(f"CSV文件中不存在 '{text_column}' 字段")
            return

        # 2. 处理每行文本
        results= []
        error_rows = []  # 记录有问题的行号

        for idx, row in df.iterrows():
            try:
                # 处理空值行
                if row.isna().all():
                    logger.warning(f"跳过全空行（行索引：{idx}）")
                    continue

                original_text = row.get(text_column, "")
                if pd.isna(original_text) or str(original_text).strip() == "":
                    logger.warning(f"跳过空文本（行索引：{idx}）")
                    continue

                # 分割文本
                segments = split_long_english_text_no_newlines(original_text, min_words, max_words)

                # 生成片段数据
                # for seg_idx, seg in enumerate(segments, 1):
                #     # 构建完整数据字典
                #     seg_data = {col: row.get(col, None) for col in original_columns}
                #     seg_data["segment_id"] = f"{idx}_{seg_idx}"
                #     seg_data["segment_text"] = seg
                #     seg_data["segment_word_count"] = count_english_words(seg)

                #     # 严格校验列数
                #     current_col_count = len(seg_data)
                #     if current_col_count != required_col_count:
                #         raise ValueError(f"列数不匹配：预期 {required_col_count} 列，实际 {current_col_count} 列")

                results += segments

            except Exception as e:
                # 记录错误但继续处理其他行
                error_rows.append(idx)
                logger.error(f"处理行 {idx} 时出错：{str(e)}，已跳过该行")
                continue

            # 打印进度
            if (idx + 1) % 100 == 0:
                logger.info(f"已处理 {idx + 1}/{total_rows} 行")

        # 3. 生成输出DataFrame
        if not results:
            logger.warning("没有生成任何分割结果")
            return

        # 显式指定列名，确保顺序一致
        output_df = pd.DataFrame(results, columns=[text_column])
        output_df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"\n处理完成！结果已保存到: {output_path}")
        logger.info(f"原始数据行数：{total_rows}")
        logger.info(f"生成的片段总数：{len(output_df)}")
        logger.info(f"输出CSV列数：{len(output_df.columns)}")
        if error_rows:
            logger.warning(f"注意：共有 {len(error_rows)} 行数据处理失败，行索引：{error_rows}")

    except FileNotFoundError:
        logger.error(f"输入文件 '{input_path}' 不存在")
    except Exception as e:
        logger.error(f"处理过程中发生致命错误：{str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--text_key", type=str, default='text')
    parser.add_argument("--min_words", type=int, default=500)
    parser.add_argument("--max_words", type=int, default=1000)
    args = parser.parse_args()
    # 输入文件夹不存在则退出
    if not os.path.exists(args.input_file):
        logger.error(f"文件夹 {args.input_dir} 不存在，退出...")
        sys.exit(1)

    process_csv(args.input_file, args.output_file, args.text_key, args.min_words, args.max_words)
