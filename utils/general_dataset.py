import pandas as pd
import os
import glob

def split_continuous_csv(source_dir, output_dir, target_rows=100000, file_suffix='.csv'):
    """
    跨多个CSV文件连续累积数据，直到达到目标行数才保存为新文件

    参数:
        source_dir: 源CSV文件所在目录
        output_dir: 输出文件目录
        target_rows: 每个输出文件的目标行数，默认为100000
        file_suffix: 源文件后缀，默认为'.csv'
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有源文件
    source_files = glob.glob(os.path.join(source_dir, f'*{file_suffix}'))
    if not source_files:
        print(f"未在 {source_dir} 找到任何{file_suffix}文件")
        return

    # 读取表头（假设所有文件表头一致）
    header = pd.read_csv(source_files[0], nrows=1).columns.tolist()

    # 初始化变量
    current_data = []  # 存储累积的数据
    current_count = 0  # 当前累积的行数
    output_file_num = 1  # 输出文件编号

    # 遍历所有源文件
    for file_path in source_files:
        file_name = os.path.basename(file_path)
        print(f"开始处理文件: {file_name}")

        # 分块读取当前文件，避免内存溢出
        for chunk in pd.read_csv(file_path, chunksize=10000):
            chunk_rows = len(chunk)

            # 如果当前累积 + 本块行数 < 目标行数，直接累积
            if current_count + chunk_rows < target_rows:
                current_data.append(chunk)
                current_count += chunk_rows
                print(f"  累积行数: {current_count}/{target_rows}")

            # 刚好满足或超过目标行数
            else:
                # 计算还需要多少行达到目标
                need_rows = target_rows - current_count

                # 拆分当前块，取需要的部分
                part1 = chunk.iloc[:need_rows]
                part2 = chunk.iloc[need_rows:]

                # 完成当前文件
                current_data.append(part1)
                current_count += need_rows

                # 保存文件
                output_path = os.path.join(output_dir, f"output_{output_file_num:04d}.csv")
                pd.concat(current_data).to_csv(output_path, index=False, columns=header)
                print(f"生成文件: {os.path.basename(output_path)}，行数: {current_count}")

                # 重置计数器，处理剩余部分
                current_data = [part2]
                current_count = len(part2)
                output_file_num += 1
                print(f"  剩余行数转入下一轮: {current_count}")

        print(f"文件 {file_name} 处理完毕，当前累积行数: {current_count}\n")

    # 处理最后剩余的不足目标行数的数据
    if current_data and current_count > 0:
        output_path = os.path.join(output_dir, f"output_{output_file_num:04d}.csv")
        pd.concat(current_data).to_csv(output_path, index=False, columns=header)
        print(f"生成最后一个文件: {os.path.basename(output_path)}，行数: {current_count}")

    print("所有文件处理完成")

# 使用示例
if __name__ == "__main__":
    # 源文件目录和输出目录
    source_directory = "./source_csv"
    output_directory = "./output_continuous"

    # 调用函数，每100000行一个文件
    split_continuous_csv(
        source_dir=source_directory,
        output_dir=output_directory,
        target_rows=100000
    )
