import pandas as pd
import os
from utils import data_util
from utils import pd_util
from utils.merge_train_data import merge_train_data
import csv
import glob
import zstandard as zst
from tqdm import tqdm
import tldextract
from loguru import logger

def load_data(input_file, output_file, field):
  text_key='text'
  df = None
  if (input_file.endswith('.csv')):
    df = pd.read_csv(input_file)
  else:
    df = pd.read_excel(input_file)

  df[text_key] = df[field].apply(data_util.clean_text)
  df = df.dropna(subset=[text_key])
  df = df[df[text_key] != ""]
  df[text_key].to_csv(output_file, index=False )

# load_data('group_data/source/Academic_Transport_WST202201.csv', 'group_data/seed/Academic_Transport_WST202201.csv','Abstract')

def add_label(in_file, label, out_file):
  df = pd.read_csv(in_file)
  df['text'] = df.text.apply(lambda x:f"__label__{label} {x}")
  with open(f'data/{out_file}', 'w', encoding='utf-8') as f:
    for text in df['text']:
      f.write(text + '\n')  # 每条文本一行

def duplicates(in_file, out_file):
  # 获取当前目录下所有 .csv 文件的路径
  df = pd.read_csv(f'data/{in_file}')

  # 去除重复行
  df_cleaned = data_util.drop_duplicates(df)

  # 保存处理后的结果到新的 CSV 文件
  df_cleaned.to_csv(f'data/{out_file}', index=False)  # index=False 表示不保存索引列

def remain_column(file, column):
  df = pd.read_csv(file)
  df[column].to_csv(file, index=False)

def read_fasttext():
    # 获取当前目录下所有 .csv 文件的路径
  file_paths = glob.glob("group_data/fasttest/test/*.txt")  # 可根据实际路径修改

  # 读取每个文件并存入列表
  for file in file_paths:
    dataframes = []
    file_name = os.path.splitext(os.path.basename(file))[0]
    with open(file, 'r', encoding='utf-8') as file:
      for line in file.readlines():
        line = line.strip()  # 去除首尾空白和换行符
        if not line:  # 跳过空行
            continue
        # 按第一个空格分割，分离标签和内容
        # split(' ', 1) 确保只分割一次，避免内容中的空格影响
        parts = line.split(' ', 1)

        if len(parts) == 2:
          label, content = parts
          # 验证标签格式（可选，根据需要保留）
          if label.startswith('__label__'):
              dataframes.append(content)
          else:
            # 处理不符合格式的行（可选）
            print(f"警告：无效标签格式 - {label}（行内容：{line[:30]}...）")
        else:
          # 处理无法分割的行（可选）
          print(f"警告：无法分割标签和内容 - {line[:30]}...")
      df = pd.DataFrame(dataframes, columns=['text'])
      df = pd_util.drop_duplicates(df)
      df.to_csv(f'group_data/seed/{file_name}.csv', index=False)

# read_fasttext()

def get_domain_conunt(input, output):
  # 获取当前目录下所有 .csv 文件的路径
  df = pd.read_csv(f'{input}')
  print(f'读取文件完成')

  df['domain'] = df.url.apply(data_util.extract_main_domain)

  group_url = df.groupby('domain').size().reset_index(name='count')

  print(f'共有url数据：{len(group_url)}')
  # 排序
  sorted_url = group_url.sort_values(by='count', ascending=False)
  print(f'保存数据')

  # 保存处理后的结果到新的 CSV 文件
  sorted_url.to_csv(f'{output}', index=False)  # index=False 表示不保存索引列
  print(sorted_url.head())

def convert_to_csv(input_file):
  df = pd.read_json(input_file, lines=True)
  df.to_csv('/work/group1/data/r6_result/dclm_0.9_50k.csv', index=False)

# convert_to_csv('/work/group1/datasets/v1/data_0001.jsonl')

def cal_url_prop(base_file, all_file, output_file):
  base_df = pd.read_csv(base_file)
  all_df = pd.read_csv(all_file)

  logger.info('begin to merge')
  # 合并两个DataFrame，只保留共同的key（内连接）
  merged = pd.merge(
    base_df.rename(columns={'count': 'df1_value'}),
    all_df.rename(columns={'count': 'df2_value'}),
    on='domain',
    how='inner'  # 只保留两个df都存在的key
  )

  logger.info('begin to cal')
  # 计算比值（处理除以0的情况）
  merged['ratio'] = merged['df1_value'] / merged['df2_value'].replace(0, pd.NA)

  logger.info('begin to sort')
  sorted_url = merged.sort_values(by='ratio', ascending=False)
  logger.info(sorted_url.head())

  sorted_url[['domain','df1_value','df2_value','ratio']].to_csv(output_file, index=False)

# cal_url_prop('result/domain/r6_domain_ge500.csv', 'result/domain/global.csv', 'result/domain/r6_domain_ge500_ratio.csv')

# def merge_dataset(base_dir):
#   metrics = []
#   for i in tqdm(range(10), desc='merge shard'):
#     file_name = f'shard_{i+1:02d}'
#     metrics.append(merge_csv(os.path.join(base_dir, file_name), os.path.join(base_dir,f'{file_name}.csv')))

#   global_total = 0
#   duplicated_total = 0
#   remain_total = 0
#   for index, metric in enumerate(metrics) :
#     global_total += metric['total']
#     duplicated_total += metric['duplicated']
#     logger.info(f'---- shard_{index+1:02d} -----')
#     logger.info(f'---- total:{metric['total']} -----')
#     logger.info(f'---- duplicated: {metric['duplicated']} -----')
#     logger.info(f'---- remain:{metric['remain']} -----')

#   logger.info(f'---- global -----')
#   logger.info(f'---- total:{global_total} -----')
#   logger.info(f'---- duplicated: {duplicated_total} -----')
#   logger.info(f'---- remain:{remain_total} -----')

# merge_dataset('/work/group1/data/r7_dclm')
#merge_csv('/work/group1/data/r7_dclm_r1/shard_05', '/work/group1/data/r7_dclm_r1/shard_05.csv')

def remove_df(input, need_remove, need_clean=False):
  in_df = pd.read_csv(input)
  logger.info(f'read: {input}')
  removed_df = pd.read_csv(need_remove)
  if need_clean:
    logger.info('clean text')
    removed_df['text'] = removed_df.text.apply(data_util.clean_text)
  logger.info(f'read: {need_remove}')
  mask = in_df["text"].isin(removed_df["text"])
  logger.info(f'save, removed {mask.sum()}')
  if mask.sum()>0:
    logger.info('save file')
    in_df[~mask].to_csv(input)
remove_df('data/fasttext/v2/pos/base/positive.csv', 'data/fasttext/v2/fst/neg_ge_3.csv')
# remove_df('data/fasttext/v2/neg/base/negative.csv', 'data/fasttext/v2/pos/base/positive.csv')
# remove_df('data/fasttext/v2/neg_llm//positive.csv', 'data/fasttext/v1/pos/long/r9.csv', True)
