import pandas as pd
import os
from utils import data_util
from utils.merge_train_data import merge_train_data
from utils.merge_csv import merge_csv
import csv
import glob
import zstandard as zst
from tqdm import tqdm
import tldextract
from loguru import logger

base_root = '/work'
base_home = '/work/group1/jack'

def load_data(file, field):
  file_name_with_suffx = os.path.basename(file)
  file_name = file_name_with_suffx.split('.')[0]
  df = None
  if (file_name_with_suffx.endswith('.csv')):
    df = pd.read_csv(file)
  else:
    df = pd.read_excel(file)

  df = df[(df['text'] != "")]
  df['text'] = df[field].apply(data_util.clean_text)
  df = df.dropna(subset=['text'])
  df['text'].to_csv(f'data/{file_name}.csv', index=False )

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
  file_paths = glob.glob("data/raw/*.txt")  # 可根据实际路径修改

  # 读取每个文件并存入列表
  dataframes = []
  for file in file_paths:
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
  df.to_csv('data/clean/paper.csv', index=False)

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

def get_domain_sum(input, output, recursive=False):
  # 获取当前目录下所有 .csv 文件的路径
  logger.info('get file list')
  files = data_util.get_all_files(input, recursive=recursive)
  dfs = []
  for index, file in tqdm(enumerate(files), total=len(files), desc='deal files'):
    df = pd.read_csv(file)
    dfs.append(df)
    if (index+1)%1000 == 0:
      cat_df = pd.concat(dfs, ignore_index=True)
      df_sum = cat_df.groupby('domain')['count'].sum().reset_index()
      dfs=[df_sum]
      logger.info(f'have domain: {len(df_sum)}')

  if len(dfs) > 1:
    concated_df = pd.concat(dfs, ignore_index=True)
  else:
    concated_df = dfs[0]
  df_sum = concated_df.groupby('domain')['count'].sum().reset_index()
  logger.info(df_sum.head())
  logger.info(f'共有domain数据：{len(df_sum)}')
  # 排序
  sorted_url = df_sum.sort_values(by='count', ascending=False)
  logger.info(f'保存数据')

  # 保存处理后的结果到新的 CSV 文件
  sorted_url.to_csv(f'{output}', index=False)  # index=False 表示不保存索引列
  logger.info(sorted_url.head())

# get_domain_sum('result/domain_new', 'result/domain_new/global.csv')

def convert_to_gzip(input_dir):
  files = data_util.get_all_files(input_dir, suffix='.jsonl')
  for file in tqdm(files):
    data_util.compress_existing_file(file, f'{file}.gz')

# convert_to_gzip('/work/group1/datasets/v2/traffic/')

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

def merge_dataset(base_dir):
  metrics = []
  for i in tqdm(range(10), desc='merge shard'):
    file_name = f'shard_{i+1:02d}'
    metrics.append(merge_csv(os.path.join(base_dir, file_name), os.path.join(base_dir,f'{file_name}.csv')))

  global_total = 0
  duplicated_total = 0
  remain_total = 0
  for index, metric in enumerate(metrics) :
    global_total += metric['total']
    duplicated_total += metric['duplicated']
    logger.info(f'---- shard_{index+1:02d} -----')
    logger.info(f'---- total:{metric['total']} -----')
    logger.info(f'---- duplicated: {metric['duplicated']} -----')
    logger.info(f'---- remain:{metric['remain']} -----')

  logger.info(f'---- global -----')
  logger.info(f'---- total:{global_total} -----')
  logger.info(f'---- duplicated: {duplicated_total} -----')
  logger.info(f'---- remain:{remain_total} -----')

merge_dataset('/work/group1/data/r7_dclm')
#merge_csv('/work/group1/data/r7_dclm_r1/shard_05', '/work/group1/data/r7_dclm_r1/shard_05.csv')
