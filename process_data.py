import pandas as pd
import os
from utils import data_util
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

def merge_datasets(path, output_file):
  # 获取当前目录下所有 .csv 文件的路径
  file_paths = glob.glob(os.path.join(path,"*.csv"))  # 可根据实际路径修改

  # 读取每个文件并存入列表
  dataframes = []
  for file in tqdm(file_paths, desc='读取文件'):
      df = pd.read_csv(file)
      dataframes.append(df[['text']])

  # 合并所有 DataFrame
  logger.info('begin merge dataframes')
  combined_df = pd.concat(dataframes, ignore_index=True)

  logger.info('begin clean text')
  combined_df['text'] = combined_df.text.apply(data_util.clean_text)
  combined_df = combined_df[(combined_df['text'] != "")]
  logger.info(combined_df.info())
  #去除重复行
  df_cleaned = data_util.drop_duplicates(combined_df)

  df_cleaned.to_csv(f'data/{output_file}.csv', index=False)
  logger.info(f"合并后的数据集数据: {len(df_cleaned)}")

def extra_data(N, input_dir, file_suffix, out_file='negative.csv'):
  file_paths = glob.glob(os.path.join(input_dir, f'*.{file_suffix}'))
  batch_sample = N//len(file_paths)*2
  dataframes = []
  for file in tqdm(file_paths, desc="处理文件"):
      df = None
      if file.endswith('csv'):
          df = pd.read_csv(file)
      else:
          df = pd.read_json(file, compression='infer', lines=True)
      df = df.sample(n=batch_sample)
      df['text'] = df.text.apply(data_util.clean_text)
      df = df[(df['text'] != "")]
      dataframes.append(df.dropna(subset=['text']))

  combined_df = pd.concat(dataframes, ignore_index=True)
  combined_df = combined_df.sample(n=N)

  print(combined_df.info())
  combined_df['text'].to_csv(f'data/{out_file}', index=False)

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


def filter_score(path, output_file, score=4, op='ge', score_key='traffic_relevance_score', columns=['text']):
  file_paths = glob.glob(os.path.join(path, '*.csv'))
  df_array = []
  columns.insert(0, score_key)
  for file in tqdm(file_paths):
    df = pd.read_csv(file)
    print(f'{file} has {len(df)} rows')
    # 将列转换为数值类型，无法转换的设为NaN
    df[score_key] = pd.to_numeric(df[score_key], errors='coerce')

    # 过滤掉转换后为NaN的行（即非数值的行）
    df = df.dropna(subset=[score_key])
    if op == 'ge':
      df = df[(df[score_key]<=5)]
      df = df[(df[score_key]>=score)]
    else:
      df = df[(df[score_key]<=score)]
      df = df[(df[score_key]>=0)]
    df = data_util.drop_duplicates(df)
    df_array.append(df[columns])

  combine_df = pd.concat(df_array, ignore_index=True)

  result = data_util.drop_duplicates(combine_df)

  result.to_csv(output_file, index=False)
  print(f'分数{op}_{score}的有{len(result)}条！')

# merge_datasets('data/pos', 'r7/positive')
# merge_datasets('data/neg', 'r7/negative')
# extra_data(30000, 'test', 'csv', 'llm_r1_le1_3w.csv')
# add_label(in_file = 'data/negative_500000.csv', label = 0 , out_file='test_negative_500000.txt')

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
# duplicated('clean/paper.csv', 'paper_dupl.csv')

def merge_files(path, output_file):
  # 获取当前目录下所有 .csv 文件的路径
  file_paths = glob.glob(os.path.join(path,"*.csv"))  # 可根据实际路径修改

  # 读取每个文件并存入列表
  dataframes = []
  for file in tqdm(file_paths, desc='read files'):
      df = pd.read_csv(file)
      dataframes.append(df)

  # 合并所有 DataFrame
  combined_df = pd.concat(dataframes, ignore_index=True)

  print(f"原始数据行数: {len(combined_df)}")
  print(f"重复行数量: {combined_df.duplicated().sum()}")
  # 3. 去除重复行
  # 默认保留第一次出现的行，删除后续重复行
  df_cleaned = combined_df.drop_duplicates()

  print(f"start to save")
  df_cleaned.to_csv(output_file, index=False)

#merge_files('r6_dclm/r6_dclm_05', 'r6_dclm/shard_05.csv')

def extract_main_domain(url):
    # 提取主域名（自动识别公共后缀如 .com、.co.uk）
    extracted = tldextract.extract(url)
    # 组合主域名和后缀（如 "example" + "com" → "example.com"）
    return extracted.fqdn

def get_url_conunt(input, output):
  # 获取当前目录下所有 .csv 文件的路径
  df = pd.read_csv(f'{input}')
  print(f'读取文件完成')

  df['url'] = df.url.apply(extract_main_domain)

  group_url = df.groupby('url').size().reset_index(name='count')

  print(f'共有url数据：{len(group_url)}')
  # 排序
  sorted_url = group_url.sort_values(by='count', ascending=False)
  print(f'保存数据')

  # 保存处理后的结果到新的 CSV 文件
  sorted_url.to_csv(f'{output}', index=False)  # index=False 表示不保存索引列
  print(sorted_url.head())


def get_domain_sum(input, output):
  # 获取当前目录下所有 .csv 文件的路径
  logger.info('get file list')
  files = data_util.get_all_files(input)
  dfs = []
  for file in tqdm(files):
    df = pd.read_csv(file)
    df_sum = df.groupby('domain')['count'].sum().reset_index()
    dfs.append(df_sum)
  logger.info('读取文件完成')

  concated_df = pd.concat(dfs, ignore_index=True)
  logger.info(concated_df.head())

  sum_url = concated_df.groupby('domain')['count'].sum().reset_index()

  logger.info(f'共有url数据：{len(sum_url)}')
  # 排序
  sorted_url = sum_url.sort_values(by='count', ascending=False)
  logger.info(f'保存数据')

  # 保存处理后的结果到新的 CSV 文件
  sorted_url.to_csv(f'{output}', index=False)  # index=False 表示不保存索引列
  logger.info(sorted_url.head())

# get_domain_sum('result/domain', 'result/domain/global.csv')

def convert_to_gzip(input_dir):
  files = data_util.get_all_files(input_dir, suffix='.jsonl')
  for file in tqdm(files):
    data_util.compress_existing_file(file, f'{file}.gz')

# convert_to_gzip('/work/group1/datasets/v2/traffic/')
