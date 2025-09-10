from sklearn.model_selection import train_test_split
import pandas as pd
import os
import fasttext
import time
from tqdm import tqdm
from loguru import logger

WORKDIR = os.getcwd()

class FaxtTextTrainer:
  def __init__(self, train_path):
    self.train_path = train_path
    self.data_path = os.path.join(train_path, "data")
    self.model_path = os.path.join(train_path, "models")
    os.makedirs(self.data_path, exist_ok=True)
    os.makedirs(self.model_path, exist_ok=True)

  def split_data(self, pos_file='positive.csv', neg_file='negative.csv', round=1, val=0.2):
    pos_df = pd.read_csv(os.path.join(self.data_path, f'r{round}/{pos_file}'))
    neg_df = pd.read_csv(os.path.join(self.data_path, f'r{round}/{neg_file}'))
    # 合并正负样本
    data_X = pd.concat((pos_df['text'], neg_df['text']))
    # 创建标签：正样本为1，负样本为0
    data_y = pd.Series([1]*pos_df.shape[0] + [0]*neg_df.shape[0])

    train_X, val_X, train_y, val_y = train_test_split(
      data_X, data_y,
      test_size=val,
      stratify=data_y,  # 保持类别比例
      random_state=42
    )

    self.save_datasets('train', train_X, train_y, round)
    self.save_datasets('val', val_X, val_y, round)

  def save_datasets(self, sub_set, X, y, round):
    data = []
    for (text, label) in zip(X, y):
      data.append(f"__label__{label} {text}")

    save_file = os.path.join(self.data_path, f"r{round}/{sub_set}.txt")
    # 将训练数据写入文本文件
    with open(save_file, "w") as f:
      for line in data:
        f.write(line.strip() + "\n")  # 每行一个样本
    logger.info(f"write {sub_set}: {len(data)} rows to {save_file}")

  def train(self, model_name='traffic', round=1):
    """
    训练FastText分类模型并保存
    """
    # 模型配置参数
    config = {
      'input': os.path.join(self.data_path, f'r{round}/train.txt'),  # 训练数据路径
      'vector_dim': 64,        # 词向量维度
      'lr': 0.8,               # 学习率
      'word_ngrams': 3,        # 最大n-gram长度
      'min_count': 3,          # 最小词频阈值
      'epoch': 50,              # 训练轮数
    }

    # 训练监督模型
    model = fasttext.train_supervised(
      input=config['input'],
      dim=config['vector_dim'],
      lr=config['lr'],
      word_ngrams=config['word_ngrams'],
      min_count=config['min_count'],
      epoch=config['epoch'],
      verbose=2  # 显示训练日志
    )

    # 保存模型文件
    model_file = os.path.join(self.model_path, f"{model_name}_r{round}.bin")
    model.save_model(model_file)

  def test(self, file, model_name='traffic', round=1):
    model = fasttext.load_model(os.path.join(self.model_path, f"{model_name}_r{round}.bin"))

    logger.info(model.test(os.path.join(self.data_path, file)))

  def inference(self, file, out_file, model_name='traffic', round=1):
    model = fasttext.load_model(os.path.join(self.model_path, f"{model_name}_r{round}.bin"))

    df = pd.read_csv(os.path.join(self.data_path, file))
    logger.info(f"Start inference...")
    start = time.time()
    neg_result = []
    pos_result = []
    for text in tqdm(df['text']):
      predictions = model.predict(text, k=1)  # 不再传 num_threads
      label = predictions[0][0].replace("__label__", "")
      if (label == '0'):
        neg_result.append({
          "prob": predictions[1][0],
          "text": text
        })
      else:
        if predictions[1][0] > 0.9:
          pos_result.append({
            "prob": predictions[1][0],
            "text": text
          })
    end = time.time()

    neg_df = pd.DataFrame(neg_result)

    pos_df = pd.DataFrame(pos_result)

    neg_df.to_csv(os.path.join(self.data_path, f'r{round}/neg_{out_file}.csv'), index=False)

    pos_df.to_csv(os.path.join(self.data_path, f'r{round}/pos_{out_file}_0.9.csv'), index=False)

    return end - start

  def predict_test(self, file, model_name='traffic', round=1, positive_label="__label__1", negative_label="__label__0"):
    # 初始化计数器
    TP = 0  # 真阳性
    FP = 0  # 假阳性
    FN = 0  # 假阴性
    TN = 0  # 真阴性
    total_samples = 0  # 总样本数（用于校验）

    # 读取测试集文件
    try:
        with open(os.path.join(self.data_path, file), 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"测试集文件不存在：{test_data_path}")
    except Exception as e:
        raise Exception(f"读取测试集失败：{str(e)}")

    model = fasttext.load_model(os.path.join(self.model_path, f"{model_name}_r{round}.bin"))

    # 遍历每行数据，解析标签和文本
    for line in tqdm(lines, desc="测试集预测进度", unit="条"):
        line = line.strip()  # 去除首尾空格和换行符
        if not line:  # 跳过空行
            continue

        # 分割标签和文本（按第一个空格分割，确保标签正确提取）
        parts = line.split(' ', 1)  # 只分割一次，避免文本中含空格导致错误
        if len(parts) != 2:
            print(f"警告：跳过格式错误的行（未找到标签和文本）：{line}")
            continue

        true_label, text = parts[0], parts[1]
        total_samples += 1

        # 模型预测
        predictions = model.predict(text, k=1)
        predicted_label = predictions[0][0]  # 预测标签

        # 更新计数器
        if true_label == positive_label:
            # 真实为正样本
            if predicted_label == positive_label:
                TP += 1
            else:
                FN += 1
        elif true_label == negative_label:
            # 真实为负样本
            if predicted_label == positive_label:
                FP += 1
            else:
                TN += 1
        else:
            # 未知标签（跳过并提示）
            print(f"警告：跳过未知标签的行（标签：{true_label}）：{line}")
            continue

    # 计算指标（处理分母为0的情况）
    precision = TP / float(TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / float(TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / float(TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    print(precision, recall, f1, accuracy)
    # 整理结果
    metrics = {
        "总样本数": total_samples,
        "有效样本数": TP + TN + FP + FN,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "精确率（正确率）": "%.4f" % precision,
        "召回率": "%.4f" % recall,
        "F1分数": "%.4f" % f1,
        "整体准确率": "%.4f" % accuracy
    }
    # 打印评估结果
    print("\n" + "="*50)
    print(f"测试集评估结果（文件：{file}）")
    print("="*50)
    print(f"总样本数：{metrics['总样本数']} 条")
    print(f"有效样本数（排除格式错误/未知标签）：{metrics['有效样本数']} 条")
    print("-"*30)
    print(f"真阳性（TP）：{metrics['TP']} 条")
    print(f"假阳性（FP）：{metrics['FP']} 条")
    print(f"假阴性（FN）：{metrics['FN']} 条")
    print(f"真阴性（TN）：{metrics['TN']} 条")
    print("-"*30)
    print(f"精确率（正确率）：{metrics['精确率（正确率）']}")
    print(f"召回率：{metrics['召回率']}")
    print(f"F1分数：{metrics['F1分数']}")
    print(f"整体准确率：{metrics['整体准确率']}")
    print("="*50)
    return metrics

if __name__ == "__main__":
  # 训练流程
  trainer = FaxtTextTrainer(train_path=WORKDIR)  # 初始化训练器
  train_iter = 7
  trainer.split_data(round=train_iter)
  trainer.train(round=train_iter)
  trainer.test(f'r{train_iter}/val.txt', round=train_iter)
  trainer.predict_test(f'r{train_iter}/val.txt', round=train_iter)
  trainer.inference('test_negative_500000.csv', 'test_neg_result', round=train_iter)
