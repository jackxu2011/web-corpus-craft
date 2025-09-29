from sklearn.model_selection import train_test_split
import pandas as pd
import os
from pathlib import Path
import fasttext
import time
from tqdm import tqdm
from loguru import logger

WORKDIR = Path(__file__).resolve().parent.parent

logger.info(f'workdir: {WORKDIR}')

class FaxtTextInfer:
  def __init__(self, model_path):
    self.model_path = model_path
    self.model = fasttext.load_model(self.model_path)

  def inference(self, df, output_file):
    df = pd.read_csv(input_file)
    logger.info(f"Start inference...")
    start = time.time()
    neg_result = []
    pos_result = []
    for text in tqdm(df['text']):
      predictions = self.model.predict(text, k=1)  # 不再传 num_threads
      label = predictions[0][0].replace("__label__", "")
      result = {
          "prob": predictions[1][0],
          "text": text
      }
      if (label == '0'):
          neg_result.append(result)
      else:
          pos_result.append(result)
    end = time.time()

    neg_df = pd.DataFrame(neg_result)
    pos_df = pd.DataFrame(pos_result)

    output_dir = os.path.dirname(output_file)
    output_file_name = os.path.splitext(os.path.basename(output_file))[0]

    neg_df.to_csv(os.path.join(output_dir, f'neg_{output_file_name}.csv'), index=False)

    pos_df.to_csv(os.path.join(output_dir, f'pos_{output_file_name}.csv'), index=False)

    return end - start

def main():
    parser = argparse.ArgumentParser(description="fasttext 推理服务")
    parser.add_argument("model_path", type=str, help="模型地址")
    parser.add_argument("input_file", type=str, help="需要推理的文件")
    parser.add_argument("output_file", type=str, help="输出文件")
    args = parser.parse_args()

    infer = FaxtTextInfer(model_path=args.model_path)  # 初始化训练器
    df = pd.read_csv(args.input_file)
    trainer.inference(df, args.output_file)

if __name__ == "__main__":
  # 训练流程
  print('test')