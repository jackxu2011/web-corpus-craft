# 模型训练流程

## 新建算法库

### 1. xmegatron 算法库

* 代码源: 仅镜像
* 镜像仓库: 共享,`qwen-2.5-0.5beval/xmegatron:6.0`


### 2. sft 算法库

* 代码源: 存储
* 算法路径：/mnt/data-token-cpfs/group-web/wjp/code/sft
* 镜像仓库：预制,`preset/pytorch:2.5.1-cuda11.8-xmegatronv1.2.0-ssh-vscode-jupyterlab-v5000`

### 3. opencompass 算法库

* 代码源: 存储
* 算法路径：/mnt/data-token-cpfs/group-web/sph/eval
* 镜像仓库：共享,`opencompass050/0.1`

## 训练步骤

### step 1: tokenizer

* 算法库： xmegatron
* 计算规格: 16C100G 0GPU
* 存储挂载: 两个cpfs都需要挂上
* 算法输出路径: /mnt/si001117d1p1/default/group1/tokenizer
* 启动命令:

    ```bash
    cd /workspace/v5000_megatron_20250905/toolkits/pretrain_data_preprocessing
    bash run_make_pretraining_dataset_megatron.sh /mnt/data-token-cpfs/group-web/group1/datasets/v4/jsonl Qwen2Tokenizer text /mnt/si001117d1p1/default/group1/tokenizer/v4/data /mnt/data-token-cpfs/group-web/eval/models/Qwen2.5-0.5B-Instruct/ /mnt/si001117d1p1/default/group1/tokenizer/v4/domain_corpus.txt
    ```

### step 2: sample corpus

* 算法库： xmegatron
* 计算规格: 16C100G 0GPU
* 存储挂载: 两个cpfs都需要挂上
* 算法输出路径: /mnt/si001117d1p1/default/group1/task5
* 启动命令:

    ```bash
    cd /workspace/v5000_megatron_20250905
    bash merge_sample_token.sh
    ```
* 启动参数：

    |变量|值|
    |---|---|
    |GENERAL_CORPUS_TXT	| /mnt/data-token-cpfs/group-web/eval/tokens/qwen2/fineweb_dclm/general_corpus_70B.txt
    |DOMAIN_CORPUS_TXT | /mnt/si001117d1p1/default/group1/tokenizer/v4/domain_corpus.txt
    |GENERAL_TOKENS |	70000000000
    |DOMAIN_TOKENS	| 30000000000
    |MIXED_CORPUS_TXT	| /mnt/si001117d1p1/default/group1/task5/tokens/mixed_corpus_100B.txt

### step 3: cpt

* 算法库： xmegatron
* 计算规格: 128C900G 8GPU
* 存储挂载: 两个cpfs都需要挂上
* 算法输出路径: /mnt/si001117d1p1/default/group1/task5
* 启动命令:

    ```bash
    cd /workspace/v5000_megatron_20250905/zj_examples/V5000/qwen2_5
    bash xmegatron_qwen2.5_0.5b_wjp_cpt_2k.sh
    ```

* 启动参数：

    |变量|值|
    |---|---|
    |SAVE_INTERVAL | 1000
    |PRETRAIN_CHECKPOINT_PATH	| /mnt/data-token-cpfs/group-web/eval/models/Qwen2.5-0.5B-mcore-TP-1-PP-1/
    |DATASET_PATH	| /mnt/si001117d1p1/default/group1/task5/tokens/mixed_corpus_100B.txt
    |TRAIN_TOKENS	| 100000000000
    |WARMUP_TOKENS |	1000000000
    |PR	| fp16

### step 4: mcore2hf

* 算法库： xmegatron
* 计算规格: 128C900G 8GPU
* 存储挂载: 两个cpfs都需要挂上
* 算法输出路径: /mnt/si001117d1p1/default/group1/task5
* 启动命令:

    ```bash
    cd /workspace/v5000_megatron_20250905/toolkits/distributed_checkpoints_convertor/scripts/qwen2_5
    TP=1 PP=1 bash run_mcore2hf_qwen2.5.sh
    ```

* 启动参数：

    |变量|值|
    |---|---|
    |MODEL_SIZE	| 0.5B
    |STORAGE_PATH	| /mnt/data-token-cpfs/group-web
    |LOAD_DIR	| /mnt/si001117d1p1/default/group1/task5/checkpoint/mcore-qwen25-0.5B-pretrain-TP1-PP1/
    |HF_DIR	| /mnt/data-token-cpfs/group-web/eval/models/Qwen2.5-0.5B-Instruct/
    |SAVE_DIR	| /mnt/si001117d1p1/default/group1/task5/checkpoint/mcore-qwen25-0.5B-pretrain-TP1-PP1/hg_checkpoint

### step 5: sft

* 算法库： sft
* 计算规格: 128C900G 8GPU
* 存储挂载: 两个cpfs都需要挂上
* 算法输出路径: /mnt/si001117d1p1/default/group1/task5
* 启动命令:

    ```bash
    export DATASET=sharegpt_clean_tr_en_chatbot_ui_history_long-00021,eprstmt-train_few_all-50-00008,medqa-001-00056,source-airoboros2.2-category-wordgame-model_name-none-00044,source-glaive-code-assist-category-none-model_name-none-00048,stemarticle-001-00060,source-airoboros2.2-category-plan-model_name-none-00039,ocnli-train_few_all-50-00018,gpt4_moss-003-Code-00012,source-airoboros2.2-category-counterfactual_contextual-model_name-none-00033,alpaca_gpt4_data_zh-00004,source-platypus-category-none-model_name-none-00052,oasst2_zh-00017,source-airoboros2.2-category-agent-model_name-none-00029,source-CogStackMed-category-none-model_name-none-00024,sharegpt_clean_tr_zh_chatbot_ui_history-00022,source-none-category-none-model_name-none-00051,source-airoboros2.2-category-misconception-model_name-none-00038,source-airoboros2.2-category-writing-model_name-none-00045,source-none-category-none-model_name-GPT-4-00050,source-Econ_domain_expert-category-none-model_name-none-00025,source-airoboros2.2-category-general-model_name-none-00037,source-airoboros2.2-category-roleplay-model_name-none-00040,source-metamath-category-none-model_name-none-00049,source-UnnaturalInstructions-category-none-model_name-none-00028,source-cot_alpaca_gpt4-category-none-model_name-none-00047,sharegpt_clean_tr_en_chatbot_ui_history_gpt4-00020,source-airoboros2.2-category-detailed_writing-model_name-none-00034,MathInstruct-00001,source-airoboros2.2-category-stylized_response-model_name-none-00041,source-airoboros2.2-category-theory_of_mind-model_name-none-00043,gpt4_moss-003-ComplexInstruction-00013,tr_name_chatbot_ui_history-00053,afqmc-train-50-00003,source-GPT-4_Comparison_Data-category-none-model_name-none-00026,source-airoboros2.2-category-card-model_name-none-00030,lima-train-00015,source-caseus_custom-category-none-model_name-none-00046,tr_name_chatbot_ui_history2-00054,source-airoboros2.2-category-summarization-model_name-none-00042,source-airoboros2.2-category-editor-model_name-none-00035,health-001-00055,legal-001-00058,edu-001-00057,actions-00002,CapybaraPure_Decontaminated-00000,source-airoboros2.2-category-cot-model_name-none-00032,source-CamelAI-category-none-model_name-none-00023,exp-001-00061,new-00016,gpt4_moss-003-Writing-00014,source-LMSys_Chatbot_Arena-category-none-model_name-none-00027,cmnli-train-50-00006,gpt4_logic_multiple_task_6k_0719-00010,react_use_function_annotated-00019,gpt4_logic_key_ele_2k-00009,source-airoboros2.2-category-coding-model_name-none-00031,source-airoboros2.2-category-experience-model_name-none-00036,gpt4_moss-003-Brainstorming-00011,dialog_chatbot_ui_history-00007,chid-train_few_all_50-00005,wiki-all-00059,if_sft_s_v2-3041,alpaca_gpt4_en,alpaca_gpt4_zh,code_alpaca,math,openhermes,codefeedback
    cd /mnt/data-token-cpfs/group-web/wjp/code/sft/app && bash qwen2.5_sft.sh
    ```

* 启动参数：

    |变量|值|
    |---|---|
    |STAGE | sft
    |MODEL_NAME_OR_PATH	| /mnt/si001117d1p1/default/group1/task5/checkpoint/mcore-qwen25-0.5B-pretrain-TP1-PP1/hg_checkpoint
    |DATASET_DIR | /mnt/data-token-cpfs/group-web/eval/data/sft
    |OUTPUT_CHECKPOINT_DIR | /mnt/si001117d1p1/default/group1/task5/sft
    |NNODES	| 1
    |TQ_GPU_NUM	| 8
    |MICRO_BATCH_SIZE	| 4
    |GRAD_ACC	| 4
    |SAVE_INTERVAL | 1000

### step 6: eval

* 算法库： opencompass
* 计算规格: 128C900G 8GPU
* 存储挂载: 两个cpfs都需要挂上
* 算法输出路径: /mnt/si001117d1p1/default/group1/task5
* 启动命令:

    ```bash
    . /root/miniconda/etc/profile.d/conda.sh
    export PATH=/usr/local/cuda-11.7/bin:/root/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    conda activate python310_torch25_cuda
    bash /mnt/data-token-cpfs/group-web/group1/EVAL/run_eval-wjp.sh
    ```

* 启动参数：

    |变量|值|
    |---|---|
    |CONFIG	| /mnt/data-token-cpfs/group-web/group1/EVAL/eval_qwen_vllm_group1.py
    |OPENCOMPASS_ROOT |	/mnt/data-token-cpfs/group-web/group1/EVAL/opencompass
    |OUTPUT_RESULT_DIR | /mnt/si001117d1p1/default/group1/task5/eval
    |MODEL_PATH	| /mnt/si001117d1p1/default/group1/task5/sft
    |VLLM_ALLOW_LONG_MAX_MODEL_LEN | 1
