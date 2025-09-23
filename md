export DATASET=sharegpt_clean_tr_en_chatbot_ui_history_long-00021,eprstmt-train_few_all-50-00008,medqa-001-00056, \
       source-airoboros2.2-category-wordgame-model_name-none-00044,source-glaive-code-assist-category-none-model_name-none-00048,\
       stemarticle-001-00060,source-airoboros2.2-category-plan-model_name-none-00039,ocnli-train_few_all-50-00018,\
       gpt4_moss-003-Code-00012,source-airoboros2.2-category-counterfactual_contextual-model_name-none-00033,alpaca_gpt4_data_zh-00004,\
       source-platypus-category-none-model_name-none-00052,oasst2_zh-00017,source-airoboros2.2-category-agent-model_name-none-00029,\
       source-CogStackMed-category-none-model_name-none-00024,sharegpt_clean_tr_zh_chatbot_ui_history-00022,source-none-category-none-model_name-none-00051,\
       source-airoboros2.2-category-misconception-model_name-none-00038,source-airoboros2.2-category-writing-model_name-none-00045,\
       source-none-category-none-model_name-GPT-4-00050,source-Econ_domain_expert-category-none-model_name-none-00025,source-airoboros2.2-category-general-model_name-none-00037,\
       source-airoboros2.2-category-roleplay-model_name-none-00040,source-metamath-category-none-model_name-none-00049,source-UnnaturalInstructions-category-none-model_name-none-00028,\
       source-cot_alpaca_gpt4-category-none-model_name-none-00047,sharegpt_clean_tr_en_chatbot_ui_history_gpt4-00020,source-airoboros2.2-category-detailed_writing-model_name-none-00034,\
       MathInstruct-00001,source-airoboros2.2-category-stylized_response-model_name-none-00041,source-airoboros2.2-category-theory_of_mind-model_name-none-00043,\
       gpt4_moss-003-ComplexInstruction-00013,tr_name_chatbot_ui_history-00053,afqmc-train-50-00003,source-GPT-4_Comparison_Data-category-none-model_name-none-00026,\
       source-airoboros2.2-category-card-model_name-none-00030,lima-train-00015,source-caseus_custom-category-none-model_name-none-00046,tr_name_chatbot_ui_history2-00054,\
       source-airoboros2.2-category-summarization-model_name-none-00042,source-airoboros2.2-category-editor-model_name-none-00035,health-001-00055,legal-001-00058,edu-001-00057,\
       actions-00002,CapybaraPure_Decontaminated-00000,source-airoboros2.2-category-cot-model_name-none-00032,source-CamelAI-category-none-model_name-none-00023,exp-001-00061,\
       new-00016,gpt4_moss-003-Writing-00014,source-LMSys_Chatbot_Arena-category-none-model_name-none-00027,cmnli-train-50-00006,gpt4_logic_multiple_task_6k_0719-00010,\
       react_use_function_annotated-00019,gpt4_logic_key_ele_2k-00009,source-airoboros2.2-category-coding-model_name-none-00031,\
       source-airoboros2.2-category-experience-model_name-none-00036,gpt4_moss-003-Brainstorming-00011,dialog_chatbot_ui_history-00007,\
       chid-train_few_all_50-00005,wiki-all-00059,if_sft_s_v2-3041,alpaca_gpt4_en,alpaca_gpt4_zh,code_alpaca,math,openhermes,codefeedback
cd /mnt/data-token-cpfs/group-web/wjp/code/sft/app && bash sft.sh


 nohup uv run utils/dedup_dataset.py /work/dclm-baseline/global-shard_01_of_10/ /work/dclm-url-dedup/shard_01 > logs/url/dedup01.log 2>&1 &
 nohup uv run utils/dedup_dataset.py /work/dclm-baseline/global-shard_02_of_10/ /work/dclm-url-dedup/shard_02 > logs/url/dedup02.log 2>&1 &
 nohup uv run utils/dedup_dataset.py /work/dclm-baseline/global-shard_03_of_10/ /work/dclm-url-dedup/shard_03 > logs/url/dedup03.log 2>&1 &
 nohup uv run utils/dedup_dataset.py /work/dclm-baseline/global-shard_04_of_10/ /work/dclm-url-dedup/shard_04 > logs/url/dedup04.log 2>&1 &
 nohup uv run utils/dedup_dataset.py /work/dclm-baseline/global-shard_05_of_10/ /work/dclm-url-dedup/shard_05 > logs/url/dedup05.log 2>&1 &
 nohup uv run utils/dedup_dataset.py /work/dclm-baseline/global-shard_06_of_10/ /work/dclm-url-dedup/shard_06 > logs/url/dedup06.log 2>&1 &
 nohup uv run utils/dedup_dataset.py /work/dclm-baseline/global-shard_07_of_10/ /work/dclm-url-dedup/shard_07 > logs/url/dedup07.log 2>&1 &
 nohup uv run utils/dedup_dataset.py /work/dclm-baseline/global-shard_08_of_10/ /work/dclm-url-dedup/shard_08 > logs/url/dedup08.log 2>&1 &
 nohup uv run utils/dedup_dataset.py /work/dclm-baseline/global-shard_09_of_10/ /work/dclm-url-dedup/shard_09 > logs/url/dedup09.log 2>&1 &
 nohup uv run utils/dedup_dataset.py /work/dclm-baseline/global-shard_10_of_10/ /work/dclm-url-dedup/shard_10 > logs/url/dedup10.log 2>&1 &

nohup uv run utils/extract_dclm_domain.py /work/dclm-url-dedup/shard_01 /work/dclm-domain-dedup/shard_01 >logs/domain/01.log 2>&1 &
nohup uv run utils/extract_dclm_domain.py /work/dclm-url-dedup/shard_02 /work/dclm-domain-dedup/shard_02 >logs/domain/02.log 2>&1 &
nohup uv run utils/extract_dclm_domain.py /work/dclm-url-dedup/shard_03 /work/dclm-domain-dedup/shard_03 >logs/domain/03.log 2>&1 &
nohup uv run utils/extract_dclm_domain.py /work/dclm-url-dedup/shard_04 /work/dclm-domain-dedup/shard_04 >logs/domain/04.log 2>&1 &
nohup uv run utils/extract_dclm_domain.py /work/dclm-url-dedup/shard_05 /work/dclm-domain-dedup/shard_05 >logs/domain/05.log 2>&1 &
nohup uv run utils/extract_dclm_domain.py /work/dclm-url-dedup/shard_06 /work/dclm-domain-dedup/shard_06 >logs/domain/06.log 2>&1 &
nohup uv run utils/extract_dclm_domain.py /work/dclm-url-dedup/shard_07 /work/dclm-domain-dedup/shard_07 >logs/domain/07.log 2>&1 &
nohup uv run utils/extract_dclm_domain.py /work/dclm-url-dedup/shard_08 /work/dclm-domain-dedup/shard_08 >logs/domain/08.log 2>&1 &
nohup uv run utils/extract_dclm_domain.py /work/dclm-url-dedup/shard_09 /work/dclm-domain-dedup/shard_09 >logs/domain/09.log 2>&1 &
nohup uv run utils/extract_dclm_domain.py /work/dclm-url-dedup/shard_10 /work/dclm-domain-dedup/shard_10 >logs/domain/10.log 2>&1 &
