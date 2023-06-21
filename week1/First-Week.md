# 第一周任务细则

- 任务主题：部署运行6个开源LLM模型，进行测对比
  - 模型：LLaMA，Alpaca，Vicuna，ChatGLM，MOSS，MPT
  - 小组成员：孙国恒，余杰，骆思缘





- #### 阶段一：项目部署

- 任务目标：完成开源LLM模型配置部署

  - 孙国恒：Vicuna(7B, 13B)，MPT
    - 项目来源参考：
      - Vicuna-7B:https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
      - Vicuna-13B:https://huggingface.co/lmsys/vicuna-13b-delta-v1.1 

  - 余杰：，LLaMA(7B, 13B)
    - 项目来源参考：
      - LLaMA-7B:https://huggingface.co/yahma/llama-7b-hf
      - LLaMA-13B:https://huggingface.co/yahma/llama-13b-hf
      - LLaMA-7B(cpp)https://github.com/ggerganov/llama.cpp
      - MOSS:https://huggingface.co/fnlp/moss-moon-003-sft
      

  - 骆思缘：ChatGLM，Alpaca(7B, 13B)
    - 项目来源参考：
      - Alpaca:https://huggingface.co/Sosaka/Alpaca-native-4bit-ggml
      - ChatGLM-6B:https://github.com/THUDM/ChatGLM-6B


- #### 阶段二：项目评测

- 任务目标：完成对已部署的模型评测，生成评测结果报告

- 评测方案(6项，细则见LLM_test_week1.md)： 
  - Test1:基础能力测试
  - Test2:中文能力测试
  - Test3:对评价者的测试：反转测试？
  - Test4:量化对模型回答质量的影响
  - Test5:生成阶段参数对模型回答质量的影响
  - Test6:组织形式（模板）对模型回答质量的影响
