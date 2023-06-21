# 第二周任务细则

- 任务主题：尝试开源模型的完全微调和参数高效微调
  - 模型：LLaMA，Chinese-LLaMA, ChatGLM
  - 涉及训练方法： finetune, LoRA finetune, QLoRA finetune
  - 涉及数据集：lawgpt的 [法律数据集](http://github.com/pengxiao-song/LaWGPT )


- #### 阶段一：搭建模型

- 任务目标：完成开源LLM模型的训练

  - 孙国恒：Chinese-LLaMA-Plus-7B

    - 项目来源：https://huggingface.co/ziqingyang/chinese-llama-plus-lora-7b

  - 余杰：llama-7b-hf+Alpaca-LoRA
    - 项目来源：https://github.com/tloen/alpaca-lora

  - 骆思缘：ChatGLM
    - 项目来源：https://huggingface.co/THUDM/chatglm-6b



- #### 阶段二：模型训练与评测

- 任务目标：完成对训练好的模型评测，生成评测结果报告

- 评测方案：
  - 对比不同base model对训练效果的影响
    - llama作为base model
    - ChatGLM作为base model
  - 对比不同数据格式对训练效果的影响
    - llama基座epoch3与epoch6对比
  - 对比不同模板对模型表现的影响
    - llama在alpaca模板下的表现
    - llama在vicuna模板下的表现
  - 对比评价得分下实际模型问答效果
    - 人工评判分析评测数据是否合理

- 评测方法：
  - ChatGLM评分比对：将每个模型的训练结果与ChatGLM投入chatgpt-3.5对比，得到以ChatGLM为基准的
  - 人工结果比对：人为评判垂直领域问答结果合理性与准确性
  - 最终综合评分与人工评判提炼评测发现问题

测试结果报告查看针对垂直领域的LLM训练及测试.pdf
