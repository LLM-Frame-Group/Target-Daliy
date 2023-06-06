# LLM评估方案



#### 选取模型：

- LLaMA(&llama.cpp)

- Vicuna

- Alpaca 

- MOSS

- ChatGLM



## LLaMA（&llama.cpp）

模型来源：

- 7B:https://huggingface.co/yahma/llama-7b-hf
- 13B:https://huggingface.co/yahma/llama-13b-hf
- 7B(cpp)https://github.com/ggerganov/llama.cpp



模型简介：

LLaMA (Large Language Model Meta AI)，是Meta开发的开源基础大语言模型，旨在用更少的计算能力和资源来进行训练和工作。LLaMA模型在大量未标记的数据集上进行训练，所提供的不同模型尺寸包括7B，13B，33B和65B。其中65B和33B是在1.4万亿上的tokens进行训练的，最小的模型7B也是在1万亿tokens上训练的。模型训练数据集种包含20多种语言，但是大部分数据集由英语文本组成。





## Vicuna

模型来源:

- 7B:https://huggingface.co/lmsys/vicuna-7b-delta-v1.1
- 13B:https://huggingface.co/lmsys/vicuna-13b-delta-v1.1



模型简介：

Vicuna是一个开源的大语言模型，通过微调LLaMA并从ShareGPT收集的大约7万个用户的共享对话进行训练。在于GPT-4的评估显示，Vicuna-13B能达到OpenAl ChatGPT和Google Bard 90%以上的质量，且优于LLaMA和Alpaca等模型。



## Alpaca

模型来源：

- https://huggingface.co/chavinlo/alpaca-native
- https://huggingface.co/lmsys/vicuna-7b-delta-v1.1

模型简介：

Alpaca是斯坦福开发的根据对LLaMA模型微调产生的的大语言模型。对LLaMA模型的改进包括，提供新的prompt，使用更激进的batch解码，通过放弃分类和非分类指令之间的差异来简化数据生成管道等。


## MOSS

模型来源：

- https://huggingface.co/fnlp/moss-moon-003-sft

模型简介：

MOSS是一个支持中英双语和多种插件的开源对话语言模型，`moss-moon`系列模型具有160亿参数。MOSS基座语言模型在约七千亿中英文以及代码单词上预训练得到，后续经过对话指令微调、插件增强学习和人类偏好训练具备多轮对话能力及使用多种插件的能力。

## ChatGLM

模型来源：

- https://huggingface.co/THUDM/chatglm-6b

模型简介：

ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持。

## 样例
* Question
```
在汤姆考试的前一天晚上，隔壁的邻居举行了一次聚会，音乐让他无法入睡，他给邻居打了电话，请她不要吵，邻居却突然挂断了电话。 汤姆并不打算射杀任何人，只是将枪口对准邻居天花板的一个角度，他很想对邻居的房子造成一些破坏，以缓解他的愤怒。 然而，子弹从天花板上弹出，击中了一个人的背部，导致他死亡。请问汤姆是否犯了罪，是什么罪？
```
* Assistant 1: MOSS
```
这个案例涉及到谋杀罪和过失杀人罪。如果汤姆没有杀人的故意，只是过失杀死了邻居，那么他可能会被判定为过失杀人罪。
但是，如果汤姆有杀人的故意，那么他将面临谋杀罪的指控。
无论哪种情况，汤姆都应该被起诉并受到法律制裁。
```
* Assistant 2: CharGLM
 ```
 汤姆的行为是犯罪，他射杀的人是在聚会中死亡的，汤姆的行为符合杀手罪的定义。
 杀手罪是指故意杀人罪，是指故意非法
 ```
* Review
```
8 6

Assistant 1 provides a very clear and accurate explanation of the legal consequences that Tom could face for his actions and is helpful in distinguishing between different types of homicide, which would be useful for someone who may not be familiar with legal terminology. However, the response could benefit from more detail and nuances regarding the specific circumstances of the case.
Assistant 2's response appears to be incomplete and cuts off in the middle of a sentence. Therefore, it is not very helpful, and it also includes legal terminology without sufficient explanations of what that terminology actually means.
```

```
8 6

Assistant 1 对汤姆的行为可能面临的法律后果做出了非常清晰准确的解释，有助于区分不同类型的凶杀案，这对可能不熟悉法律术语的人来说很有用。然而，有关案件具体情况的更多细节和细微差别可能会使答复受益。

Assistant 1 的回答似乎不完整，并在句子中间中断。因此，它没有太大帮助，它还包括法律术语，但没有充分解释该术语的实际含义。
```

## Test1:基础能力测试

* 涉及模型
  - LLaMA 7B
  - Vicuna 7B
  - Alpaca 7B
  - MOSS 7B
  - ChatGLM 7B

* 模型参数选定(LLaMA 7B除外)
  * temperature=0.7 （LLaMA:0.8）
  * top_p=0.8（LLaMA:0.95）
  * top_k=50（LLaMA:40）
  * num_beams=1（LLaMA:1）
  * max_new_tokens=1024 （LLaMA:512）
  * repetition_penalty=1.02 （LLaMA:1.10）

  

* 评分标准
  - 使用OpenAI提供的API进行测试，gpt3.5作为评判者。
  - 模型两两pk，由gpt3.5打出1-10分，统计每个模型每次pk的平均分数，再取平均作为最终得分

* 评分prompt
  - 使用FastChat的prompt

* 测试集
  - 从FastChat提供的80道题目中筛选出18道，共有```generic, knowledge, roleplay, common-sense, fermi, counterfactual, coding, math, writing```8类主题，每个类包含两个问题，这18道题目构成了```EN_test_16```
  - ```
    {"question_id": 4, "text": "How can I increase my productivity while working from home?", "category": "generic"}
    {"question_id": 5, "text": "Can you explain the basics of quantum computing?", "category": "generic"}
    {"question_id": 15, "text": "Describe a scenario where artificial intelligence could be used to improve the quality and efficiency of healthcare delivery.", "category": "knowledge"}
    {"question_id": 16, "text": "Explain the process of gene editing using CRISPR-Cas9 technology, and discuss its potential applications and ethical implications.", "category": "knowledge"}
    {"question_id": 22, "text": "As a pirate captain, what would you say to your crew to motivate them to search for hidden treasure?", "category": "roleplay"}
    {"question_id": 23, "text": "If you were a Shakespearean character, how would you declare your love for someone in a soliloquy?", "category": "roleplay"}
    {"question_id": 32, "text": "What are some subtle clues that suggest someone is pretending to understand a topic or conversation when they are actually confused or uninformed?", "category": "common-sense"}
    {"question_id": 33, "text": "Why might someone choose to use a paper map or ask for directions instead of relying on a GPS device or smartphone app?", "category": "common-sense"}
    {"question_id": 42, "text": "How many atoms are in a grain of salt? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step.", "category": "fermi"}
    {"question_id": 43, "text": "How many lightning strikes occur on Earth each day? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step.", "category": "fermi"}
    {"question_id": 53, "text": "What if the Black Death had not occurred in the 14th century?", "category": "counterfactual"}
    {"question_id": 54, "text": "What if Isaac Newton had focused on biology instead of physics?", "category": "counterfactual"}
    {"question_id": 62, "text": "Implement a Python function to find the longest common subsequence of two input strings using dynamic programming.", "category": "coding"}
    {"question_id": 63, "text": "Implement a regular expression in Python to validate an email address.", "category": "coding"}
    {"question_id": 68, "text": "Given that f(x) = 5x^3 - 2x + 3, find the value of f(2).", "category": "math"}
    {"question_id": 69, "text": "Solve for x in the equation 3x + 10 = 5(x - 2).", "category": "math"}
    {"question_id": 75, "text": "Draft an apology email to a customer who experienced a delay in their order, and provide reassurance that the issue has been resolved.", "category": "writing"}
    {"question_id": 76, "text": "Write a script for a YouTube video exploring the history and cultural significance of jazz.", "category": "writing"}

    
    ```
| 评测对象        | 评分 |
| :-------------- | ---: |
| LLaMA-7B-16bit  |  5.2 |
| Vicuna-7B-16bit |  7.6 |
| Alpaca-7B-16bit |  7.1 |
| MOSS            |  8.2 |
| ChatGLM-6B      |  7.2 |
|                 |      |

##### 评测结果分析：

Test1是针对5个模型的基础能力测试。中令5个模型使用相同模型参数（LLaMA除外）对同一批英文文本任务进行推理得到结果，并将结果投入GPT-3.5做1-10分评测，最终对比5个模型的基础能力效果，得到量化关系。在GPT-3.5给定的评测结果中，对结果评分最高的为MOSS，最低为LLaMa-7B-16bit，Vicuna-7B-16bit评分第二，ChatGLM与Alpaca-7B-16bit评分非常接，都低于  Vicuna-7B-16bit。GPT-3.5分析结果表示对英文文本分析中，MOSS的性能相对其他几个模型要更优越。

##### 评测思考：

实验为体现控制变量原则，在测试过除LLaMA外的四个模型，对微调参数后的结果质量改变较小后，将ChatGLM-6B，Alpaca-7B-16bit，  Vicuna-7B-16bit三个模型参数调整到与MOSS相同。但此方法可能存在对原模型默认参数改变后，输出质量可能下降，导致结果可能没有达到最优的问题。


## Test2:中文能力测试

* 涉及模型
  - 基础能力测试中涉及的6种模型中，MOSS和ChatGLM是针对中文的，其他模型的训练语料中中文占比很少，初步测试发现，只有Vicuna具有较好的表现，因此纳入以下三种模型：
    - Vicuna 7B
    - MOSS 7B
    - ChatGLM 7B

* 模型参数选定(LLaMA 7B作回答质量检测)
  * temperature=0.7 
  * top_p=0.8
  * top_k=50
  * num_beams=1
  * max_new_tokens=1024 
  * repetition_penalty=1.02 


* 评分标准
  - 使用OpenAI提供的API进行测试，gpt3.5作为评判者。
  - 模型两两pk，由gpt3.5打出1-10分，统计每个模型每次pk的平均分数，再取平均作为最终得分


* 评分prompt
  - 使用FastChat的prompt. TODO: 自制中文prompt


* 测试集
  - Z-Bench提供了3类问题，用于测试LLM在中文上的```基础能力, 进阶能力, 垂直能力```，我们从每类中挑选出6个问题，构成了```ZH_test_16```
  
  - ```
    //common
    {"question_id":"1","text":"1955 年谁是美国总统？他是什么党派？"}
    {"question_id":"2","text":"猫、白菜和鲸鱼都属于什么？"}
    {"question_id":"3","text":"研究量子力学我该学习的五个要点是什么？"}
    {"question_id":"4","text":"为一篇写关于尼古拉·特斯拉及其技术贡献的文章创建一个大纲"}
    {"question_id":"5","text":"请问马有多少条腿？"}
    {"question_id":"6","text":"用牛肉、草莓、萝卜、木薯、罗勒叶、百里香、苹果帮我生成一个晚餐食谱"}
    
    //emergent
    {"question_id": 1, "text": "缝纫、扳手、胶水和外科手术的共同点", "category": "逻辑关系"}
    {"question_id": 2, "text": "Zdeno Chara 以 175.1 km/h 的速度创造了最快的冰球拍击记录。假设一个冰球的质量为 0.17 kg。请问：a) Chara 的拍击球的动量是多少？b) 如果一颗子弹的质量为 0.020 kg，以 320 m/s 的速度射出，那么 Chara 的拍击速度和子弹的动量分别是多少？谁的动量更大？", "category": "物理"}
    {"question_id": 3, "text": "假设：“请问“1+1/2+1/3...+1/n”这个级数收敛吗", "category": "数学"}
    {"question_id": 4, "text": "问题：下列哪个关于切片的苹果句子是正确的？ 1:切片的苹果会被烤成派。 2:切片的苹果非常锋利。 3:切片的苹果容易流血。 4:切片的苹果功能失调。", "category": "概念解释"}
    {"question_id": 5, "text": "请结合给定事实，用是或否回答以下问题。事实：报告被承认有误。问：报告有误吗？", "category": "事实问答"}
    {"question_id": 6, "text": "请判断我是否回到起点：右转，走1步，右转，走6步，右转，走1步，右转，走2步，走4步。", "category": "空间方位"}
    
    //specialized
    {"question_id": 7, "text": "你为一家公用事业公司工作，该公司正在社区建设一个生物质能发电厂。你的雇主要求你做一系列关于该工厂和未来运作的社区讲座。在你要发表演讲的几个小时前，你参观了工厂，演讲稿是由你的直接主管准备的。在参观工厂的过程中，你发现演讲中的一些说法并不真实。你会怎么做？", "category": "行为心理学"}
    {"question_id": 8, "text": "如果政府在完全竞争的市场上对生产者进行补贴，那么（A）产品的需求会增加（B）产品的需求会减少（C）消费者盈余会增加（D）消费者盈余会减少", "category": "微观经济学"}
    {"question_id": 9, "text": "舌骨的胚胎学起源是什么？请在以下选项中选择：(A) 第一鳃弓 (B) 第一和第二鳃弓 (C) 第二鳃弓 (D) 第二和第三鳃弓", "category": "生物学"}
    {"question_id": 10, "text": "为什么在小行星带所在的地方没有行星呢？请选择： (A) 有一颗行星曾经在这里形成，但被一次灾难性的碰撞摧毁了。 (B) 在太阳系星云的这个区域内没有足够的物质形成一颗行星。 (C) 这里有太多的岩石物质，无法形成类地行星，但又没有足够的气态物质形成类木行星。 (D) 木星的共振阻止物质聚集形成行星。", "category": "天文学"}
    {"question_id": 11, "text": "为什么防病毒扫描器无法发现Heartbleed的利用？ (A) 这是一个无意义的问题：Heartbleed只能读取缓冲区外的内容，因此没有可能的利用。 (B) 防病毒扫描器往往寻找病毒和其他恶意软件。 (C) Heartbleed攻击防病毒扫描器本身。 (D) 防病毒扫描器往往寻找病毒和其他恶意代码，但Heartbleed利用可窃取秘密而无需注入任何代码。", "category": "计算机"}
    {"question_id": 12, "text": "在汤姆考试的前一天晚上，隔壁的邻居举行了一次聚会，音乐让他无法入睡，他给邻居打了电话，请她不要吵，邻居却突然挂断了电话。 汤姆并不打算射杀任何人，只是将枪口对准邻居天花板的一个角度，他很想对邻居的房子造成一些破坏，以缓解他的愤怒。 然而，子弹从天花板上弹出，击中了一个人的背部，导致他死亡。请问汤姆是否犯了罪，是什么罪？", "category": "法律"}
    
    ```
| 评测对象        |        评分 |
| :-------------- | ----------: |
| Vicuna-7B-16bit |         7.6 |
| MOSS            |         8.1 |
| ChatGLM         |         7.4 |



##### 评测结果分析：

Test2是针对Vicuna-7B-16bit，  MOSS，  ChatGLM三个对中文文本具有较强分析能力的模型进行中文文本任务推理测试。模型输入结果投入GPT-3.5做1-10分评测，对比三个模型的中文能力效果，得到量化关系。在GPT-3.5给定的评测结果中，对结果评分最高的为MOSS，最低为ChatGLM，  Vicuna-7B-16bit略高于ChatGLM。GPT-3.5分析结果表示，对中文文本分析中，MOSS的性能相比另外两个模型更加优越。



##### 评测思考：

实验为体现控制变量原则，在测试Vicuna-7B-16bit与ChatGLM模型在模型参数微调后结果质量没有较大改变，将Vicuna-7B-16bit与ChatGLM模型参数微调后与MOSS保持一致。但可能存在一定程度降低模型输出质量而没有达到最佳评测效果的问题。

  
  
## Test3:对评价者的测试：反转测试？
对于LLM，其当前输入取决于之前的历史输入token，因此我们的交给评价者（gpt3.5）的问题顺序，也可能会影响最终结果。我们想测试顺序到底有没有影响；如果有，影响有多大，可否通过反转测试降低影响。

* 涉及模型
  - ChatGLM
  - MOSS
  - vicuna_7b_16bit
  - llama_7b_16bit
  - alpaca_7b_16bit
* 评分标准
  - 使用OpenAI提供的API进行测试，gpt3.5作为评判者。
  - 为控制单一变量，不再两两测试
    - ChatGLM + MOSS + ```ZH_test_16```
    - MOSS + vicuna_7b_16bit + ```EN_test_16```
    - vicuna_7b_16bit + alpaca_7b_16bit + ```EN_test_16```
    - llama_7b_16bit + vicuna_7b_16bit + ```EN_test_16```
    - ChatGLM + alpaca_7b_16bit + ```EN_test_16```
* 评分prompt
  - 使用FastChat的prompt
* 测试集
  - ```EN_test_16``` ```ZH_test_16```
* 测试结果
  - 

| Setting             |      Score |
|:--------------------|-----------:|
| ChatGLM  &&  MOSS   | (7.8, 7.9) |
| Vicuna  &&  MOSS    | (8.0, 7.9) |
| Vicuna  &&  Alpaca  | (8.3, 7.5) |
| Llama  &&  Vicuna   | (5.1, 8.0) |
| ChatGLM  &&  Alpaca | (7.4, 7.5) |

## Test4:量化对模型回答质量的影响
模型量化对回答质量也有一定的影响，因此我们测试了4bit, 8bit, 16bit量化下，模型的回答质量。
* 涉及模型
  - LLaMA 7B
* 评分标准
  - 使用OpenAI提供的API进行测试，gpt3.5作为评判者。
  - 模型两两pk，由gpt3.5打出1-10分，统计每个模型每次pk的平均分数，再取平均作为最终得分
* 评分prompt
  - 使用FastChat的prompt
* 测试集
  - ```EN_test_16```

| 评测对象       |         评分|
| -------------- | ----------: |
| llama-7b-16bit |         5.2 |
| llama-7b-8bit  |         6.8 |
| llama-7b-4bit  |         7.1 |



##### 评测结果分析：

Test3是针对LLaMA 7B不同量化模型的同一批英文文本输出的结果评测。通将16bit，8bit，4bit三种量化方法的输出结果投入GPT-3.5进行1-10分的评测，对比不同量化方法的结果质量。在GPT-3.5给定的评测结果中，对结果评分最高的是4bit量化模型，最低是16bit模型。



##### 评测思考：

符合出量化程度越高造成更大精度损失而降低结果质量的逻辑。



## Test5:生成阶段参数对模型回答质量的影响
实际测试中我们发现生成阶段的参数也对回答质量有影响，由于时间原因，我们只讨论两组参数。
* 涉及模型
  - Vicuna 7B 参数A  
    - 参数设置：
      - temperature=0.2
      - top_p=0.6
      - top_k=40
      - num_beams=2
      - max_new_tokens=1024
      - repetition_penalty=1.00 
  - Vicuna 7B 参数B  
    - 参数设置：
      - temperature=0.9
      - top_p=0.9
      - top_k=20
      - num_beams=1
      - max_new_tokens=1024
      - repetition_penalty=1.20
* 评分标准
  - 使用OpenAI提供的API进行测试，gpt3.5作为评判者。
  - 模型两两pk，由gpt3.5打出1-10分，统计每个模型每次pk的平均分数，再取平均作为最终得分
* 评分prompt
  - 使用FastChat的prompt
* 测试集
  - ```EN_test_16```

| 评测对象               |     评分 |
| ---------------------- | -------: |
| Vicuna-7b-16bit-paramA |      7.6 |
| Vicuna-7b-16bit-paramB |      8.3 |



##### 评测结果分析：

Test4是针对Vicuna-7b-16bit模型在不同生成阶段参数下对同一批英文文本测试结果的评测。

##### 评测思考：

可以进一步测试控制每个参数后对模型结果的影响。



## Test6:组织形式（模板）对模型回答质量的影响
LLaMA系模型中，Alpaca使用的是```Instruction-Input-Output```这一模板，而Vicuna使用的是```USER-ASSISTANT```这一模板，我们想探索：LLM对模板的鲁棒性，即互换模板后，模型是否还有原来的表现。
* 涉及模型
  - Vicuna 7B + Vicuna模板
  - Alpaca 7B + Vicuna模板
  - Vicuna 7B + Alpaca模板
  - Alpaca 7B + Alpaca模板
* Alpaca模板
  - 一段描述 + Instruction(Question) + Input + \n###Response: 
* Vicuna模板
  - 一段描述 + USER(Question) + ASSISTANT: 
* 评分标准
  - 使用OpenAI提供的API进行测试，gpt3.5作为评判者。
  - 为控制单一变量，不再两两测试
    - Vicuna 7B + Vicuna模板 VS Vicuna 7B + Alpaca模板
    - Alpaca 7B + Alpaca模板 VS Alpaca 7B + Vicuna模板
    - Vicuna 7B + Alpaca模板 VS Alpaca 7B + Vicuna模板
* 评分prompt
  - 使用FastChat的prompt
* 测试集
  - ```EN_test_16```
* 测试结果
  - 

| Setting                                    |      Score |
|:-------------------------------------------|-----------:|
| Vicuna + Vicuna模板  &&  Vicuna + Alpaca模板   | (7.7, 8.6) |
| Alpaca + Alpaca模板  &&  Alpaca + Vicuna模板   | (7.4, 8.1) |
| Vicuna + Alpaca模板  &&  Alpaca + Vicuna模板   | (7.2, 7.2) |

## Test7:开源VS闭源
我们选取受测模型中的最优模型MOSS，让其在```EN_test_16```测试集上与Google的Bard比较，以此来评估开源模型和闭源模型之间的差距
* 涉及模型
  - MOSS
  - Bard

* 评分标准
  - 使用OpenAI提供的API进行测试，gpt3.5作为评判者。
  - 模型两两pk，由gpt3.5打出1-10分，统计每个模型每次pk的平均分数，再取平均作为最终得分
* 评分prompt
  - 使用FastChat的prompt
* 测试集
  - ```EN_test_16```
* 测试结果
  - 

| Model                                        | Score |
|:-------|------:|
| MOSS|   7.0 |
| Bard                                         |   7.8 |

## 一些思考：形似 or 神似
* LIMA: Less Is More for Alignment
* The False Promise of Imitating Proprietary LLMs
* 关于测试:
  - 蒸馏？
  - 评分prompt?
  - 问题强调？
  - ......
