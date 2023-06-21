## Gradio并行推理

https://gradio.app/docs/#interface-launch-header
https://gradio.app/key-features/#queuing

# gradio.Blocks.queue(···)

https://gradio.app/docs/#interface-launch-header

1. **concurrency_count**

   Number of worker threads that will be processing requests from the queue concurrently. Increasing this number will increase the rate at which requests are processed, but will also increase the memory usage of the queue.

   ```python
   with gr.Blocks() as demo:
       button = gr.Button(label="Generate Image")
       button.click(fn=image_generator, inputs=gr.Textbox(), outputs=gr.Image())
   
   demo.queue(concurrency_count=3)
   
   demo.launch()
   
   ```
   
# gradio.Interface.launch(···)

https://github.com/microsoft/DeepSpeedExamples

1. **max_threads**

   the maximum number of total threads that the Gradio app can generate in parallel. The default is inherited from the starlette library (currently 40). Applies whether the queue is enabled or not. But if queuing is enabled, this parameter is increaseed to be at least the concurrency_count of the queue.

   ```python

   import gradio as gr
   def reverse(text):
       return text[::-1]
   demo = gr.Interface(reverse, "text", "text")
   demo.launch(max_threads=80))
   ```

# gradio.Interface(···)

https://gradio.app/key-features/#queuing

1. **Batch Functions**
   
   The advantage of using batched functions is that if you enable queuing, the Gradio server can automatically batch incoming requests and process them in parallel, potentially speeding up your demo. Here's what the Gradio code looks like (notice the batch=True and max_batch_size=16 -- both of these parameters can be passed into event triggers or into the Interface class)

   ```python
   import time

   def trim_words(words, lens):
       trimmed_words = []
       time.sleep(5)
       for w, l in zip(words, lens):
           trimmed_words.append(w[:int(l)])        
       return [trimmed_words]
   
   
   
   demo = gr.Interface(trim_words, ["textbox", "number"], ["output"], 
                    batch=True, max_batch_size=16)
   demo.queue()
   demo.launch()
   ```

2. **Batch 报错**
   
   使用alpaca-lora的```generate.py```测试上述功能时，遇到了如下问题

   ```shell
   raise ValueError("Gradio does not support generators in batch mode.")
   ```

3. **gardio支持batch**
   
   3.14.0版本没找到这个接口
   
   https://github.com/gradio-app/gradio/pull/2218   

   ```shell
   The .queue() will have the following parameters:
   
   batch_size (by default None)
   batch_timeout (by default 1 seconds)
   The event handlers will have the following parameters:
   
   batch_fn
   Eventually, we may add a batch_size that overrides the global batch_size in queue()
   Eventually, we may add a batch_timeout that overrides the global batch_timeout in queue()
   Eventually, we may add a boolean force_batch_size that manually increases a smaller batch size to a larger one by repeating input samples.
   ```
   
# Int8量化时bnb会报错

   报错示例
   
   ```shell
    File "xxxxx/miniconda3/envs/petals/lib/python3.10/site-packages/bitsandbytes/autograd/_functions.py", line 397, in forward
    output += torch.matmul(subA, state.subB)
    RuntimeError: mat1 and mat2 shapes cannot be multiplied (550x11 and 10x4096)
   ```
   issues:
   1. https://github.com/TimDettmers/bitsandbytes/issues/162
   2. https://github.com/h2oai/h2ogpt/issues/104

# 测试结果
   1. 虽然设置了```concurrency```的相关参数，但没有用```Batching```，请求处理速度实际上不会比顺序执行快多少。
   2. ```concurrency_count = 20; max_threads = 8 ```的设置下
      1. 同时发送10个请求，大概需要80s全部执行完
      2. 顺序执行大概是15s/query
      3. 显存占用25000MiB
      4. GPU没有完全被利用
     
## API并发简单测试

#### FastChat API 容器错误

ConnectionError: HTTPConnectionPool(host='localhost', port=21001): Max retries exceeded with url: /register_worker (Caused by...

model workers 与 serve无法通讯，Issues已有相同问题但尚未解决

尝试解决：

1. 换用端口
2. localhost->0.0.0.0
3. 减小limit-model-concurrency(默认5)



#### chatGLM API

参数信息：

- max_length=2048
- top_p=0.7
- temperature=0.95


测试1：进行限时异步请求相应，加入任务分割并合并为一个prompt推理，与分词处理比对


| 单任务无并发平均时间 | 双任务并发平均时间 |
| :------------------: | ------------------ |
|          7s          | 12s                |

测试样例：

1. 如何与新同学相处？
2. 如何做一道宫保鸡丁？

回答

1.与新同学和谐相处需要一些时间和努力。以下是一些建议: \n\n1.尊重对方:尊重新同学,尊重他们的观点和信仰。不要试图打断他们说话或让他们感到不舒服。\n\n2.保持礼貌:与新同学和谐相处需要使用适当的礼貌和语言。尽量避免使用粗俗或冒犯性的语言。\n\n3.互相合作:与新同学和谐相处需要互相合作。尝试与同学建立积极的互动关系，并一起完成任务。\n\n4.倾听：倾听是建立良好关系的关键。尝试倾听新同学说话,并理解他们的观点。\n\n5.显示耐心:与新同学和谐相处需要耐心。可能需要一些时间来适应新的环境,建立新的人际关系。\n\n6.避免冲突:尽量避免冲突。如果出现问题,尝试通过沟通解决。\n\n7.接受不同:与新同学和谐相处需要接受不同的观点和文化。尝试理解并尊重他们的差异。\n\n最重要的是,与新同学和谐相处需要尊重、理解和耐心。如果能够遵循这些建议，那么与新同学和谐相处 will be a breeze!

2. 做一道宫保鸡丁的步骤如下：\n\n材料：\n\n-鸡胸肉 200 克\n-干辣椒 10 个\n-花生 100 克\n-葱姜蒜适量\n料酒、生抽、盐、糖、鸡精适量\n\n步骤：\n\n1.将鸡胸肉切成小块,用盐、料酒和鸡精腌制10分钟。\n\n2.干辣椒剪成段,去掉籽，葱姜蒜切末备用。\n\n3.热锅凉油，将腌制好的鸡肉块放入煸炒至变色，盛出备用。\n\n4.锅中留底油，放入葱姜蒜末煸炒出香味,加入干辣椒煸炒至香味出来。\n\n5.加入鸡肉块、花生翻炒均匀，加入适量的生抽、料酒、糖和鸡精，翻炒均匀。\n\n6.汤汁收浓后，起锅即可。\n\n宫保鸡丁的关键在于干辣椒和花生的比例，干辣椒要煸炒出香味,花生要炒到酥脆。另外,鸡肉要腌制入味,炒制的时间也不能太长,否则会煮糊。","history":[["怎么做一道宫保鸡丁？？"，"做一道宫保鸡丁的步骤如下： \n\n材料： \n\n-鸡胸肉200克\n-干辣椒10个\n-花生100克\n-葱姜蒜适量\n料酒、生抽、盐、糖、鸡精适量\n\n步骤：\n\n1.将鸡胸肉切成小块,用盐、料酒和鸡精腌制10分钟。\n\n2.干辣椒剪成段,去掉籽，葱姜蒜切末备用。\n\n3.热锅凉油，将腌制好的鸡肉块放入煸炒至变色，盛出备用。\n\n4.锅中留底油，放入葱姜蒜末煸炒出香味,加入干辣椒煸炒至香味出来。\n\n5.加入鸡肉块、花生翻炒均匀，加入适量的生抽、料酒、糖和鸡精，翻炒均匀。\n\n6.汤汁收浓后,起锅即可。\n\n宫保鸡丁的关键在于干辣椒和花生的比例，干辣椒要煸炒出香味，花生要炒到酥脆。另外，鸡肉要腌制入味,炒制的时间也不能太长,否则会煮糊。

3. 与新同学和谐相处需要尊重、理解和信任。以下是一些建议: \n\n1.尊重:尊重他人的个性、观点和价值观。不要嘲笑、贬低或歧视他人。\n\n2.理解:尝试理解他人的立场和感受。听取他人的意见和建议,避免产生误解和冲突。\n\n3.信任:建立互相信任的关系。遵守承诺,尊重他人的隐私和机密信息,不轻易相信他人。\n\n下面是一道宫保鸡丁的做法: \n\n材料: \n-鸡肉(切成小块) 250克\n-干辣椒10个\n-花生米50克\n-葱姜蒜适量\n-酱油、料酒、糖、醋、盐、淀粉适量\n\n做法: \n1.鸡肉用盐、料酒、淀粉腌制10分钟。\n2.干辣椒剪成段,去掉籽,花生米炒,香备用。\n3,葱姜蒜切末备用。\n4.热锅凉油,油热后加入葱姜蒜爆香,再加入鸡肉煸炒至变色。\n5.加入干辣椒、花生米煸炒至香。\n6.加入适量的酱油、料酒、糖、醋,翻炒均匀。\n7.最后加入适量的淀粉水勾芡,翻炒均匀即可。\n\n希望这些建议能够帮助与新同学和谐相外，享受美好的校园生活。


测试评价：
在合并两个并发请求时，通过修改prompt提示模型分别回答问题，总体回答质量过关，但依然存在少量的答案交叉且由于max_size限制答案相较单独提问要短，但形成一个batch后的平均处理时长也有提速效果，为研究并发问题还需要进一步测试并发上限对答案质量的影响以及如何分list在一个batch中进行推理。
