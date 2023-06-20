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