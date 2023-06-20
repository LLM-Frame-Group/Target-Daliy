## 训练环境搭建——问题汇总

1. **Exec docker**

   Within the Docker container, we can only perform the following steps.

   ```shell
   sudo docker exec -it <Your Docker> bash
   
   ```
   
# Deepspeed

https://github.com/microsoft/DeepSpeedExamples

1. **Unable to pre-compile async_io**

   报错示例：

   ```shell
   AssertionError: Unable to pre-compile async_io
   DS_BUILD_OPS=1
   [WARNING] async_io requires the dev libaio .so object and headers but these were not found.
   [WARNING] If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found.
   [WARNING] One can disable async_io with DS_BUILD_AIO=0
   [ERROR] Unable to pre-compile async_io
   [end of output]
   
   note: This error originates from a subprocess, and is likely not a problem with pip.
   error: metadata-generation-failed
   
   × Encountered error while generating package metadata.
   ╰─> See above for output.
   
   note: This is an issue with the package mentioned above, not pip.
   hint: See above for details.

   ```
   解决方案1：[手动 build extension](https://www.deepspeed.ai/tutorials/advanced-install/) 
   ```shell
   DS_BUILD_OPS=1 pip install deepspeed
   ```
   解决方案2：[不预先编译OPS](https://github.com/microsoft/DeepSpeed/issues/3145) 
   ```shell
   set DS_BUILD_OPS=0   # for win - Set-Item Env:\DS_BUILD_OPS 0
   pip install deepspeed
   ```
   解决方案3：如果前两种方案都出现问题，而编译失败的ops是LLM训练用不到的，如SPARSE_ATTN，可以[不编译特定OPS](https://www.deepspeed.ai/tutorials/advanced-install/) 
   ```shell
   set DS_BUILD_OPS=1   # for win - Set-Item Env:\DS_BUILD_OPS 0
   set DS_BUILD_SPARSE_ATTN=0
   pip install deepspeed
   ```
   解决方案4：如果只有特定ops没有编译成功，如fused_adam，尝试pip from source
   ```shell
   git clone <deepspeed repo>
   DS_BUILD_FUSED_ADAM=1 pip install .
   ```
2. **Note**: 
   1. deepspeed > 0.9.3 不太稳定，建议用低版本
   2. deepspeed支持的lr_schedule比较少，如cosine，不支持，
   3. 资源不受限的情况下 deepspeed + zero stage3 速度不如 flash-attn + fsdp
   4. 资源受限的情况下 deepspeed + zero stage3 + offload 速度快于 flash-attn + fsdp(offload)


# FastChat (falsh attention)

https://github.com/lm-sys/FastChat

1. **Pip err ModuleNotFoundError: No module named 'torch'**
   
   解决方案1：
   ```shell
   pip install flash-attn --no-build-isolation
   ```
   解决方案2：
   尝试flash-attn<=1.0.4
2. **Functools Err**
   
   https://github.com/lm-sys/FastChat/issues/1056
   ```shell
   ImportError: cannot import name 'cache' from 'functools' (/usr/lib/python3.8/functools.py)
   ```
3. **保存时发生OOM**
   
   https://github.com/pytorch/pytorch/issues/98823
   ```shell
   FSDP state dict OOM during model saving
   ```
   解决方案
   ```shell
   FOR python3.10 and torch==2.0 BY CHANGE 
   /python3.10/site-packages/torch/distributed/fsdp/_state_dict_utils.py ON LINE 309 
   FROM state_dict[fqn] = state_dict[fqn].clone().detach() 
   TO state_dict[fqn] = state_dict[fqn].cpu().clone().detach() 
   ```

4. **Deepspeed & flash-attn**
   https://github.com/lm-sys/FastChat/pull/177
   https://github.com/HazyResearch/flash-attention/issues/108

5. **Note**
   1. 新版本fastchat弃用falsh-attn，改用了xformer，V100也可以支持了。
   2. 对于3090、4090等，falsh-attn 仅可以训练7b，13b及以上会提示head_num超出限制

# Qlora
https://github.com/artidoro/qlora

问题主要出在很多库都要用最新的

1. **bitsandbytes/libbitsandbytes_cpu.so: undefined symbol: cget_col_row_stats**
   
   解决方案1：
   ```shell
   reinstall cudatoolkit

   conda uninstall cudatoolkit -y
   conda install cudatoolkit -y
   ```

2. **Transformers 版本**

   ```shell
   pip install --force-reinstall git+https://github.com/huggingface/transformers.git@796162c51298547c357b20cc33d64cbcf77d0241
   ```
3. **保存出错**

   adapter权重文件大小只有433kb，peft的老问题。

   解决方案1：(适用于其他任何情况，但不适用qlora，因为qlora只有最新版本的库才支持)
   ```shell
   pip install peft==0.3.0
   ```
   解决方案2: (适用于Qlora)https://github.com/artidoro/qlora/issues/41

4. **Note**
   1. 速度太慢，尤其是多卡。
      1. A40*1: max_token=768; lora_r=32; all linear; 0.6% para trainable
         1. VMEM 48500MiB
         2. 62h
      2. A40*2: max_token=1024; lora_r=128; all linear; 2.3% para trainable
         1. VMEM 89500MiB
         2. 225h
   2. 单卡48g可以跑65B，但是参数太丐，没有太大用处。参数量增长带来的提升完全弥补不了推理时的额外开销。
   3. 比较理想的用法是单卡A100 40G 微调falcon 40b或llama 33b。速度仍然比较慢，480k数据，微调参数量4%左右，一个epoch跑一周



# lora

https://github.com/tloen/alpaca-lora

1. **保存出错**

   adapter权重文件大小只有433kb，peft的老问题。

   解决方案：
   ```shell
   pip install peft==0.3.0
   ```
   
2. **Note**
   1. 没有太多环境方面的问题。
   2. 默认设置使用的是```decapoda-research/llama-7b-hf```，任何时候都应该避免使用这个版本的llama xB (如果更新了另说)。
      1. 这个模分片比较小，小内存也方便加载
      2. tokenizer有问题，权重转换的时候好像也有问题
      3. 如果使用该base model并按默认设置运行，输出不能截断，且会出现重复
      4. 可以通过更改eos_token来修复

# 其他问题

1. **程序莫名其妙被killed**
   1. 模型分片太大，内存不足
   2. ```decapoda-research/llama-7b-hf```的分片(405M)，16G应该就能加载
   3. hf默认的10G分片，可能需要40G才能加载，80G稳定加载

2. **CUDA掩盖报错**
   
   报错示例：
   ```shell
      mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
   RuntimeError: CUDA error: device-side assert triggered
   CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
   For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
   ```
   解决方案：
   1. 看程序打印的报错信息 
   2. CUDA_LAUNCH_BLOCKING=1，重新运行看报错
   3. cpu调试，看真正的报错

3. **其他加速方案**
   1. torch compile
      1. torch>=2.0.0支持
      2. 把flash-attn编译到kernel
   2. https://github.com/princeton-nlp/MeZO
   3. https://github.com/OpenLMLab/LOMO


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




