### DeepSeek-R1蒸馏1.5B Qwen模型调用流程

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
```

```plaintext
/root/miniconda3/envs/R1/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
```

```python
# 输入模型下载地址
model_name = "./DeepSeek-R1-1.5B"
```

```python
# 实例化预训练模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

```python
# 创建消息
prompt = "你好，好久不见，请介绍下你自己。"
messages = [
    {"role": "system", "content": "你是一名助人为乐的助手。"},
    {"role": "user", "content": prompt}
]
```

```python
# 分词
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
```

```python
# 创建回复
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

```plaintext
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
```

```python
print(response)
```

```plaintext
<think>
好，用户说“你好，好久不见，请介绍下你自己。”我应该先感谢他，然后介绍自己。我需要保持礼貌和亲切，所以开头可以用“你好！很高兴见到你。”然后简要介绍一下自己，比如是人工智能助手，专注于帮助用户解决问题。这样既介绍了自己，又回应了用户的需求。

接着，我应该邀请他继续交流，用“如果需要更多帮助，欢迎随时提问。”这样可以保持对话的流畅性，让用户觉得我们愿意继续帮助。

整体语气要自然，不要太生硬，让用户感觉被重视。最后，检查一下有没有遗漏的部分，确保信息完整，没有多余的内容。

这样，整个回复就完成了，既回答了用户的问题，又表现出对他的关心和帮助意愿。
</think>

你好！很高兴见到你。我是人工智能助手，专注于帮助您解决问题。如果需要更多帮助，欢迎随时提问。
```

```python
```

```python
```

```python
```

```python
```

# 课程说明：

* 体验课内容节选自[《2025大模型Agent智能体开发实战》](https://whakv.xetslk.com/s/1zrFP8)完整版付费课程

  体验课时间有限，若想深度学习大模型技术，欢迎大家报名由我主讲的[《2025大模型Agent智能体开发实战》](https://whakv.xetslk.com/s/1zrFP8)：

![](images/image.png)

![](images/f339b04b7b20233dd1509c7fb36d5c0.png)

此外，公开课全部训练项目代码、数据、以及训练完的模型，已上传至课件网盘，联系⬆️助教即可领取。
