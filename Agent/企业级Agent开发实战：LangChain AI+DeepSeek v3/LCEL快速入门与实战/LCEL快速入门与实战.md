## 课程说明：

* 体验课时间有限，若想深度学习大模型技术，欢迎大家报名我主讲的[《2025大模型Agent智能体开发实战》](https://whakv.xetslk.com/s/3xFEAA)

**[《2025大模型Agent智能体开发实战》](https://whakv.xetslk.com/s/3xFEAA)为【100+小时】体系大课，总共20大模块精讲精析，零基础直达大模型企业级应用！**

![](images/image.png)

![](images/f339b04b7b20233dd1509c7fb36d5c0.png)

此外，公开课全部课件，以及项目代码、数据等学习资料，扫码⬆️联系助教即可领取～

***

# 《2025大模型智能体Agent开发实战》体验课

## LangChain AI+ DeepSeek v3 企业 Agent 开发实战

## Part 1. LCEL (LangChain Expression Language) 快速入门与实战

# 1. LCEL基本概念入门

  `LCEL(LangChain Expression Language)`是 `LangChain` 中非常关键的概念，它指的是一种声明式的表达式语言，用于通过链式组合不同的模块。它将不同的组件通过统一的`Runable`接口连接起来，从而实现具体的流程或功能（比如RAG、Agent）。

![](images/b852682c-c798-44f5-872b-164dfc71b819.png)

  因为每个模块都是`LCEL`对象，因此基于`LCEL`对象连接形成的链，本身也是一个 `LCEL` 对象，所以当通过一组通用的调用方法（invoke、 batch、stream、 ainvoke ）方法时，就能够做到定制化组合链、并行化组件、回退、动态配置链内部结构等标准化的操作。它的优势非常明显：

1. 统一的接口：每个 `LCEL` 对象都实现`Runnable`接口，因此可以非常方便的连接到一起；

2. 模块化操作：每个组件都可以独立开发和测试，处理好输入和输出就可以通过`LCEL`集成到一起；

3. 良好扩展性：各个模块组件之间都实现了通用的调用方法（invoke、 batch、stream、 ainvoke ）方法，因此可以灵活组合使用不同的使用场景；

  比如`LangChain`中抽象出来的最简单的 Model I/O 模块。

![](images/20400d87-8ce5-4e7f-8a6f-ca9d0e6968fd.png)

  LangChain的Model I/O模块提供了标准的、可扩展的接口实现与大语言模型的外部集成。所谓的Model I/O，包括模型输入（Prompts）、模型输出（OutPuts）和模型本身（Models），简单理解就是通过该模块，我们可以快速与某个大模型进行对话交互，整个内部逻辑就相当于我们最熟悉的这个过程：输入Prompt，得到大模型针对该Prompt的推理结果。如下示例为OpenAI的 GPT 系列模型的API 调用规范：

```python
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "请问，什么是机器学习？"}
  ]
)
```

  在LangChain的Model I/O模块设计中，包含三个核心部分： Prompt Template（对应上图中的Format部分）， Model（对应上图中的Predict部分） 和Output Parser（对应上图中的Parse部分）。

* **Format：即指代Prompts Template，通过模板化来管理大模型的输入；**

* **Predict：即指代Models，使用通用接口调用不同的大语言模型；**

* **Parse：即指代Output部分，用来从模型的推理中提取信息，并按照预先设定好的模版来规范化输出。**

* **Format**

  对于Prompt Template第一部分，传统上我们创建提示词是通过手工编写来实现的，在这个过程中会利用各种提示工程技巧，如Few-Shot、链式推理（CoT）等方法，以提高大模型的推理性能。然而，**在应用开发中，一个关键的考量是提示词不能是一成不变的。** 其原因在于，应用开发需要适应多变的用户需求和场景。固定的提示词限制了模型的灵活性和适用范围。例如，如果我们正在开发一个天气查询应用，用户可能会以多种方式提出查询，如“今天的天气怎么样？”或“明天纽约的温度是多少度？”。如果提示词是固定的，它可能只能处理一种特定类型的查询，而无法适应这种多样性的需求。

  而Prompt Template，就像ReAct模版，将API的使用、问题解答过程等复杂逻辑封装成了一套结构化的格式。我们只需准备具体的外部函数信息和用户查询，即可生成定制化的提示词，引导模型按照既定逻辑进行思考和回答，从而实现外部函数的调用过程，即：

```json
# 将一个插件的关键信息拼接成一段文本的模版。
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

# ReAct prompting 的 instruction 模版，将包含插件的详细信息。
PROMPT_REACT = """Answer the following questions as best you can. You have access to the following APIs:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""
```

  因此，引入Prompt Template可以支持变量和动态内容的插入，使得同一个应用可以根据不同的输入动态调整提示词，从而更好地响应用户的具体需求。LangChain通过这种方式来提高应用的通用性和用户体验。

* **Predict**

  在Predict部分，实质上是处理模型从接收输入到执行推理的整个过程。考虑到存在两种主要类型的大模型——Base类模型和Chat类模型，LangChain在其Model I/O模块中对这两种模型都进行了抽象，分别归类为LLMs（Large Language Models）和Chat Models。我们还是以OpenAI 的 Completion 和 Chatcompletions方法为例：

```python

# Base类模型
client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt="Say this is a test",
)


# 聊天模型
client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "你是一位乐于助人的AI智能小助手"},
    {"role": "user", "content": "你好，请你介绍一下你自己。"}
  ]
)
```

  LLMs是简化的大语言模型抽象，即基于给定的Prompt提供内容生成的功能。而Chat Models则专注于聊天API的抽象，需要维护上下文的记忆（聊天记录），呈现出更接近对话或聊天形式的交互。

* **Parse**

  我们知道，大模型的输出是不稳定的，同样的输入Prompt往往会得到不同形式的输出。在自然语言交互中，不同的语言表达方式通常不会造成理解上的障碍。但在应用开发中，大模型的输出可能是下一步逻辑处理的关键输入。因此，在这种情况下，规范化输出是必须要做的任务，以确保应用能够顺利进行后续的逻辑处理。

  输出解析器 Output Parser就是一个帮助结构化语言模型响应的抽象，可以获取格式指令或者进行更深层次的解析。这我们会在后面的实践中直观的体验到。

  整体而言，在Model I/O模块的抽象中，其一能够让开发者快速的接入不同的大模型，比如OpenAI、ChatGLM、Qwen等，按照既定规范执行模型推理。其二通过输入和输出的模板化处理，使其更贴合于应用开发的最佳实践。接下来，我们就逐步的介绍上述三个流程在LangChain下是如何进行集成和操作的。

![](images/2cbe28d7-a393-43c6-b2f5-1a777b1c09da.png)

# 2. DeepSeek v3 模型注册与使用

* DeepSeek v3账号注册与API获取

  `DeepSeek`官网：https://www.deepseek.com/

![](images/5d3eef35-1a6c-4106-b7c6-a8d162409c0c.png)

  新用户注册即赠送10元额度，约500万token额度（如果发现没有赠送额度，那就是官方停止了活动。）

![](images/3076db91-e239-44ed-8880-be481d9ef9fd.png)

  对比`GPT4o`价格，约降低`90%`以上：输入价格为`GPT4o`的`6%`，输出价格为`GPT4o`的`3%`。

![](images/5a04b115-ca2e-42ff-961d-68d9d0a718fc.png)

&#x20;

![](images/814b74b3-4382-4ca4-94d0-6e064d2b0cde.png)

  而且其`API`调用不限速：

![](images/6ada8ed9-4ae7-4799-9996-13caededfbd4.png)

  最关键的是，调用风格和`OpenAI`完全一致：`Function calling`、`Json Output`等功能完全相同：

![](images/ee453526-e2f2-472c-bad9-b5333b51ab5e.png)

&#x20;

![](images/af9a80f3-97bd-4768-b1c9-f2a2b87e2764.png)

# 3. 基于LCEL 实现 DeepSeep v3 的集成

  `LangChain`中最基本和常见的用例是将提示模板和模型链接在一起。为了看看这是如何工作的，我们创建一个接受某个主题并生成响应的链：

```python
# ! pip install -qU langchain langchain-openai
```

```python
from langchain_openai import ChatOpenAI

# 使用DeepSeek-chat模型作为chatmodel
model = ChatOpenAI(model="deepseek-chat",
                   api_key='sk-6c960cb2a7',
                   base_url='https://api.deepseek.com')
```

  使用 `LCEL` 将这些不同的组件拼凑成一个链。其中 `|` 符号类似于`unix`管道运算符，它将不同的组件链接在一起，将一个组件的输出作为下一个组件的输入。

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("请你介绍一下什么是 {topic}")

chain = prompt | model | StrOutputParser()
```

  在此链中，用户输入传递到提示模板，然后提示模板输出传递到模型，然后模型输出传递到输出解析器。

```python
chain.invoke({"topic": "人工智能"})
```

```plaintext
'人工智能（Artificial Intelligence，简称 AI）是指通过计算机系统模拟、扩展和增强人类智能的技术和科学领域。它旨在使机器能够执行通常需要人类智能的任务，例如学习、推理、问题解决、感知、语言理解和决策等。\n\n### 人工智能的主要特点：\n1. **模拟人类智能**：AI 系统试图模仿人类的思维过程，例如学习、推理和决策。\n2. **自动化**：AI 可以自动执行任务，减少人类干预。\n3. **适应性和学习能力**：许多 AI 系统能够通过数据学习并改进性能，例如机器学习和深度学习。\n4. **处理复杂问题**：AI 可以处理大量数据并解决复杂问题，例如图像识别、自然语言处理和预测分析。\n\n### 人工智能的分类：\n1. **弱人工智能（Narrow AI）**：专注于特定任务，例如语音助手（如 Siri、Alexa）或图像识别系统。这是目前最常见的 AI 形式。\n2. **强人工智能（General AI）**：具备与人类相当的通用智能，能够理解、学习和执行任何智力任务。目前尚未实现。\n3. **超级人工智能（Superintelligent AI）**：超越人类智能的 AI，能够在几乎所有领域超越人类。这仍然是理论上的概念。\n\n### 人工智能的关键技术：\n1. **机器学习（Machine Learning）**：通过数据训练模型，使系统能够从经验中学习并改进性能。\n2. **深度学习（Deep Learning）**：一种机器学习方法，使用神经网络模拟人脑的工作方式，适用于图像识别、语音识别等任务。\n3. **自然语言处理（NLP）**：使机器能够理解、生成和处理人类语言，例如聊天机器人和翻译系统。\n4. **计算机视觉（Computer Vision）**：使机器能够“看”并理解图像和视频内容，例如人脸识别和自动驾驶。\n5. **强化学习（Reinforcement Learning）**：通过试错和奖励机制训练 AI 系统，使其在特定环境中做出最佳决策。\n\n### 人工智能的应用领域：\n- **医疗**：疾病诊断、药物研发、个性化治疗。\n- **金融**：风险评估、欺诈检测、自动化交易。\n- **交通**：自动驾驶、交通流量优化。\n- **教育**：个性化学习、智能辅导系统。\n- **娱乐**：游戏 AI、内容推荐系统。\n- **制造业**：自动化生产线、质量控制。\n\n### 人工智能的挑战与争议：\n1. **伦理问题**：AI 的决策可能涉及隐私、偏见和公平性等问题。\n2. **就业影响**：自动化可能取代某些工作岗位，引发社会经济问题。\n3. **安全性**：AI 系统可能被滥用或出现不可预测的行为。\n4. **技术限制**：目前的 AI 仍无法完全模拟人类的创造力和情感。\n\n总之，人工智能是一项快速发展的技术，正在深刻改变我们的生活和工作方式。尽管它带来了巨大的潜力，但也需要谨慎应对其带来的挑战。'
```

  在这个过程中，`Runnable` 会动态生成输入的描述，显示的将其传入`Pydantic` 模型。我们可以对其调用.schema()来获取 JSONSchema 表示。

```python
chain.input_schema.schema()
```

```plaintext
{'properties': {'topic': {'title': 'Topic', 'type': 'string'}},
 'required': ['topic'],
 'title': 'PromptInput',
 'type': 'object'}
```

```python
prompt.input_schema.schema()
```

```plaintext
{'properties': {'topic': {'title': 'Topic', 'type': 'string'}},
 'required': ['topic'],
 'title': 'PromptInput',
 'type': 'object'}
```

```python
prompt.output_schema.schema()
```

```plaintext
{'$defs': {'AIMessage': {'additionalProperties': True,
   'description': 'Message from an AI.\n\nAIMessage is returned from a chat model as a response to a prompt.\n\nThis message represents the output of the model and consists of both\nthe raw output as returned by the model together standardized fields\n(e.g., tool calls, usage metadata) added by the LangChain framework.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'ai',
     'default': 'ai',
     'enum': ['ai'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'example': {'default': False, 'title': 'Example', 'type': 'boolean'},
    'tool_calls': {'default': [],
     'items': {'$ref': '#/$defs/ToolCall'},
     'title': 'Tool Calls',
     'type': 'array'},
    'invalid_tool_calls': {'default': [],
     'items': {'$ref': '#/$defs/InvalidToolCall'},
     'title': 'Invalid Tool Calls',
     'type': 'array'},
    'usage_metadata': {'anyOf': [{'$ref': '#/$defs/UsageMetadata'},
      {'type': 'null'}],
     'default': None}},
   'required': ['content'],
   'title': 'AIMessage',
   'type': 'object'},
  'AIMessageChunk': {'additionalProperties': True,
   'description': 'Message chunk from an AI.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'AIMessageChunk',
     'default': 'AIMessageChunk',
     'enum': ['AIMessageChunk'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'example': {'default': False, 'title': 'Example', 'type': 'boolean'},
    'tool_calls': {'default': [],
     'items': {'$ref': '#/$defs/ToolCall'},
     'title': 'Tool Calls',
     'type': 'array'},
    'invalid_tool_calls': {'default': [],
     'items': {'$ref': '#/$defs/InvalidToolCall'},
     'title': 'Invalid Tool Calls',
     'type': 'array'},
    'usage_metadata': {'anyOf': [{'$ref': '#/$defs/UsageMetadata'},
      {'type': 'null'}],
     'default': None},
    'tool_call_chunks': {'default': [],
     'items': {'$ref': '#/$defs/ToolCallChunk'},
     'title': 'Tool Call Chunks',
     'type': 'array'}},
   'required': ['content'],
   'title': 'AIMessageChunk',
   'type': 'object'},
  'ChatMessage': {'additionalProperties': True,
   'description': 'Message that can be assigned an arbitrary speaker (i.e. role).',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'chat',
     'default': 'chat',
     'enum': ['chat'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'role': {'title': 'Role', 'type': 'string'}},
   'required': ['content', 'role'],
   'title': 'ChatMessage',
   'type': 'object'},
  'ChatMessageChunk': {'additionalProperties': True,
   'description': 'Chat Message chunk.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'ChatMessageChunk',
     'default': 'ChatMessageChunk',
     'enum': ['ChatMessageChunk'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'role': {'title': 'Role', 'type': 'string'}},
   'required': ['content', 'role'],
   'title': 'ChatMessageChunk',
   'type': 'object'},
  'ChatPromptValueConcrete': {'description': 'Chat prompt value which explicitly lists out the message types it accepts.\nFor use in external schemas.',
   'properties': {'messages': {'items': {'oneOf': [{'$ref': '#/$defs/AIMessage'},
       {'$ref': '#/$defs/HumanMessage'},
       {'$ref': '#/$defs/ChatMessage'},
       {'$ref': '#/$defs/SystemMessage'},
       {'$ref': '#/$defs/FunctionMessage'},
       {'$ref': '#/$defs/ToolMessage'},
       {'$ref': '#/$defs/AIMessageChunk'},
       {'$ref': '#/$defs/HumanMessageChunk'},
       {'$ref': '#/$defs/ChatMessageChunk'},
       {'$ref': '#/$defs/SystemMessageChunk'},
       {'$ref': '#/$defs/FunctionMessageChunk'},
       {'$ref': '#/$defs/ToolMessageChunk'}]},
     'title': 'Messages',
     'type': 'array'},
    'type': {'const': 'ChatPromptValueConcrete',
     'default': 'ChatPromptValueConcrete',
     'enum': ['ChatPromptValueConcrete'],
     'title': 'Type',
     'type': 'string'}},
   'required': ['messages'],
   'title': 'ChatPromptValueConcrete',
   'type': 'object'},
  'FunctionMessage': {'additionalProperties': True,
   'description': 'Message for passing the result of executing a tool back to a model.\n\nFunctionMessage are an older version of the ToolMessage schema, and\ndo not contain the tool_call_id field.\n\nThe tool_call_id field is used to associate the tool call request with the\ntool call response. This is useful in situations where a chat model is able\nto request multiple tool calls in parallel.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'function',
     'default': 'function',
     'enum': ['function'],
     'title': 'Type',
     'type': 'string'},
    'name': {'title': 'Name', 'type': 'string'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'}},
   'required': ['content', 'name'],
   'title': 'FunctionMessage',
   'type': 'object'},
  'FunctionMessageChunk': {'additionalProperties': True,
   'description': 'Function Message chunk.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'FunctionMessageChunk',
     'default': 'FunctionMessageChunk',
     'enum': ['FunctionMessageChunk'],
     'title': 'Type',
     'type': 'string'},
    'name': {'title': 'Name', 'type': 'string'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'}},
   'required': ['content', 'name'],
   'title': 'FunctionMessageChunk',
   'type': 'object'},
  'HumanMessage': {'additionalProperties': True,
   'description': 'Message from a human.\n\nHumanMessages are messages that are passed in from a human to the model.\n\nExample:\n\n    .. code-block:: python\n\n        from langchain_core.messages import HumanMessage, SystemMessage\n\n        messages = [\n            SystemMessage(\n                content="You are a helpful assistant! Your name is Bob."\n            ),\n            HumanMessage(\n                content="What is your name?"\n            )\n        ]\n\n        # Instantiate a chat model and invoke it with the messages\n        model = ...\n        print(model.invoke(messages))',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'human',
     'default': 'human',
     'enum': ['human'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'example': {'default': False, 'title': 'Example', 'type': 'boolean'}},
   'required': ['content'],
   'title': 'HumanMessage',
   'type': 'object'},
  'HumanMessageChunk': {'additionalProperties': True,
   'description': 'Human Message chunk.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'HumanMessageChunk',
     'default': 'HumanMessageChunk',
     'enum': ['HumanMessageChunk'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'example': {'default': False, 'title': 'Example', 'type': 'boolean'}},
   'required': ['content'],
   'title': 'HumanMessageChunk',
   'type': 'object'},
  'InputTokenDetails': {'description': 'Breakdown of input token counts.\n\nDoes *not* need to sum to full input token count. Does *not* need to have all keys.\n\nExample:\n\n    .. code-block:: python\n\n        {\n            "audio": 10,\n            "cache_creation": 200,\n            "cache_read": 100,\n        }\n\n.. versionadded:: 0.3.9',
   'properties': {'audio': {'title': 'Audio', 'type': 'integer'},
    'cache_creation': {'title': 'Cache Creation', 'type': 'integer'},
    'cache_read': {'title': 'Cache Read', 'type': 'integer'}},
   'title': 'InputTokenDetails',
   'type': 'object'},
  'InvalidToolCall': {'description': 'Allowance for errors made by LLM.\n\nHere we add an `error` key to surface errors made during generation\n(e.g., invalid JSON arguments.)',
   'properties': {'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'title': 'Name'},
    'args': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Args'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
    'error': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'title': 'Error'},
    'type': {'const': 'invalid_tool_call',
     'enum': ['invalid_tool_call'],
     'title': 'Type',
     'type': 'string'}},
   'required': ['name', 'args', 'id', 'error'],
   'title': 'InvalidToolCall',
   'type': 'object'},
  'OutputTokenDetails': {'description': 'Breakdown of output token counts.\n\nDoes *not* need to sum to full output token count. Does *not* need to have all keys.\n\nExample:\n\n    .. code-block:: python\n\n        {\n            "audio": 10,\n            "reasoning": 200,\n        }\n\n.. versionadded:: 0.3.9',
   'properties': {'audio': {'title': 'Audio', 'type': 'integer'},
    'reasoning': {'title': 'Reasoning', 'type': 'integer'}},
   'title': 'OutputTokenDetails',
   'type': 'object'},
  'StringPromptValue': {'description': 'String prompt value.',
   'properties': {'text': {'title': 'Text', 'type': 'string'},
    'type': {'const': 'StringPromptValue',
     'default': 'StringPromptValue',
     'enum': ['StringPromptValue'],
     'title': 'Type',
     'type': 'string'}},
   'required': ['text'],
   'title': 'StringPromptValue',
   'type': 'object'},
  'SystemMessage': {'additionalProperties': True,
   'description': 'Message for priming AI behavior.\n\nThe system message is usually passed in as the first of a sequence\nof input messages.\n\nExample:\n\n    .. code-block:: python\n\n        from langchain_core.messages import HumanMessage, SystemMessage\n\n        messages = [\n            SystemMessage(\n                content="You are a helpful assistant! Your name is Bob."\n            ),\n            HumanMessage(\n                content="What is your name?"\n            )\n        ]\n\n        # Define a chat model and invoke it with the messages\n        print(model.invoke(messages))',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'system',
     'default': 'system',
     'enum': ['system'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'}},
   'required': ['content'],
   'title': 'SystemMessage',
   'type': 'object'},
  'SystemMessageChunk': {'additionalProperties': True,
   'description': 'System Message chunk.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'SystemMessageChunk',
     'default': 'SystemMessageChunk',
     'enum': ['SystemMessageChunk'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'}},
   'required': ['content'],
   'title': 'SystemMessageChunk',
   'type': 'object'},
  'ToolCall': {'description': 'Represents a request to call a tool.\n\nExample:\n\n    .. code-block:: python\n\n        {\n            "name": "foo",\n            "args": {"a": 1},\n            "id": "123"\n        }\n\n    This represents a request to call the tool named "foo" with arguments {"a": 1}\n    and an identifier of "123".',
   'properties': {'name': {'title': 'Name', 'type': 'string'},
    'args': {'title': 'Args', 'type': 'object'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
    'type': {'const': 'tool_call',
     'enum': ['tool_call'],
     'title': 'Type',
     'type': 'string'}},
   'required': ['name', 'args', 'id'],
   'title': 'ToolCall',
   'type': 'object'},
  'ToolCallChunk': {'description': 'A chunk of a tool call (e.g., as part of a stream).\n\nWhen merging ToolCallChunks (e.g., via AIMessageChunk.__add__),\nall string attributes are concatenated. Chunks are only merged if their\nvalues of `index` are equal and not None.\n\nExample:\n\n.. code-block:: python\n\n    left_chunks = [ToolCallChunk(name="foo", args=\'{"a":\', index=0)]\n    right_chunks = [ToolCallChunk(name=None, args=\'1}\', index=0)]\n\n    (\n        AIMessageChunk(content="", tool_call_chunks=left_chunks)\n        + AIMessageChunk(content="", tool_call_chunks=right_chunks)\n    ).tool_call_chunks == [ToolCallChunk(name=\'foo\', args=\'{"a":1}\', index=0)]',
   'properties': {'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'title': 'Name'},
    'args': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Args'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
    'index': {'anyOf': [{'type': 'integer'}, {'type': 'null'}],
     'title': 'Index'},
    'type': {'const': 'tool_call_chunk',
     'enum': ['tool_call_chunk'],
     'title': 'Type',
     'type': 'string'}},
   'required': ['name', 'args', 'id', 'index'],
   'title': 'ToolCallChunk',
   'type': 'object'},
  'ToolMessage': {'additionalProperties': True,
   'description': 'Message for passing the result of executing a tool back to a model.\n\nToolMessages contain the result of a tool invocation. Typically, the result\nis encoded inside the `content` field.\n\nExample: A ToolMessage representing a result of 42 from a tool call with id\n\n    .. code-block:: python\n\n        from langchain_core.messages import ToolMessage\n\n        ToolMessage(content=\'42\', tool_call_id=\'call_Jja7J89XsjrOLA5r!MEOW!SL\')\n\n\nExample: A ToolMessage where only part of the tool output is sent to the model\n    and the full output is passed in to artifact.\n\n    .. versionadded:: 0.2.17\n\n    .. code-block:: python\n\n        from langchain_core.messages import ToolMessage\n\n        tool_output = {\n            "stdout": "From the graph we can see that the correlation between x and y is ...",\n            "stderr": None,\n            "artifacts": {"type": "image", "base64_data": "/9j/4gIcSU..."},\n        }\n\n        ToolMessage(\n            content=tool_output["stdout"],\n            artifact=tool_output,\n            tool_call_id=\'call_Jja7J89XsjrOLA5r!MEOW!SL\',\n        )\n\nThe tool_call_id field is used to associate the tool call request with the\ntool call response. This is useful in situations where a chat model is able\nto request multiple tool calls in parallel.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'tool',
     'default': 'tool',
     'enum': ['tool'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'tool_call_id': {'title': 'Tool Call Id', 'type': 'string'},
    'artifact': {'default': None, 'title': 'Artifact'},
    'status': {'default': 'success',
     'enum': ['success', 'error'],
     'title': 'Status',
     'type': 'string'}},
   'required': ['content', 'tool_call_id'],
   'title': 'ToolMessage',
   'type': 'object'},
  'ToolMessageChunk': {'additionalProperties': True,
   'description': 'Tool Message chunk.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'ToolMessageChunk',
     'default': 'ToolMessageChunk',
     'enum': ['ToolMessageChunk'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'tool_call_id': {'title': 'Tool Call Id', 'type': 'string'},
    'artifact': {'default': None, 'title': 'Artifact'},
    'status': {'default': 'success',
     'enum': ['success', 'error'],
     'title': 'Status',
     'type': 'string'}},
   'required': ['content', 'tool_call_id'],
   'title': 'ToolMessageChunk',
   'type': 'object'},
  'UsageMetadata': {'description': 'Usage metadata for a message, such as token counts.\n\nThis is a standard representation of token usage that is consistent across models.\n\nExample:\n\n    .. code-block:: python\n\n        {\n            "input_tokens": 350,\n            "output_tokens": 240,\n            "total_tokens": 590,\n            "input_token_details": {\n                "audio": 10,\n                "cache_creation": 200,\n                "cache_read": 100,\n            },\n            "output_token_details": {\n                "audio": 10,\n                "reasoning": 200,\n            }\n        }\n\n.. versionchanged:: 0.3.9\n\n    Added ``input_token_details`` and ``output_token_details``.',
   'properties': {'input_tokens': {'title': 'Input Tokens', 'type': 'integer'},
    'output_tokens': {'title': 'Output Tokens', 'type': 'integer'},
    'total_tokens': {'title': 'Total Tokens', 'type': 'integer'},
    'input_token_details': {'$ref': '#/$defs/InputTokenDetails'},
    'output_token_details': {'$ref': '#/$defs/OutputTokenDetails'}},
   'required': ['input_tokens', 'output_tokens', 'total_tokens'],
   'title': 'UsageMetadata',
   'type': 'object'}},
 'anyOf': [{'$ref': '#/$defs/StringPromptValue'},
  {'$ref': '#/$defs/ChatPromptValueConcrete'}],
 'title': 'ChatPromptTemplateOutput'}
```

```python
model.input_schema.schema()
```

```plaintext
{'$defs': {'AIMessage': {'additionalProperties': True,
   'description': 'Message from an AI.\n\nAIMessage is returned from a chat model as a response to a prompt.\n\nThis message represents the output of the model and consists of both\nthe raw output as returned by the model together standardized fields\n(e.g., tool calls, usage metadata) added by the LangChain framework.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'ai',
     'default': 'ai',
     'enum': ['ai'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'example': {'default': False, 'title': 'Example', 'type': 'boolean'},
    'tool_calls': {'default': [],
     'items': {'$ref': '#/$defs/ToolCall'},
     'title': 'Tool Calls',
     'type': 'array'},
    'invalid_tool_calls': {'default': [],
     'items': {'$ref': '#/$defs/InvalidToolCall'},
     'title': 'Invalid Tool Calls',
     'type': 'array'},
    'usage_metadata': {'anyOf': [{'$ref': '#/$defs/UsageMetadata'},
      {'type': 'null'}],
     'default': None}},
   'required': ['content'],
   'title': 'AIMessage',
   'type': 'object'},
  'AIMessageChunk': {'additionalProperties': True,
   'description': 'Message chunk from an AI.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'AIMessageChunk',
     'default': 'AIMessageChunk',
     'enum': ['AIMessageChunk'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'example': {'default': False, 'title': 'Example', 'type': 'boolean'},
    'tool_calls': {'default': [],
     'items': {'$ref': '#/$defs/ToolCall'},
     'title': 'Tool Calls',
     'type': 'array'},
    'invalid_tool_calls': {'default': [],
     'items': {'$ref': '#/$defs/InvalidToolCall'},
     'title': 'Invalid Tool Calls',
     'type': 'array'},
    'usage_metadata': {'anyOf': [{'$ref': '#/$defs/UsageMetadata'},
      {'type': 'null'}],
     'default': None},
    'tool_call_chunks': {'default': [],
     'items': {'$ref': '#/$defs/ToolCallChunk'},
     'title': 'Tool Call Chunks',
     'type': 'array'}},
   'required': ['content'],
   'title': 'AIMessageChunk',
   'type': 'object'},
  'ChatMessage': {'additionalProperties': True,
   'description': 'Message that can be assigned an arbitrary speaker (i.e. role).',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'chat',
     'default': 'chat',
     'enum': ['chat'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'role': {'title': 'Role', 'type': 'string'}},
   'required': ['content', 'role'],
   'title': 'ChatMessage',
   'type': 'object'},
  'ChatMessageChunk': {'additionalProperties': True,
   'description': 'Chat Message chunk.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'ChatMessageChunk',
     'default': 'ChatMessageChunk',
     'enum': ['ChatMessageChunk'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'role': {'title': 'Role', 'type': 'string'}},
   'required': ['content', 'role'],
   'title': 'ChatMessageChunk',
   'type': 'object'},
  'ChatPromptValueConcrete': {'description': 'Chat prompt value which explicitly lists out the message types it accepts.\nFor use in external schemas.',
   'properties': {'messages': {'items': {'oneOf': [{'$ref': '#/$defs/AIMessage'},
       {'$ref': '#/$defs/HumanMessage'},
       {'$ref': '#/$defs/ChatMessage'},
       {'$ref': '#/$defs/SystemMessage'},
       {'$ref': '#/$defs/FunctionMessage'},
       {'$ref': '#/$defs/ToolMessage'},
       {'$ref': '#/$defs/AIMessageChunk'},
       {'$ref': '#/$defs/HumanMessageChunk'},
       {'$ref': '#/$defs/ChatMessageChunk'},
       {'$ref': '#/$defs/SystemMessageChunk'},
       {'$ref': '#/$defs/FunctionMessageChunk'},
       {'$ref': '#/$defs/ToolMessageChunk'}]},
     'title': 'Messages',
     'type': 'array'},
    'type': {'const': 'ChatPromptValueConcrete',
     'default': 'ChatPromptValueConcrete',
     'enum': ['ChatPromptValueConcrete'],
     'title': 'Type',
     'type': 'string'}},
   'required': ['messages'],
   'title': 'ChatPromptValueConcrete',
   'type': 'object'},
  'FunctionMessage': {'additionalProperties': True,
   'description': 'Message for passing the result of executing a tool back to a model.\n\nFunctionMessage are an older version of the ToolMessage schema, and\ndo not contain the tool_call_id field.\n\nThe tool_call_id field is used to associate the tool call request with the\ntool call response. This is useful in situations where a chat model is able\nto request multiple tool calls in parallel.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'function',
     'default': 'function',
     'enum': ['function'],
     'title': 'Type',
     'type': 'string'},
    'name': {'title': 'Name', 'type': 'string'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'}},
   'required': ['content', 'name'],
   'title': 'FunctionMessage',
   'type': 'object'},
  'FunctionMessageChunk': {'additionalProperties': True,
   'description': 'Function Message chunk.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'FunctionMessageChunk',
     'default': 'FunctionMessageChunk',
     'enum': ['FunctionMessageChunk'],
     'title': 'Type',
     'type': 'string'},
    'name': {'title': 'Name', 'type': 'string'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'}},
   'required': ['content', 'name'],
   'title': 'FunctionMessageChunk',
   'type': 'object'},
  'HumanMessage': {'additionalProperties': True,
   'description': 'Message from a human.\n\nHumanMessages are messages that are passed in from a human to the model.\n\nExample:\n\n    .. code-block:: python\n\n        from langchain_core.messages import HumanMessage, SystemMessage\n\n        messages = [\n            SystemMessage(\n                content="You are a helpful assistant! Your name is Bob."\n            ),\n            HumanMessage(\n                content="What is your name?"\n            )\n        ]\n\n        # Instantiate a chat model and invoke it with the messages\n        model = ...\n        print(model.invoke(messages))',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'human',
     'default': 'human',
     'enum': ['human'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'example': {'default': False, 'title': 'Example', 'type': 'boolean'}},
   'required': ['content'],
   'title': 'HumanMessage',
   'type': 'object'},
  'HumanMessageChunk': {'additionalProperties': True,
   'description': 'Human Message chunk.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'HumanMessageChunk',
     'default': 'HumanMessageChunk',
     'enum': ['HumanMessageChunk'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'example': {'default': False, 'title': 'Example', 'type': 'boolean'}},
   'required': ['content'],
   'title': 'HumanMessageChunk',
   'type': 'object'},
  'InputTokenDetails': {'description': 'Breakdown of input token counts.\n\nDoes *not* need to sum to full input token count. Does *not* need to have all keys.\n\nExample:\n\n    .. code-block:: python\n\n        {\n            "audio": 10,\n            "cache_creation": 200,\n            "cache_read": 100,\n        }\n\n.. versionadded:: 0.3.9',
   'properties': {'audio': {'title': 'Audio', 'type': 'integer'},
    'cache_creation': {'title': 'Cache Creation', 'type': 'integer'},
    'cache_read': {'title': 'Cache Read', 'type': 'integer'}},
   'title': 'InputTokenDetails',
   'type': 'object'},
  'InvalidToolCall': {'description': 'Allowance for errors made by LLM.\n\nHere we add an `error` key to surface errors made during generation\n(e.g., invalid JSON arguments.)',
   'properties': {'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'title': 'Name'},
    'args': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Args'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
    'error': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'title': 'Error'},
    'type': {'const': 'invalid_tool_call',
     'enum': ['invalid_tool_call'],
     'title': 'Type',
     'type': 'string'}},
   'required': ['name', 'args', 'id', 'error'],
   'title': 'InvalidToolCall',
   'type': 'object'},
  'OutputTokenDetails': {'description': 'Breakdown of output token counts.\n\nDoes *not* need to sum to full output token count. Does *not* need to have all keys.\n\nExample:\n\n    .. code-block:: python\n\n        {\n            "audio": 10,\n            "reasoning": 200,\n        }\n\n.. versionadded:: 0.3.9',
   'properties': {'audio': {'title': 'Audio', 'type': 'integer'},
    'reasoning': {'title': 'Reasoning', 'type': 'integer'}},
   'title': 'OutputTokenDetails',
   'type': 'object'},
  'StringPromptValue': {'description': 'String prompt value.',
   'properties': {'text': {'title': 'Text', 'type': 'string'},
    'type': {'const': 'StringPromptValue',
     'default': 'StringPromptValue',
     'enum': ['StringPromptValue'],
     'title': 'Type',
     'type': 'string'}},
   'required': ['text'],
   'title': 'StringPromptValue',
   'type': 'object'},
  'SystemMessage': {'additionalProperties': True,
   'description': 'Message for priming AI behavior.\n\nThe system message is usually passed in as the first of a sequence\nof input messages.\n\nExample:\n\n    .. code-block:: python\n\n        from langchain_core.messages import HumanMessage, SystemMessage\n\n        messages = [\n            SystemMessage(\n                content="You are a helpful assistant! Your name is Bob."\n            ),\n            HumanMessage(\n                content="What is your name?"\n            )\n        ]\n\n        # Define a chat model and invoke it with the messages\n        print(model.invoke(messages))',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'system',
     'default': 'system',
     'enum': ['system'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'}},
   'required': ['content'],
   'title': 'SystemMessage',
   'type': 'object'},
  'SystemMessageChunk': {'additionalProperties': True,
   'description': 'System Message chunk.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'SystemMessageChunk',
     'default': 'SystemMessageChunk',
     'enum': ['SystemMessageChunk'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'}},
   'required': ['content'],
   'title': 'SystemMessageChunk',
   'type': 'object'},
  'ToolCall': {'description': 'Represents a request to call a tool.\n\nExample:\n\n    .. code-block:: python\n\n        {\n            "name": "foo",\n            "args": {"a": 1},\n            "id": "123"\n        }\n\n    This represents a request to call the tool named "foo" with arguments {"a": 1}\n    and an identifier of "123".',
   'properties': {'name': {'title': 'Name', 'type': 'string'},
    'args': {'title': 'Args', 'type': 'object'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
    'type': {'const': 'tool_call',
     'enum': ['tool_call'],
     'title': 'Type',
     'type': 'string'}},
   'required': ['name', 'args', 'id'],
   'title': 'ToolCall',
   'type': 'object'},
  'ToolCallChunk': {'description': 'A chunk of a tool call (e.g., as part of a stream).\n\nWhen merging ToolCallChunks (e.g., via AIMessageChunk.__add__),\nall string attributes are concatenated. Chunks are only merged if their\nvalues of `index` are equal and not None.\n\nExample:\n\n.. code-block:: python\n\n    left_chunks = [ToolCallChunk(name="foo", args=\'{"a":\', index=0)]\n    right_chunks = [ToolCallChunk(name=None, args=\'1}\', index=0)]\n\n    (\n        AIMessageChunk(content="", tool_call_chunks=left_chunks)\n        + AIMessageChunk(content="", tool_call_chunks=right_chunks)\n    ).tool_call_chunks == [ToolCallChunk(name=\'foo\', args=\'{"a":1}\', index=0)]',
   'properties': {'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'title': 'Name'},
    'args': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Args'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
    'index': {'anyOf': [{'type': 'integer'}, {'type': 'null'}],
     'title': 'Index'},
    'type': {'const': 'tool_call_chunk',
     'enum': ['tool_call_chunk'],
     'title': 'Type',
     'type': 'string'}},
   'required': ['name', 'args', 'id', 'index'],
   'title': 'ToolCallChunk',
   'type': 'object'},
  'ToolMessage': {'additionalProperties': True,
   'description': 'Message for passing the result of executing a tool back to a model.\n\nToolMessages contain the result of a tool invocation. Typically, the result\nis encoded inside the `content` field.\n\nExample: A ToolMessage representing a result of 42 from a tool call with id\n\n    .. code-block:: python\n\n        from langchain_core.messages import ToolMessage\n\n        ToolMessage(content=\'42\', tool_call_id=\'call_Jja7J89XsjrOLA5r!MEOW!SL\')\n\n\nExample: A ToolMessage where only part of the tool output is sent to the model\n    and the full output is passed in to artifact.\n\n    .. versionadded:: 0.2.17\n\n    .. code-block:: python\n\n        from langchain_core.messages import ToolMessage\n\n        tool_output = {\n            "stdout": "From the graph we can see that the correlation between x and y is ...",\n            "stderr": None,\n            "artifacts": {"type": "image", "base64_data": "/9j/4gIcSU..."},\n        }\n\n        ToolMessage(\n            content=tool_output["stdout"],\n            artifact=tool_output,\n            tool_call_id=\'call_Jja7J89XsjrOLA5r!MEOW!SL\',\n        )\n\nThe tool_call_id field is used to associate the tool call request with the\ntool call response. This is useful in situations where a chat model is able\nto request multiple tool calls in parallel.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'tool',
     'default': 'tool',
     'enum': ['tool'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'tool_call_id': {'title': 'Tool Call Id', 'type': 'string'},
    'artifact': {'default': None, 'title': 'Artifact'},
    'status': {'default': 'success',
     'enum': ['success', 'error'],
     'title': 'Status',
     'type': 'string'}},
   'required': ['content', 'tool_call_id'],
   'title': 'ToolMessage',
   'type': 'object'},
  'ToolMessageChunk': {'additionalProperties': True,
   'description': 'Tool Message chunk.',
   'properties': {'content': {'anyOf': [{'type': 'string'},
      {'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]},
       'type': 'array'}],
     'title': 'Content'},
    'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'},
    'response_metadata': {'title': 'Response Metadata', 'type': 'object'},
    'type': {'const': 'ToolMessageChunk',
     'default': 'ToolMessageChunk',
     'enum': ['ToolMessageChunk'],
     'title': 'Type',
     'type': 'string'},
    'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Name'},
    'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}],
     'default': None,
     'title': 'Id'},
    'tool_call_id': {'title': 'Tool Call Id', 'type': 'string'},
    'artifact': {'default': None, 'title': 'Artifact'},
    'status': {'default': 'success',
     'enum': ['success', 'error'],
     'title': 'Status',
     'type': 'string'}},
   'required': ['content', 'tool_call_id'],
   'title': 'ToolMessageChunk',
   'type': 'object'},
  'UsageMetadata': {'description': 'Usage metadata for a message, such as token counts.\n\nThis is a standard representation of token usage that is consistent across models.\n\nExample:\n\n    .. code-block:: python\n\n        {\n            "input_tokens": 350,\n            "output_tokens": 240,\n            "total_tokens": 590,\n            "input_token_details": {\n                "audio": 10,\n                "cache_creation": 200,\n                "cache_read": 100,\n            },\n            "output_token_details": {\n                "audio": 10,\n                "reasoning": 200,\n            }\n        }\n\n.. versionchanged:: 0.3.9\n\n    Added ``input_token_details`` and ``output_token_details``.',
   'properties': {'input_tokens': {'title': 'Input Tokens', 'type': 'integer'},
    'output_tokens': {'title': 'Output Tokens', 'type': 'integer'},
    'total_tokens': {'title': 'Total Tokens', 'type': 'integer'},
    'input_token_details': {'$ref': '#/$defs/InputTokenDetails'},
    'output_token_details': {'$ref': '#/$defs/OutputTokenDetails'}},
   'required': ['input_tokens', 'output_tokens', 'total_tokens'],
   'title': 'UsageMetadata',
   'type': 'object'}},
 'anyOf': [{'type': 'string'},
  {'$ref': '#/$defs/StringPromptValue'},
  {'$ref': '#/$defs/ChatPromptValueConcrete'},
  {'items': {'oneOf': [{'$ref': '#/$defs/AIMessage'},
     {'$ref': '#/$defs/HumanMessage'},
     {'$ref': '#/$defs/ChatMessage'},
     {'$ref': '#/$defs/SystemMessage'},
     {'$ref': '#/$defs/FunctionMessage'},
     {'$ref': '#/$defs/ToolMessage'},
     {'$ref': '#/$defs/AIMessageChunk'},
     {'$ref': '#/$defs/HumanMessageChunk'},
     {'$ref': '#/$defs/ChatMessageChunk'},
     {'$ref': '#/$defs/SystemMessageChunk'},
     {'$ref': '#/$defs/FunctionMessageChunk'},
     {'$ref': '#/$defs/ToolMessageChunk'}]},
   'type': 'array'}],
 'title': 'ChatOpenAIInput'}
```

* **流式输出**

```python
for chunk in chain.stream("大模型技术"):
    print(chunk, end="", flush=True)
```

```plaintext
大模型技术（Large Model Technology）是指基于大规模数据和计算资源训练的深度学习模型，通常具有数十亿甚至数千亿个参数。这类模型在自然语言处理（NLP）、计算机视觉、语音识别等领域表现出色，能够处理复杂的任务并生成高质量的结果。大模型技术的核心在于通过海量数据和强大的计算能力，学习到更加通用和深层次的模式，从而实现更智能化的应用。

### 大模型技术的特点
1. **参数规模大**：大模型的参数量通常在数十亿到数千亿之间，例如 OpenAI 的 GPT-3 有 1750 亿个参数。
2. **数据需求高**：训练大模型需要海量的高质量数据，通常来自互联网、书籍、论文等。
3. **计算资源密集**：训练大模型需要高性能的计算设备（如 GPU、TPU）和分布式计算框架。
4. **通用性强**：大模型通常具有多任务学习能力，能够处理多种任务，如文本生成、翻译、问答等。
5. **迁移学习能力**：大模型可以通过微调（Fine-tuning）快速适应特定任务，减少对任务特定数据的需求。

### 大模型技术的应用
1. **自然语言处理（NLP）**：
   - 文本生成（如 GPT 系列）
   - 机器翻译（如 Google 的 Transformer 模型）
   - 问答系统（如 ChatGPT）
   - 情感分析、文本分类等
2. **计算机视觉**：
   - 图像生成（如 DALL·E）
   - 目标检测、图像分类
3. **语音识别与合成**：
   - 语音转文字（如 Whisper）
   - 语音合成（如 Tacotron）
4. **多模态任务**：
   - 结合文本、图像、语音等多种模态的任务（如 CLIP、Flamingo）

### 大模型技术的挑战
1. **计算成本高**：训练和部署大模型需要大量的计算资源和能源。
2. **数据隐私与安全**：大模型训练依赖于海量数据，可能涉及隐私泄露问题。
3. **模型可解释性**：大模型的决策过程复杂，难以解释其内部机制。
4. **环境影响**：训练大模型消耗大量能源，可能对环境造成负面影响。
5. **伦理与偏见**：大模型可能从训练数据中学习到偏见，导致不公平的结果。

### 典型的大模型
1. **GPT 系列**（OpenAI）：如 GPT-3、GPT-4，专注于文本生成和自然语言理解。
2. **BERT**（Google）：基于 Transformer 架构，擅长文本分类、问答等任务。
3. **T5**（Google）：统一了多种 NLP 任务的框架。
4. **DALL·E**（OpenAI）：生成高质量图像的多模态模型。
5. **PaLM**（Google）：参数规模达 5400 亿的语言模型。

### 未来发展方向
1. **更高效的训练方法**：如稀疏模型、模型压缩技术。
2. **多模态融合**：结合文本、图像、语音等多种模态的模型。
3. **绿色 AI**：降低大模型的能耗和环境影响。
4. **个性化与隐私保护**：在保护用户隐私的前提下实现个性化服务。
5. **可解释性与公平性**：提高模型的可解释性，减少偏见和歧视。

大模型技术正在推动人工智能的快速发展，但也面临诸多挑战。未来需要在技术、伦理和社会影响之间找到平衡，以实现更广泛的应用和可持续发展。
```

  整个过程其实并不难理解，在`LCEL`中，所有组件都实现了`runable`接口，而`runable`的接口协议会将每个组件的输入和输出做一个描述，传入`pydantic`模型做一个强检验，如果一切正常则会正常传递数据，否则程序则抛出异常。

# 3. 实现复杂RAG聊天机器人

  接下来，我们进一步探讨 `LangChain` 和 `DeepSeek v3`模型如何构建一个复杂的 `RAG` 聊天机器人，能够处理复杂的查询，并且可以通过聊天历史记录维护上下文，并使用 `LangChain` 的 `LCEL`语法遵守严格的`Guardrails`（护栏）。

  `Guardrails`(护栏)对于确保`AI`系统的安全性和可靠性是比较重要的。通过设定明确的界限，我们可以防止大模型生成有害或误导性的内容。拒绝机制使机器人能够礼貌地拒绝违反这些护栏的请求，例如与敏感主题或非法活动相关的请求。

  这里我们创建一个智能HR聊天机器人助手，该机器人将能够利用私有知识库回答有关公司政策、程序和福利的问题。

![](images/34093aad-e100-498f-8e6c-c81961be98b6.png)

  使用`DeepSeek v3`模型作为对话模型。

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="deepseek-chat",
                   api_key='sk-6c960cb2a3a24fa6b92f6993d5942777',
                   base_url='https://api.deepseek.com')
```

  使用`OpenAI`的`Embeddings`模型将自然语言转化成词向量的表示。

```python
from langchain_openai import OpenAIEmbeddings

embed = OpenAIEmbeddings(
    api_key='sk-proj-Vs1hvbqAjlKBnTS9yTrNygLSq16ElNdDuTV1ADDgfezt3b4B5oA',
    base_url='https://ai.devtool.tech/proxy/v1',     # 国内中转地址
    model="text-embedding-3-large" 
)
```

```python
# ! pip install faiss-cpu
```

  接下来，我们使用`FAISS`作为矢量数据库。 `FAISS` 是 `Facebook AI Research` 开发的一个库，用于高效相似性搜索和密集向量聚类。`LangChain`在第三方集成模块（Langchain\_community）中已经接入了`FAISS`向量数据库，所以我们就可以直接使用。

```python
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# 加载一些模拟的假数据
doc1 = Document(page_content="员工每年享有一定数量的病假。有关资格和具体细节可以在员工手册中找到。")
doc2 = Document(page_content="员工请病假时，必须首先通知其主管关于病情和预计缺勤时间。员工需填写病假申请表，并提交给人力资源部门或主管。")
doc3 = Document(page_content="病假申请表可以在公司内部网找到。表格需要填写员工姓名、部门、缺勤日期和缺勤原因等信息。")


# 创建 Faiss 向量存储
vector_store = FAISS.from_documents([doc1, doc2, doc3], embed)

# 将文件保存到本地，包括：向量数据、索引文件和元数据文件
vector_store.save_local(folder_path='.')
```

  创建矢量数据库后，我们可以进行测试：

```python
# 加载本地的Faiss向量文件，allow_dangerous_deserialization 用于控制是否允许在加载向量存储时进行潜在的危险反序列化操作。
vector_store = FAISS.load_local(embeddings=embed, folder_path='.',allow_dangerous_deserialization=True)
 
# 将 FAISS 向量存储转换为一个 retriever（检索器），并为该检索器设置一些搜索相关的参数。k=1 表示检索时返回 最相似的 1 个文档
retriever = vector_store.as_retriever(search_kwargs={'k': 1})

# 执行相似度搜素
query = "请问我们公司有没有病假？"
results = retriever.invoke(query)

for doc in results:
    print(f"Content: {doc.page_content}")
```

```plaintext
Content: 员工每年享有一定数量的病假。有关资格和具体细节可以在员工手册中找到。
```

  我们从一个最简单的链开始，只接受用户问题，在提示中格式化它并输出该问题的答案（不检索）。这里使用 `Langchain` 的`PromptTemplate`并使用`LCEL`对其进行管道传输。

```python
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 定义提示模板
prompt = PromptTemplate(
  input_variables = ["question"],
  template = "你是一个乐于助人的智能小助理。擅长根据用户输入的问题给出一个简短的回答：: {question}"
)


# 构建Chains
chain = (
  prompt
  | model
  | StrOutputParser()
)
print(chain.invoke({"question": "请问什么是人工智能？"}))
```

```plaintext
人工智能（AI）是指通过计算机系统模拟人类智能的能力，包括学习、推理、问题解决、感知和语言理解等。AI可以分为弱人工智能（专注于特定任务）和强人工智能（具备通用智能，能像人类一样处理各种任务）。常见的AI应用包括语音助手、图像识别、自动驾驶等。
```

  在这个过程中，会将带有`question`键的字典被传递到提示模板中，其中`question`值被提取并在模板中格式化，然后作为输入传递到`model`，最后将结果提取为使用`StrOutputPaser()`最终输出字符串。

  接下来，因为最终我们想要构建一个聊天机器人，所以需要让它支持聊天历史记录，作为`RAG`系统的一个基础组件。当调用链时，以列表的形式传递历史记录，指定每条消息是由用户还是助手发送的。例如：

```python
［ {"role": "user", "content": "我每年可以请多少天病假？"}，
   {"role": "assistant", "content": "你每年可以请的病假天数取决于你的具体雇佣合同和公司政策。然而，一般来说，员工有权享受一定的病假。具体细节请参阅员工手册或与人力资源部门联系。"},
   {"role": "user", "content": "我在哪里可以找到员工手册？"}，
   {"role": "assistant", "content“: ”员工手册通常在公司内部网上提供。你也可以联系你的人力资源部门索要一份实体副本。"}
］
```

  然后创建链组件，将此输入转换为传递给`prompt_with_history`的输入。与上面的代码类似，但在这里我们需要创建一个 `RunnableLambda`，它用来获取消息列表并从中提取问题和历史记录。然后使用 `LangChain LCEL` 为变量`问题`分配一个管道，该管道首先从字典中提取关键消息。

```python
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

# 问题是历史记录中的最后一项
def extract_question(input):
    return input[-1]["content"]

# 历史记录是除了最后一个问题之外的所有内容
def extract_history(input):
    return input[:-1]


prompt_with_history_str = """
你是一个人力资源助理聊天机器人。请只回答HR相关问题。如果你不知道或者这个问题与人力资源无关，就不要回答。
这是你与用户对话的历史记录: {chat_history}
现在，请回答这个问题: {question}
"""

# 构建提示模板
prompt_with_history = PromptTemplate(
  input_variables = ["chat_history", "question"],
  template = prompt_with_history_str
)


# 构建带有历史会话记录的链
chain_with_history = (
    {
        # Itemgetter：从输入字典中提取特定键，这里指定的是 messages 列表
        # 自定义 lambda 函数可用于进一步处理提取的数据，从messages列表中提取question和chat_history 
        "question": itemgetter("messages") | RunnableLambda(extract_question), 
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | prompt_with_history
    | model
    | StrOutputParser()
)

print(chain_with_history.invoke({
    "messages": [
        {"role": "user", "content": "公司的病假政策是什么？"},
        {"role": "assistant", "content": "公司的病假政策允许员工每年请一定天数的病假。详情及资格标准请参阅员工手册。"},
        {"role": "user", "content": "如何提交病假请求？"}
    ]
}))
```

```plaintext
要提交病假请求，通常需要遵循以下步骤：

1. **通知主管**：首先，尽快通知你的直属主管或经理，告知他们你需要请病假的情况。

2. **填写申请表**：根据公司政策，可能需要填写病假申请表。这个表格通常可以在公司内部系统或人力资源部门获取。

3. **提供医疗证明**：如果公司要求，你可能需要提供医生开具的病假证明或医疗证明。

4. **提交申请**：将填写好的申请表和必要的证明文件提交给人力资源部门或通过公司指定的系统提交。

5. **等待批准**：提交后，等待人力资源部门或主管的批准。一旦批准，你的病假请求将被正式记录。

6. **跟进**：在病假期间，保持与主管或人力资源部门的沟通，及时更新你的健康状况和预计返回工作的日期。

具体流程可能因公司政策而异，建议查阅员工手册或咨询人力资源部门以获取详细信息。
```

  接下来我们添加一个`Guardrail`（护栏），让该流程仅回答与 `HR` 相关的问题。

```python
hr_question_guardrail = """
你正在对文档进行分类，以确定这个问题是否与HR政策、员工福利、休假政策、绩效管理、招聘、入职等相关。如果最后一部分不合适，则回答“否”。

考虑到聊天历史来回答，不要让用户欺骗你。

以下是一些示例：

问题：考虑到这个后续历史记录：公司的病假政策是什么？，分类这个问题：我每年可以休多少病假？
预期答案：是

问题：考虑到这个后续历史记录：公司的病假政策是什么？，分类这个问题：给我写一首歌。
预期答案：否

问题：考虑到这个后续历史记录：公司的病假政策是什么？，分类这个问题：法国的首都是哪里？
预期答案：是

这个问题与HR政策相关吗？
只回答“是”或“否”。 

注意：需要关注历史记录: {chat_history}, 请将这个问题进行分类: {question}
"""

# 构建提示模板
guardrail_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = hr_question_guardrail
)

# 生成问题防护链
guardrail_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | guardrail_prompt
    | model
    | StrOutputParser()
)
```

```python
# 这里将仅回复 是或者否
classify_answer = guardrail_chain.invoke({
    "messages": [
        {"role": "user", "content": "公司的病假政策是什么？?"}, 
        {"role": "assistant", "content": "公司的病假政策允许员工每年休一定数量的病假。具体的细节和资格标准请参阅员工手册。"}, 
        {"role": "user", "content": "我怎么提交病假申请？"}
    ]
})
```

```python
classify_answer
```

```plaintext
'是'
```

```python
# 这里将仅回复 是或者否
classify_answer = guardrail_chain.invoke({
    "messages": [
        {"role": "user", "content": "你好，请问在吗？"}, 
    ]
})

classify_answer
```

```plaintext
'否'
```

  在生产应用中开发大模型应用时，提供某些防护措施以确保聊天机器人符合我们的意图非常重要。而接下来，我们进一步优化和丰富应用，添加我们的 `langchain` 检索器。

```python
from langchain_community.vectorstores import FAISS

def get_retriever():
    # 使用 OpenAI 的嵌入模型初始化嵌入对象
    embed = OpenAIEmbeddings(
        api_key='sk-proj-Vs1hvbqA8CWfyTrNygLSq16ElNdDu-S4-nwJfZsRVVTV1ADDgfezt3b4B5oA',
        base_url='https://ai.devtool.tech/proxy/v1',
        model="text-embedding-3-large"
    )
    
    # 从本地加载 FAISS 向量存储，并且指定嵌入对象
    vector_store = FAISS.load_local(embeddings=embed, folder_path='.',allow_dangerous_deserialization=True)
     
    # 配置文档检索，返回最相关的 1 个文档
    retriever = vector_store.as_retriever(search_kwargs={'k': 1})
    return retriever

# 构建检索器实例
retriever = get_retriever()

# 生成检索链
retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | retriever
)

print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "病假的HR政策是什么？"}]}))
```

```plaintext
[Document(metadata={}, page_content='员工每年享有一定数量的病假。有关资格和具体细节可以在员工手册中找到。')]
```

  最后，我们实现完整的链来连接检索器。

![](images/34093aad-e100-498f-8e6c-c81961be98b6-1.png)

```python
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough

question_with_history_and_context_str = """
你是一个可信赖的 HR 政策助手。你将回答有关员工福利、休假政策、绩效管理、招聘、入职以及其他与 HR 相关的话题。如果你不知道问题的答案，你会诚实地说你不知道。
阅读讨论以获取之前对话的上下文。在聊天讨论中，你被称为“系统”，用户被称为“用户”。

历史记录: {chat_history}

以下是一些可能帮助你回答问题的上下文： {context}

请直接回答，不要重复问题，不要以“问题的答案是”之类的开头，不要在答案前加上“AI”，不要说“这是答案”，不要提及上下文或问题。

根据这个历史和上下文，回答这个问题： {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "context", "question"],
  template = question_with_history_and_context_str
)

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])


# 定义不相关的链
irrelevant_question_chain = (
  RunnableLambda(lambda x: {"result": '我不能回答与 HR 政策无关的问题。'})
)

# 定义相关的链
relevant_question_chain = (
  RunnablePassthrough() 
  |
  {
    "relevant_docs": prompt | model | StrOutputParser() | retriever,
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
 |
  {
    "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "prompt": question_with_history_and_context_prompt,         
  }
  |
  {
    "result": itemgetter("prompt") | model | StrOutputParser(),
  }
)


# 定义分支
branch_node = RunnableBranch(
  (lambda x: "是" in x["question_is_relevant"].lower(), relevant_question_chain),
  (lambda x: "否" in x["question_is_relevant"].lower(), irrelevant_question_chain),
  irrelevant_question_chain
)

full_chain = (
  {
    "question_is_relevant": guardrail_chain,
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),    
  }
  | branch_node
)
```

```python
import json

non_relevant_dialog = {
    "messages": [
        {"role": "user", "content": "公司的病假政策是什么？"},
        {"role": "assistant", "content": "公司的病假政策允许员工每年休一定数量的病假。具体的细节和资格标准请参阅员工手册。"},
        {"role": "user", "content": "你好，请你介绍一下你自己呀。"}
    ]
}

print(f'用不相关的问题测试')
response = full_chain.invoke(non_relevant_dialog)
```

```plaintext
用不相关的问题测试
```

```python
response
```

```plaintext
{'result': '我不能回答与 HR 政策无关的问题。'}
```

```python
dialog = {
    "messages": [
        {"role": "user", "content": "公司的病假政策是什么？"},
        {"role": "assistant", "content": "公司的病假政策允许员工每年休一定数量的病假。具体的细节和资格标准请参阅员工手册。"},
        {"role": "user", "content": "我应该如何提交病假的申请？"}
    ]
}
```

```python
print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "我应该如何提交病假的申请？?"}]}))
```

```plaintext
[Document(metadata={}, page_content='员工请病假时，必须首先通知其主管关于病情和预计缺勤时间。员工需填写病假申请表，并提交给人力资源部门或主管。')]
```

```python
print(f'用相关的问题测试')
response = full_chain.invoke(dialog)
```

```plaintext
用相关的问题测试
```

```python
response
```

```plaintext
{'result': '你应该首先通知你的主管关于病情和预计缺勤时间。然后，填写病假申请表并提交给人力资源部门或你的主管。'}
```

***

公开课**全部课件，以及项目代码、数据等学习资料**，扫码⬇️联系助教即可领取～

此外我们还创办了**赋范大模型技术交流群**，超活跃氛围，来遇见更多志同道合的小伙伴\~

![](images/image-1.png)

![](images/f339b04b7b20233dd1509c7fb36d5c0-1.png)

* 体验课时间有限，若想深度学习大模型技术，欢迎大家报名由我主讲的[《2025大模型Agent智能体开发实战》](https://whakv.xetslk.com/s/3xFEAA)

**[《2025大模型Agent智能体开发实战》](https://whakv.xetslk.com/s/3xFEAA)为【100+小时】体系大课，总共20大模块精讲精析，零基础直达大模型企业级应用！了解详细信息可扫码咨询**
