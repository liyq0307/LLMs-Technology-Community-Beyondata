# 课程说明：

* 体验课内容节选自[《2025大模型Agent智能体开发实战》](https://whakv.xetslk.com/s/1zrFP8)完整版付费课程

  体验课时间有限，若想深度学习大模型技术，欢迎大家报名由我主讲的[《2025大模型Agent智能体开发实战》](https://whakv.xetslk.com/s/1zrFP8)：

![](images/image-7.png)

![](images/image-9.png)

此外，公开课全部训练项目代码、数据、以及训练完的模型，已上传至课件网盘，联系⬆️助教即可领取。

***

## **一** **、** **主** **流** **大** **模** **型** **W** **e** **b** **对** **话** **框** **架** **介** **绍**

本节公开课，我们来详细介绍如何借助DeepSeek  R1进行本地知识库问答。

此前公开课中，我们已经详细讨论了DeepSeek R1从原版模型到蒸馏模型的各系列模型本地安装部 署方法，以及借助ollama、VLLM、LMDeploy   等主流推理框架进行调用等方法，详情见：https://www.bilibili.com/video/BV19kFoe6Ef7/

![](images/image-8.png)

本节公开课，我们继续探讨如何借助DeepSeek  R1进行本地知识库问答。

关于本地知识库问答，目前有非常多实现方法，主流的本地知识库问答开源框架如下：

### 【个人】多端&多模型聊天框架：ChatBox

项目展示：

![](images/image-11.png)

GitHub地址：https://github.com/Bin-Huang/chatbox



### **【个人】轻量级便捷RAG** **推理框架：AnythingLLM**

**o 项目展示：**

![](images/image-10.png)

![](images/image.png)

o  项目官网：<https://anythingllm.com/>

### 【企业级】RAG\&GraphRAG   知识库对话框架：kotaemon

o  项目展示：

![](images/image-1.png)

![](images/image-2.png)

Github主页地址https:/github.com/Cinnamon/kotaemon&#x20;

### 【企业级】RAG\&Agent  多功能综合前端框架：Open-WebUI

项目展示：

![](images/image-3.png)

![](images/image-4.png)

GitHub项目地址：https://github.com/open-webui/open-webui&#x20;

### 【企业级】多模态交互框架：lobe-chat

项目展示：

![](images/image-5.png)

![](images/image-6.png)

GitHub项目地址：https://github.com/lobehub/lobe-chat

本节公开课将以AnythingLLM为例介绍DeepSeek R1本地知识库问答实现全流程，而对于更加复杂的框 架，如Open-WebUI、Kotaemon  等，我们提供了完整的完整的参考课件资料，与本节公开课课件、代 码、软件等保存在同一网盘内，链接: https://pan.baidu.com/s/1SdOUanl5eLTb8bqNbsfQFQ 提取码: 76ym&#x20;

![](images/image40.jpeg)

![](images/image41.jpeg)

![](images/image42.jpeg)

# **二、O** **llama+DeepSeek** **R1** **Distill模型本地部署**   &#x20;

●模型选择与硬件配置说明

本节公开课以Windows 系统为例进行演示，MacOS、Linux 系统实现方式也类似。同时，公开课重 点介绍完全本地部署实现方案，并采用DeepSeek  R1蒸馏模型进行演示，若硬件条件允许，也可考虑部 署原版DeepSeek R1模型。各模型运行所需显存如下：

| **DeepSeek** **R1模型组**        | **模型推理性能**       | **半精度运行所需显存** | **全精度运行所需显存** |
| ----------------------------- | ---------------- | ------------- | ------------- |
| **DeepSeek-R1-1.5B(Distill)** | **GPT4o级**       | **1.1G**      | **4G**        |
| **DeepSeek-R1-7B(Distill)**   | **超越GPT40**      | **4.7G**      | **14G**       |
| **DeepSeek-R1-8B(Distill)**   | **超越GPT4o**      | **4.9G**      | **14G**       |
| **DeepSeek-R1-14B(Distill)**  | **超越GPT4o**      | **9G**        | **24G**       |
| **DeepSeek-R1-32B(Distill)**  | **o1** **mini级** | **20G**       | **55G**       |
| **DeepSeek-R1-70B(Distill)**  | **o1** **mini级** | **43G**       | **120G**      |
| **DeepSeek-R1-671B**          | **o1级**          | **404G**      | **1000G**     |

而关于如何使用DeepSeek R1在线API实现相关功能，详见Part 4部分内容介绍。

●模型本地调度框架选取

同其他主流大模型对话Web框架类似，AnythingLLM 同时支持调用在线模型API、或本地模型进行推理， 并且AnythingLLM 本身是一款集成度非常高的聊天框架，我们只需要借助可视化页面手动点击选择在线 模型AP 或者本地推理框架即可，例如一些主流API厂商：

![](images/image-12.png)

**或者一些主流的本地推理框架。**

![](images/image-13.png)

本节公开课我们将以Ollama 为例进行演示。

# **Step 1.安装ollama**

首先需要先安装ollama  地址，下载地址： <https://ollama.com/>

![](images/image-14.png)

**选择对应的操作系统进行安装即可：**

![](images/image-15.png)

Ollama 下载存在网络限制，对应Windows、MacOS    软件安装包开在课件网盘中领取，链接: https://pan.baidu.com/s/1SdOUanl5eLTb8bqNbsfQFQ 提取码: 76ym&#x20;

![](images/image-25.png)

下载后即可开始安装：

![](images/image-26.png)

![](images/image-24.png)

而当Ollama 安装结束后，会自动在后台开启Ollama 进 程 ，Windows  系统下可以按住Win+r 键，输入cmd 开启命令行，并在命令行中查看Ollama 运行情况：

![](images/image-22.png)

例如，输入tasklist   |findstr   ollama查 看Ollama 进程是否存在

![](images/image-21.png)

如此则说明Ollama 已启动。若未启动，可输入`ollama start`启 动Ollama。

# **Step 2.下载DeepSeek R1模型**

接下来即可进一步下载DeepSeek    R1模型了，这里我们可以直接使用`ollama pull`命令自动下载与 注册，也可以手动下载模型的GGUF 文件，然后手动完成Ollama 模型注册，再进行调用。

这里我们考虑使用`ollama   pull`方法进行模型权重下载，而若网络条件不允许，也可以参考我的另 一个视频：《DeepSeek  R1本地部署流程详解》<https://www.bilibili.com/video/BV19kFoe6Ef7/>        , 从 魔 搭社区上手动下载模型GGUF 权重，并完成Ollama 注册与调用。

![](images/image-18.png)

**首先输入01lama** **list查看当前模型列表：**

![](images/image-23.png)

然 后 回 到ollama deepseek  r1模 型 列 表 ， 查 看 模 型 组 情 况 ：<https://ollama.com/librarv/deepseek-r1>

![](images/image-20.png)

![](images/image-19.png)

**其中不同尺寸模型运行所需占用显存大小如上所示。需要注意的是，ollama** **只支持运行Q4** **K** **M** **模** **型，相当于是4bit量化后的模型，因此每个模型运行所需占用显存并不大。**

**接下来使用如下命令下载模型，这里以14b模型为例，下载其他模型的话只需要修改模型名称即可，** **例如修改为deepseek-r1:7b,     则代表下载7b模型：**

```python
ollama pull deepseek-r1:14b
```

![](images/image-16.png)

稍等片刻即可完成下载。下载完成后即可使用0llama   list命令查看当前模型是否完成下载与注册：

![](images/image-17.png)

接下来即可调用DeepSeek R1 Distill 14B模型进行本地知识库问答了。

# **三** **、** **A** **n** **y** **t** **h** **i** **n** **g** **L** **L** **M** **安** **装** **与** **使** **用**

AnythingLLM  是一款集成度非常高、且功能非常稳定的大模型对话前端产品，虽无法进行二次开 发，但功能实现方面非常便捷，可以直接从AnythingLLM  官网进行下载： https://anvthingllm.com/

![](images/image-41.png)

选择对应的操作系统版本下载安装即可开始安装：

![](images/image-40.png)

安装完成后，会自动启动AnythingLLM, 并进入设置页面

![](images/image-39.png)

点击Get   started,则进入到选择LLM页面：

![](images/image-36.png)

这里我们选择借助Ollama 来调用本地的14B模 型 。这 里需要注意的是，AnythingLLM 会自动检测Ollama  进程是否存在(也就是Ollama 是否启动),以及现在Ollama 中已经注册了哪些模型。因此在进入到上面 这个设置也面前，需要确保Ollama 已经启动，并且已完成了相关模型注册。选择好14B模型之后即可点  击进入下一步：

![](images/image-35.png)

该页面是进行模型配置说明，当前的模型配置是：

●对话模型：负责完成对话任务，由Ollama 调 度DeepSeek R1 Distill 14B来完成对话；

●Embedding   模型：负责进行词向量化的模型，由AnythingLLM  官方提供；

●词向量数据库：用于存储词向量化之后的对象，默认为LancelotDB;

对于任何一个本地知识库问答的过程，都需要经历文本切割、词向量化、词向量存储、词向量检索与问 答几个基本环节，因此至少需要Chat 模 型 、Embedding   模型以及词向量数据库作为基础组件。对于RAG 的基本原理回顾如下：

![](images/image-38.png)

在浏览了当前模型配置后，接下来即将进入AnythingLLM注册页面，这里可以直接跳过：

![](images/image-37.png)

然后需要创建一个workspace,  相当于是ChatGPT 或者其他开发项目的一个Project,用于对对话进行分 类：

![](images/image-34.png)

**当我们进入到如下页面，则说明安装成功：**

![](images/image-33.png)

接下来即可尝试进行对话了。AnythingLLM  对于DeepSeek    R1系列模型的支持是非常友好的，甚至会区

分思考过程和回答过程：

![](images/image-31.png)

并且支持多轮对话：

![](images/image-30.png)

# **四、借助Anything LL M进行本地知识库问答**    &#x20;

AnythingLLM 是一款集成度非常高、且运行稳定的RAG+ 前端对话框架，我们可以非常便捷的为每次 对话设置知识库。主要方法有以下两种：

●在单次问答中设置背景文本

例如，我们可以直接在对话框下方点击回形针按钮，即可上传文本并进行对话，整个过程类似 ChatGPT的功能实现：

注：这里所谓上传文本，并不是把文本上传到远程服务器，而是将文本进行切分、Embedding 后 后上传到本地的LanceDB词向量数据库中。整个过程全部都在本地完成。

![](images/image-32.png)

**上** **传** **文** **本** **：**

![](images/image-27.png)

**等** **待** **文** **本** **切** **分** **与** **词** **向** **量** **化** **过** **程**

![](images/image-28.png)

**然** **后** **即** **可** **开** **始** **问** **答** **：**

![](images/image-29.png)

此 时 迷 哦 行 问 答 即 会 根 据 文 档 进 行 回 答 。

●统一知识库管理

此外，若想让整个文本长期存在于当前对话中，则需要借助AnythingLLM 的知识库管理功能。下面这个 icon就是知识库管理图标(UI 需优化):

![](images/image-56.png)

**点击后即可进入到如下页面：**

![](images/image-55.png)

这里我们额外上传一份DeepSeek V3技术报告，该报告的内容超出了当前模型只是范畴，适合进行本地 文档问答测试：

![](images/image-53.png)

上传后即可在当前文件夹中看到该文件，这里点击Move to Workspace,即可将其列举为备选问答文 档：

![](images/image-51.png)

当Move to workspace后，在右侧即可看到这篇文档，然后点击Save and Embed,即可开始进行文档切 分和词向量化过程：

![](images/image-54.png)

**需要稍等片刻：**

![](images/image-52.png)

完成后，再点击图钉按钮，即可将这篇文档设置为当前对话的背景文档：

![](images/image-50.png)

**然后点击got it**

![](images/image-49.png)

然后即可提问：

![](images/image-47.png)

能够看出，此时回答结果就会根据当前文档进行回答，细节丰满回答流畅。而若不绑定知识库，则模型 并不知道DeepSeek  V3模型：

![](images/image-46.png)

至此，即完成了DeepSeek R1 Distill 14B模型的本地知识库问答全流程。

# 五、DeepSeek  R1 API接入AnythingLLM   进行本地知识库 问 答

最后，让我们来看下如何借助DeepSeek R1 API+AnythingLLM来进行本地知识库问答。 首先，新建一个工作区，并命名为RAG Test2

![](images/image-45.png)

**然后点击设置按钮：**

![](images/image-44.png)

设置页面，我们可以灵活设置当前工作区域的名称(或者删除工作区)

![](images/image-48.png)

或者修改默认模型，以及system message (系统默认提示):

![](images/image-43.png)

**以及进行词向量数据库设置等。**

![](images/image-42.png)

这里我们需要在聊天设置中修改模型为DeepSeek R1 API,先点击聊天设置，然后选择模型：

![](images/image-63.png)

然后在模型提供商中选择DeepSeek:

![](images/image-60.png)

**然后在弹出的页面输入API-KEY**

![](images/image-62.png)

输入完API-Key后，即可选择对话模型，deepseek-chat 就是DeepSeek  V3,而 deepseek-reasoner则是 DeepSeek  R1。这里我们选择deepseek-reasoner:

![](images/image-61.png)

**然后点击Save, 即可返回设置页面：**

![](images/image-57.png)

然后翻到最下面点击Update    workspace,即可完成设置：

![](images/image-58.png)

然后点击左侧聊天栏，选择default会话即可开始聊天了：

![](images/image-59.png)

如此即可完成对话，而若要进一步设置知识库，可参考Part 4中部分内容，无论是调用在线模型API, 还 是调用本地模型，知识库设置流程都完全一样。
