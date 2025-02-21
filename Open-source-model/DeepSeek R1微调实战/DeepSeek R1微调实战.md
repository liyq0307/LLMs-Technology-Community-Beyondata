# **DeepSeek** **R1高效微调实战入门**                 &#x20;

本节公开课，我们来探讨DeepSeek R1的一个热门技术应用方向——模型微调。本节公开课我们将  重点介绍如何使用主流微调框架unsloth，围绕DeepSeek R1 Distill 7B和8B模型进行高效微调，并详细  介绍专门用于推理大模型高效微调的COT数据集的创建和使用方法，并在一个医学数据集上完成高效微   调实战，并最终达到问答风格优化+知识灌注目的，让模型在微调过程中掌握复杂医学问题的专业推理过 程，并提高疾病诊断的准确率。

* **硬件要求**：本节公开课最小化复现仅需7G显存、半小时运行时间即可完成，并获得微调效果。

* 训练流程迁移：本节公开课介绍的DeepSeek R1模型的高效微调流程可以迁移至DeepSeek R1任意 蒸馏模型、任意COT数据集，甚至是进行DeepSeek R1模型高效微调。

* **课件代码领取**：公开课随课提供全部课件、代码、训练数据、模型微调前后权重等各项内容。&#x20;

* **课程参考资料**：为了更好的辅助学习，随公开课附赠相关参考资料。

* 体验课内容节选自[《2025大模型Agent智能体开发实战》](https://whakv.xetslk.com/s/1tKbjV)完整版付费课程

  体验课时间有限，若想深度学习大模型技术，欢迎大家报名由我主讲的[《2025大模型Agent智能体开发实战》](https://whakv.xetslk.com/s/1tKbjV)：

![](images/image5.jpeg)

![](images/image.png)

此外，公开课全套学习资料，已上传至网盘（ https://pan.baidu.com/s/1LSCc4\_3V9bEofy\_luy\_1WA?pwd=gain ）

**需要更系统深入学习大模型可扫码⬆️添加助教咨询喔～**

***

* **公开课大纲**

**DeepSeekR1高效微调实战入门**

一、大模型微调技术快速入门

1.微调基础概念介绍

2.高效微调的应用场景

3.微调与强化学习训练、模型蒸馏等概念辨析

4.主流微调工具介绍

5.模型微调所需硬件与服务器环境搭建&#x20;

二、DeepSeekR1  Distill高效微调环境准备

1.unsloth安装部署 &#x20;

三、DeepSeek  R1模型微调代码实战

## **一、大模型微调技术快速入门**

### **1.** **微调基础概念介绍**

#### **1.1** **微调基本概念**

所谓大模型微调，指的在已有的大规模预训练模型基础上，通过对标注数据进行训练，进一步优化   模型的表现，以适应特定任务或场景的需求。不同于RAG或者Agent技术，通过搭建工作流来优化模型表

现，微调是通过修改模型参数来优化模型能力，是一种能够让模型“永久”掌握某种能力的方法。

#### **1.2** **全量微调与高效微调**

而从方法的大类上来划分，微调又可以划分为全量微调：带入全部数据进行微调，和高效微调：只 带入部分数据进行微调。毫无疑问，全量微调是一种算力消耗更大、但对模型的能力改造更为彻底的方 法，而高效微调则更类似一种“ 四两拨千斤”的方法，通过修改模型部分参数，来调整模型整体能力。

#### **1.3** **全量指令微调**

全量微调的一个最核心的应用场景就是全量指令微调，在当代大模型的训练流程中，首先需要进行 模型预训练，在这个过程中我们需要输入大量不带标签的文本段落，让模型学习到语言的基本结构、语 法规则、上下文信息等。经过预训练的模型就拥有了“成语接龙的能力”，如下所示：

![](images/image11.jpeg)

但目前在很多场景下，我们需要大模型进行对话（chat）而非补全（completion），因此我们需要围绕 经过预训练的模型进行进一步的全量指令微调，通过输入海量的对话文本（一问一答构成的文本，同时  也是有监督的文本），让模型学会对话的能力，例如：

![](images/image12.jpeg)

现在绝大多数开源模型，在开源的时候都会公布两个版本的模型，其一是Base模型，该模型只经过了预 训练，没有经过指令微调；其二则是Chat模型（或者就是不带尾缀的模型），则是在预训练模型基础上 进一步进行全量指令微调之后的对话模型：

![](images/image14.jpeg)

注1——全量指令微调是全量微调的一种；

注2——大模型训练公开课，详见：【全网独家】手动复现DeepSeekv3| [https://www.bilibili.co](https://www.bilibili.com/video/BV1KtwueYE54/) [m/video/BV1KtwueYE54/](https://www.bilibili.com/video/BV1KtwueYE54/)



#### **1.4** **高效微调与LoRA、** **QLoRA**

尽管全量微调可以对模型的能力进行深度改造，但要带入模型全部参数进行训练，需要消耗大量的 算力，且有一定的技术门槛。相比之下，在绝大多数场景中，如果我们只想提升模型某个具体领域的能  力，那高效微调会更加合适。尽管在2020年前后，深度学习领域诞生了很多高效微调的方法，但现在适 用于大模型的最主流的高效微调方法只有一种——LoRA。

LoRA（ Low-Rank Adaptation）微调是一种参数高效的微调方法，旨在通过引入低秩矩阵来减少微  调时需要调整的参数数量，从而显著降低显存和计算资源的消耗。具体来说，  LoRA 微调并不直接调整原 始模型的所有参数，而是通过在某些层中插入低秩的适配器（Adapter）层来进行训练。

**LoRA的原理：**

**LoRA的优势：**

1\. **显存优化：** 只需要调整少量的参数（适配器），显著减少了显存需求，适合显存有限的GPU。

2\. **计算效率：** 微调过程中的计算负担也更轻，因为减少了需要调整的参数量。

3\. **灵活性：** 可以与现有的预训练模型轻松结合使用，适用于多种任务，如文本生成、分类、问答等。

而**QLoRA（Quantized** **Low-Rank** **Adaptation）** 则是 LoRA 的一个扩展版本，它结合了 LoRA 的  低秩适配器和量化技术。 QLoRA 进一步优化了计算效率和存储需求，特别是在极端显存受限的环境下。

与 LoRA 不同的是， **QLoRA** **会将插入的低秩适配器层的部分权重进行量化（通常是量化为** **INT4** **或**  **INT8）** ，在保持性能的同时显著降低模型的存储和计算需求。

* **核心思想：** 在 LoRA  的基础上加入量化技术，减少权重表示的位数，从而降低显存和计算需求。 QLoRA 结合了低秩适配器和量化的优点，能够在显存有限的设备上进行更高效的微调。

* **量化：** 通过将模型权重量化为低精度（如 INT4），减少内存占用，并提高推理和训练速度。&#x20;

* 优势：

  * 在显存非常有限的情况下仍能进行微调。&#x20;

  * 可以处理更大规模的模型。

  * 适合用于边缘设备和需要低延迟推理的场景。

**LoRA** **与** **QLoRA** **二者对比如下**

| **特性**        | **LoRA**                 | **QLoRA**                                      |
| ------------- | ------------------------ | ---------------------------------------------- |
| **核心技** **术** | 低秩适配器（Low-Rank Adapters） | 低秩适配器 + 量化技术（Low-Rank Adapters + Quantization） |
| **适用场** **景** | 显存受限，但设备性能较好             | 极限显存受限或需要快速推理的设备                               |
| **计算效** **率** | 提高计算效率，减少调整的参 数数量        | 进一步提升效率，减少内存使用并加快推理速度                          |
| **量化技** **术** | 无量化                      | 将权重量化为低精度（如 INT4 或 INT8）                       |

| **特性**         | **LoRA**        | **QLoRA**                    |
| -------------- | --------------- | ---------------------------- |
| **内存消** **耗**  | 较低，但不如 QLoRA 低  | 显著降低内存消耗，适合更小的设备             |
| **训练复** **杂度** | 较简单，适用于大多数微调场 景 | 需要更多的量化和适配工作，但适合超大模型和设 备受限场景 |

微调原理参考：【入门】大语言模型常用微调框架介绍|[https://www.bilibili.com/video/BV1Yc41](https://www.bilibili.com/video/BV1Yc411g78a/) [1g78a/](https://www.bilibili.com/video/BV1Yc411g78a/)

![](images/image21.jpeg)

### **2.** **高效微调的应用场景**

在实际大模型应用场景中，高效微调主要用于以下四个方面：

* **对话风格微调**：高效微调可以用于根据特定需求调整模型的对话风格。例如，针对客服系统、虚拟 助理等场景，模型可以通过微调来适应不同的 **语气、礼貌程度** 或 **回答方式**，从而在与用户互动时  提供更符合要求的对话体验。通过微调少量的参数（例如对话生成的策略、情感表达等），可以使 模型表现出更具针对性和个性化的风格。

* **知识灌注**：知识灌注是指将外部知识或领域特定的信息快速集成到已有的预训练模型中。通过高效 微调，模型可以更好地学习新领域的专有知识，而无需重新从头开始训练。例如，对于法律、医疗 等专业领域，可以使用少量的标注数据对预训练模型进行微调，帮助模型理解特定行业的术语、规 则和知识，进而提升专业领域的问答能力。

* **推理能力提升**：高效微调还可以用于提升大模型的推理能力，尤其是在处理更复杂推理任务时。通 过微调，模型能够更加高效地理解长文本、推理隐含信息，或者从数据中提取逻辑关系，进而在多 轮推理任务中提供更准确的答案。这种微调方式可以帮助模型在解答复杂问题时，提高推理准确性 并减少错误。

* **Agent能力（Function** **calling能力）提升**：在多任务协作或功能调用场景中，高效微调能够显著   提升模型**Agent能力**，使得模型能够有效地与其他系统进行交互、调用外部API或执行特定任务。 通过针对性微调，模型可以学会更精准的功能调用策略、参数解析和操作指令，从而在自动化服     务、智能助手或机器人控制等领域表现得更加高效和智能。

### **3.** **微调与强化学习训练、模型蒸馏等概念辨析**

而伴随着DeepSeek R1的兴起，关于强化学习训练、模型蒸馏等概念也逐渐被人熟知，这里我们简   单总结下这三者的异同。 **微调**、 **强化学习训练** 和 **模型蒸馏** 都是常用的技术手段，它们有着不同的应用场景和目标。尽管这些方法在某些方面有所交集，但它们的核心原理和任务目标却存在显著差异。

#### **1.** **微调（Fine-tuning）：**

微调是指在一个已经预训练的大型模型基础上，使用较少的任务特定数据对模型进行再训练，以适应特 定任务的需求。微调通常针对模型的某些层进行调整，或者通过在全模型基础上进一步训练来优化其在 目标任务中的表现。微调不需要从零开始训练模型，而是通过 **小范围的参数调整** 来获得较高的任务表现。

* **目标**：通过少量的标注数据对预训练模型进行优化，适应具体任务（如文本分类、问答、生成 等）。

* **特点**：微调的计算量相对较小，能够在有限的数据和计算资源下提升模型在特定任务上的性能。

* **应用**：常用于下游任务如情感分析、机器翻译、推荐系统等。

#### **2.** **强化学习训练（Reinforcement** **Learning）：**

强化学习是一种通过与环境互动来学习如何最大化长期奖励的学习方式。与微调不同，强化学习是一个  **决策优化过程**，其主要目标是通过 **试错** 和反馈来学习最优策略。强化学习的智能体通过与环境的交互获 得奖励信号，并根据反馈调整策略，长期进行优化。

* **目标**：通过与环境的交互，学习最优的行为策略，最大化累积奖励。

* **特点**：强化学习强调 **动态决策**，通过 **探索和利用** 的平衡，优化策略。它通常不依赖于预定义的数 据集，而是依赖于与环境的持续交互。

* **应用**：强化学习在游戏AI（如AlphaGo）、机器人控制、自动驾驶等任务中有广泛应用。

#### **3.** **模型蒸馏（Model** **Distillation）：**

模型蒸馏是一种将 **复杂、计算密集型的教师模型** 的知识转移到 **小型、高效的学生模型** 上的技术。通过 蒸馏，学生模型能够学习教师模型的决策过程或表示，从而在保留较高效能的同时，降低模型的计算和 存储成本。蒸馏通常通过教师模型生成软标签或行为模仿来指导学生模型训练。

* **目标**：通过教师模型的“知识转移” ，帮助学生模型提升性能，特别是计算能力有限的设备上。

* **特点**：蒸馏的核心在于知识的迁移，尤其是在模型压缩和部署方面的优势。学生模型通常在性能上 能接近教师模型，但参数量更小，计算更高效。

* **应用**：常见于模型压缩、边缘计算、低功耗设备的部署中，用于提升部署效率并降低计算需求。

**三者的异同**

| **特征**        | **微调（Fine-tuning）**           | **强化学习训练****（Reinforcement** **Learning）** | **模型蒸馏（Model** **Distillation）** |
| ------------- | ----------------------------- | ------------------------------------------ | -------------------------------- |
| **目标**        | 优化已预训练模型在特 定任务上的表现            | 学习最优行为策略，最大化 长期奖励                          | 将复杂模型的知识转移到更 小的模型上，减少计算开销        |
| **数据** **依赖** | 依赖于标注数据，通常 需要针对具体任务的少 量数据进行训练 | 依赖于与环境的交互，智能 体从奖励信号中学习                     | 依赖于教师模型，学生模型 通过模仿教师模型的行为来 学习     |

| **特征**              | **微调（Fine-tuning）**          | **强化学习训练****（Reinforcement** **Learning）** | **模型蒸馏（Model** **Distillation）** |
| ------------------- | ---------------------------- | ------------------------------------------ | -------------------------------- |
| **训练** **方式**       | 通过微调已预训练的模 型参数来适应新的任务        | 通过试错学习，智能体与环 境交互，优化决策过程                    | 教师模型生成软标签或行为，学生模型通过模仿学习 教师模型的行为  |
| **应用** **场景**       | 特定任务（如文本分 类、情感分析、翻译 等）       | 游戏AI、机器人控制、自动 驾驶、策略优化                      | 模型压缩、边缘计算、低资 源设备上部署              |
| **计算** **复杂** **度** | 相对较低，计算量通常 较小，但依赖于任务规 模      | 较高，需要大量的环境交互 和计算资源                         | 较低，通过蒸馏学生模型来 减少计算资源和存储需求         |
| **反馈** **机制**       | 基于标注数据的监督学 习，通常通过计算损失 函数进行优化 | 基于环境反馈的强化学习， 通过奖励信号进行优化决策                  | 通过教师模型的行为或预测 结果来为学生模型提供指导        |

需要注意的是， **模型微调** 和 **强化学习训练** 都可以作为 **模型蒸馏** 的一个环节或技术实现手段，它们并不 互相排斥，反而在某些情况下能够互相补充，结合起来达到更好的效果。

DeepSeek R1模型蒸馏流程：借助DeepSeek R1进行模型蒸馏，模型蒸馏入门实战！  |[https://ww](https://www.bilibili.com/video/BV1X1FoeBEgW/) [w.bilibili.com/video/BV1X1FoeBEgW/](https://www.bilibili.com/video/BV1X1FoeBEgW/)

![](images/image35.jpeg)

### **4.** **主流微调工具介绍**

在入手学习大模型微调时，首先推荐功能层次封装层次较高的微调三剑客：  unsloth、 Llama-

Factory和ms-SWIFT。除此之外，也可以借助更加底层的库，如peft、 LoRA、 transformer等实现高效   微调。对于初学者来说，首先推荐unsloth、 Llama-Factory和ms-SWIFT来进行微调，三种工具基本说明 如下。

#### **4.1** **unsloth**

![](images/image36.jpeg)

* unsloth GitHub主页：<https://github.com/unslothai/unsloth>

unsloth 是一个专为大型语言模型（LLM）设计的微调框架，旨在提高微调效率并减少显存占用。 它通过手动推导计算密集型数学步骤并手写 GPU  内核，实现了无需硬件更改即可显著加快训练速度。

![](images/image37.jpeg)

unsloth 与 HuggingFace 生态兼容，可以很容易地transformers、 peft、 trl 等库结合，以实现模型 的监督微调（SFT）和直接偏好优化（DPO），仅需模型的加载方式，无需对现有训练代码进行修改。

**主要功能点：**

* **高效微调：** unsloth 通过深度优化，使 LLM 的微调速度提高 2-5 倍，显存使用量减少约 80%，且 准确度无明显下降。

* **广泛的模型支持：** 目前支持的模型包括目前各类主流模型，用户可以根据需求适合的模型进行微调。

* **兼容性：** unsloth 与 HuggingFace态系统兼容，用户可以轻松将其与 traformers、 peft、 l 等库结  合，实现模型的监督微调（SFT）和直接偏好优化（DPO），仅需修改模型的加载方式，无需对现   有训练代码进行过多修改。 **内存优化：** 通过 4 位和 16 位的 QLoRA/LoRA 微调， unsloth 显著了显 存占用，使得在资源受限的环境中也能大的微调。

**unsloth核心优势：**

* **显著提升微调效率：** 相比传统方法， Unsloth 能够在更短的时间内完成微调任务，节省时间成本。

* **降低硬件要求：** 通过优化显存使用，用户可以在显存较小的 GPU 上进行大模型的微调，降低了硬 件门槛。

* **开源免费：** Unsloth 提供开源版本，用户可以在 Google Colab 或 Kaggle Notebooks 上免费试 用，方便上手体验。

总的来说， unsloth 为大型语言模型的微调提供了高效、低成本的解决方案，适合希望在有限资源下进行

模型微调的开发者和研究人员。

#### **4.2** **LLama-Factory**

![](images/image42.jpeg)

* LLama-Factory GitHub主页：<https://github.com/hiyouga/LLaMA-Factory>

LLaMA-Factory 是一个统一且高效的微调框架，旨在为超过 100 种大型语言模型（LLMs）和视觉 语言模型（VLMs）提供便捷的微调支持。 用户能够灵活地定制模型以适应各种下游任务。

**主要功能和特点：**

* **广型支持：** LLaMA-Factory 支持对 100 多LLMs 和 VLMs 进行微调，包括最新的模型版本，如 Llama 3、 GLM-4、 Mistral Small、 PaliGemma2 等。

* **高效的微调方法：** 框架集成了多nk Adaptation）、 QRA（Quantized LoRA）等，以提高训练速度 并减少显存占用。

* **多模态任务支持：** 除了传统的文本任务外， LLaMA-Factory 还支频识别、音频理解等多种任务类 型。

* **实验监控：** 提供了丰富的实验监控工具，如 LlamaBoard、TensorBoard、Wandb、 MLflow、练 过程。

* **快速：** 框架提供了类似 OpenAI 风格的 API、 Gradio UI 和命令行界面，并结合 vLLM worker，实 现了高效的推理能力。

**4.3** **ms-SWIFT**

![](images/image46.jpeg)

* ms-SWIFT GitHub项目主页： <https://github.com/modelscope/swift>

ms-swift（ Scalable lightWeight Infrastructure for Fine-Tuning）是由魔搭社区（ModelScope）  开发的高效微调和部署框架，旨在为研究人员和开发者提供一站式的大模型与多模态大模型的训练、推  理、评测、量化和部署解决方案。  的模型支持： \*\* ms-swift 支持超过 450 种大型模型（LLMs）和 150 多种多模态大模型（MLLMs）的训练和部署\*\* ，包括最新的模型版本，如 Qwen2.5、 InternLM3、GLM4、 Llama3.3、 Mistral、 DeepSeek-R1、Yi1.5、 Baichuan2、 Gemma2 等，以及多模态模型如 Qwen2.5-VL、 Qwen2-Audio、 Llama3.2-Vision、 Llava、 InternVL2.5 等。

* **多样化的训练技术：** 框架集oRA、 Llama-Pro、 LonoRA、 GaLore、 Q-GaLore、 LoRA+、 LISA、 DoRA、 FourierFt、 ReFT、 UnSloth 和 Liger 等，满足不同的微调需求。

* **轻量级微调：** 支持多种轻量级微调方法，如 LoRA、 QLoRA、 DoLLaMAPro、Adapt、 GaLore、 Q- Galore、 LISA、 UnSloth、 Liger-Kernel 等，降低显存和计算资源的消耗。

* **分布式训练：** 支持分布式数据并行（DDP）、 DeepSpeed ZeRO2/ZeRO3、 FSDP 等技术，提升推 理加速： \*\* 提供 BNBWQ、 GPTQ、AQLM、 HQQ、 EETQ 等量化方法，并支持使用 vLLM 和LMDeploy 对推理、评测和部署 支持图像、视频和语音等多种模态型训练，涵盖 VQA、 Caption、 OCR、 Grounding 等任务。

* **用户友好的界面：** 提供基于 Gradio 的 We和量化操作，简化了大模型的全链路流程。

| 注一：若是强化学习训练，则推荐veRL和OpenRLHF等框架。注二：其他更底层微调框架推荐 |              |              |                  |
| ---------------------------------------------- | ------------ | ------------ | ---------------- |
|                                                | **框架**       | **优势**       | **适用场景**         |
|                                                | Hugging Face | 高度兼容，易用，文档丰富 | 一般 NLP 任务，模型选择丰富 |
|                                                |              |              |                  |

| **框架**      | **优势**             | **适用场景**                   |
| ----------- | ------------------ | -------------------------- |
| LoRA        | 显存节省，减少微调计算量       | 显存有限的设备，微调大规模模型            |
| PEFT        | 高效微调，低计算开销         | 资源有限的环境，适合大规模预训练模 型的微调     |
| DeepSpeed   | 大规模分布式训练，显存优化      | 超大规模训练，多卡分布式训练             |
| AdapterHub  | 低资源消耗，快速微调         | 多任务微调，资源有限的环境              |
| Alpaca-LoRA | 生成任务优化， LoRA 技术结 合 | 对话生成、文本生成                  |
| FastChat    | 对话系统微调，快速集成        | 对话生成任务，尤其是对 ChatGPT 等模 型微调 |
| FairScale   | 大规模分布式训练优化，自动 化优化  | 多卡分布式训练，大规模微调              |

### **5.模型微调所需硬件与服务器环境搭建**

大模型微调属于大模型进阶类技术，不同于普通的模型对话或搭建基础应用，微调往往需要一定的 软硬件条件支持。

* &#x20; 大模型微调所需硬件一览

硬件方面，不同尺寸模型、不同精度微调时所需显存如下：

![](images/image52.jpeg)

接下来我们将以8B模型为例进行高效微调，最低INT4情况下仅需7G显存即可运行。

更多大模型基础硬件配置要求，详见公开课《【2025版】大模型GPU硬件配置保姆级指南》 |[http](https://www.bilibili.com/video/BV1VFqZYSEoj) [s://www.bilibili.com/video/BV1VFqZYSEoj](https://www.bilibili.com/video/BV1VFqZYSEoj)

* 操作系统选择

而操作系统方面，由于绝大多数工业场景下微调会涉及多卡微调，目前只有Linux系统对

DeepSpeed和其他多卡并行加速库支持较好，因此绝大多数工业场景下都会使用Ubuntu操作系统或 CentOS操作系统。本节公开课我们以Ubuntu系统为例来进行高效微调。

若无相关软件环境，本节公开课的相关代码也可以在Windows下运行（本节微调示例不涉及多卡并 行）。但若想体验更加真实的工业场景下的微调流程，也可以考虑在AutoDL上租赁显卡并配置Ubuntu   服务器来完成操作。最小化实现微调效果，仅需单卡3090运行两小时即可得到结果，仅需不到5元即可  完成训练：

![](images/image55.jpeg)

AutoDL相关操作详见公开课：《AutoDL快速入门与GPU租赁使用指南》 |[https://www.bilibili.com/vide](https://www.bilibili.com/video/BV1bxB7YYEST/) [o/BV1bxB7YYEST/](https://www.bilibili.com/video/BV1bxB7YYEST/)

![](images/image56.jpeg)

更多大模型微调技术介绍，欢迎报名由我主讲的《2025大模型Agent智能体开发实战》（2月DeepSeek 强化班）<https://whakv.xetslk.com/s/1tKbjV> 进行更深度系统的学习哦\~

![](images/image57.jpeg)

## **二、** **DeepSeek** **R1** **Distill高效微调环境准备**

### **1.** **unsloth安装部署**

unsloth是推理、微调一体式框架，可以使用如下命令快速安装：

| pip install unslothpip install --force-reinstall --no-cache-dir --no-deps git+<https://github.com/unslothai/unsloth.git> |
| ------------------------------------------------------------------------------------------------------------------------ |

![](images/image62.jpeg)

注意，这里的第二行、第三行是一条命令，只是换行了而已，需要复制在一起然后运行

而如果是使用AutoDL且Github网速不稳，则可以使用如下命令开启AutoDL学术加速：

| source /etc/network\_turbo |
| -------------------------- |

然后再使用上面的安装命令进行安装。

AutoDL学术资源加速说明： <https://www.autodl.com/docs/network_turbo/>

### **2.wandb安装与注册**

#### **2.1** **wandb基本说明**

在大规模模型训练中，我们往往需要监控和分析大量的训练数据，而WandB可以帮助我们实现这一 目标。它提供了以下几个重要的功能：

**实时可视化**：WandB可以实时展示训练过程中关键指标的变化，如损失函数、学习率、训练时间 等。通过这些可视化数据，我们能够直观地了解模型的训练进展，快速发现训练中的异常或瓶颈。

**自动记录与日志管理**：WandB会自动记录每次实验的参数、代码、输出结果，确保实验结果的可追 溯性。无论是超参数的设置，还是模型的架构调整，  WandB都能够帮助我们完整保留实验记录，方便后 期对比与调优。

**支持中断与恢复训练**：在长时间的预训练任务中，系统中断或需要暂停是常见的情况。通过WandB 的checkpoint功能，我们可以随时恢复训练，从上次中断的地方继续进行，避免数据和时间的浪费。

**多实验对比**：当我们尝试不同的模型配置或超参数时， WandB允许我们在多个实验之间轻松进行对 比分析，帮助我们选择最优的模型配置。

**团队协作**：WandB还支持团队协作，多个成员可以共同查看实验结果，协同调试模型。这对研究和 项目开发中团队的合作非常有帮助。

#### **2.2** **wandb注册与使用**

<https://wandb.ai/site>

![](images/image64.png)

![](images/image65.png)

![](images/image66.jpeg)

然后即可在令行中输入如下代码安装wandb：

| pip install wandb |
| ----------------- |

![](images/image67.jpeg)

接下来在unsloth微调前，我们即可设置wandb进行微调记录，并可在对应网站上观察到训练过程如下：



![](images/image68.jpeg)

### **3.DeepSeek** **R1模型下载**

本次实验我们将同时介绍DeepSeek R1 Qwen蒸馏模型和LLama蒸馏模型的高效微调流程，因此需 要提前进行模型下载。考虑到国内网络环境，我们可以使用ModelScope进行下载，这里首先需要安装  ModelScope：

| pip install modelscope |
| ---------------------- |

#### **3.1** **DeepSeek** **R1** **Distill** **Qwen** **7B模型下载**

ModelScope模型地址： [https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwe](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) [n-7B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)

![](images/image69.jpeg)

然后创建DeepSeek-R1-Distill-Qwen-7B文件夹，用于保存模型权重：

| mkdir ./DeepSeek-R1-Distill-Qwen-7B |
| ----------------------------------- |

然后即可使用如下命令进行模型下载：

| modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local\_dir ./DeepSeek-R1-Distill-Qwen-7B |
| -------------------------------------------------------------------------------------------------------------- |

![](images/image70.jpeg)

下载完成后模型完整项目如下：

![](images/image71.jpeg)

#### **3.2** **DeepSeek** **R1** **Distill** **LLama** **8B模型下载**

ModelScope模型地址： [https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llam](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) [a-8B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)

![](images/image72.jpeg)

然后创建DeepSeek-R1-Distill-Llama-8B文件夹，用于保存模型权重：

| mkdir ./DeepSeek-R1-Distill-Llama-8B |
| ------------------------------------ |

然后即可使用如下命令进行模型下载：

| modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --local\_dir ./DeepSeek-R1-Distill-Llama-8B |
| ---------------------------------------------------------------------------------------------------------------- |

下载完成后模型完整项目如下：

![](images/image73.jpeg)

两个模型下载完成后文件结构如下所示：

![](images/image74.jpeg)

此外，也可以在课件网盘中下载模型权重：

![](images/image75.jpeg)

![](images/image-1.png)

扫码联系⬆️助教即可领取～

***

### **4.推理模型微调数据集下载**

#### **4.1** **DeepSeek** **R1模型组回复结构与微调数据集结构要求**

DeepSeek R1及其蒸馏模型，推理过程的具体体现就是在回复内容中，会同时包含推理部分内容和 最终回复部分内容，并且其推理部分内容会通过（一种在模型训练过程中注入的特殊标记）来进行区

分。

![](images/image76.jpeg)

也就是说， DeepSeek R1模型组的回复格式是一种非常特殊的格式，即包含think部分内容，也包含 response部分内容。

因此，在围绕DeepSeek R1模型组进行微调的时候，微调数据集的回复部分文本也需要是包含推理 和最终回复两部分内容，才能使得DeepSeek R1模型组在保持既定回复风格的同时，强化模型能力，反 之则会导致指令消融问题（模型回复不再包含think部分）。

此时think部分和最终回复，共同构成有监督微调的标签。

而这种同时包含思考和结果的数据集，在推理模型大行其道的当下也并不少见，例如非常著名的数  学问答数据集NuminaMath CoT，就同时包含数学问题、问题的解题思路（也就是think部分）和问题最 终的答案。而该数据集也是可以用于推理模型微调的数据集。

![](images/image79.png)

NuminaMath CoT数据集地址： <https://huggingface.co/datasets/AI-MO/NuminaMath-CoT>

除了NuminaMath CoT数据集外，还有APPs（编程数据集）、 TACO（编程数据集）、

long\_form\_thought\_data\_5k （通用问答数据集）等，都是CoT数据集，均可用于推理模型微调。 相关数据集介绍，详见公开课：《借助DeepSeek R1进行模型蒸馏，模型蒸馏入门实战！》  |[http](https://www.bilibili.com/video/BV1X1FoeBEgW/) [s://www.bilibili.com/video/BV1X1FoeBEgW/](https://www.bilibili.com/video/BV1X1FoeBEgW/)

![](images/image80.jpeg)

#### **4.2** **medical-o1-reasoning-SFT数据集介绍**

本节公开课，我们选取24年12月31号最新发布的一个包含推理过程的医学数据集：由深圳大数据研 究院发布的HuatuoGPT-o1模型的微调数据集——medical-o1-reasoning-SFT。

![](images/image81.jpeg)

数据集总共包含25400条数据，均为医学领域疾病诊断数据集，且不乏一些疑难杂症的推理和判

断，数据集整体质量较高，推理过程严谨准确，非常适合进行医疗领域模型微调，可以极大程度提高模 型对于病理的推理过程，并在这个过程中完成一些医疗知识的灌注。

例如一种一条数据集内容如下：

> 一位45岁的男性，有饮酒史，过去10年一直戒酒，现因突然出现构音困难、步态蹒跚和意向 性震颤就诊。根据这一临床表现和病史，最可能的诊断是什么？

Complex\_CoT：Alright, let’s break this down. We have a 45-year-old man here, who suddenly  starts showing some pretty specific symptoms: dysarthria, shuffling gait, and those intention   tremors. This suggests something's going wrong with motor control, probably involving the       cerebellum or its connections. Now, what's intriguing is that he's had a history of alcohol use,  but he's been off it for the past 10 years. Alcohol can do a number on the cerebellum, leading to degeneration,and apparently, the effects can hang around or even appear long after one     stops drinking. At first glance, these symptoms look like they could be some kind of chronic       degeneration,maybe something like alcoholic cerebellar degeneration, but hold on. This looks different. The symptoms just came on all of a sudden. Chronic degenerations typically    have a more gradual onset. Okay, let’s reconsider this sudden nature. It’s making me think of something more acute, more rapid onset. Hmm, if we dig back into his history of drinking,there might have been some damagedone, leading to nutritional deficiencies, like a lack of      thiamine. Wernicke’s encephalopathy is a classic possibility here. That condition pops up due to thiamine deficiency, often after a history of alcohol use. It’s known for causing ataxia,

confusion, and eye movement issues. However, he’s not showing the full triad of symptoms;   there's no mention of confusion or eye problems, so maybe it doesn’t fit perfectly. Oh, wait a second, maybe we're missing something simpler. Given the acute nature of the symptoms,maybe this is more inline with something like a stroke. Sudden onset can definitely suggest a vascular event. With his alcohol history, he’s at risk for things like high blood pressure, which    can increase stroke risk. In particular, lacunar strokes can mess with motor coordination,speech, the works. These are small, but significant enough to cause these exact symptoms: dysarthria, tremors, and a shuffling gait. But hang on, what if there’s another angle we’re missing? Chronic alcohol effects might still play a role here, justin a different way. There’s a condition called acquired hepatocerebral degeneration. This can happen in people with a history of liver disease due to alcohol, even years after they’ve quit. It can result in symptoms like these because of how substances get deposited in the brain. Linking this back to our patient, his history with alcohol could’ve led to some liver issues, and now, here we are with these symptoms showing up suddenly. Makes sense, right? So, considering everything,

acquired hepatocerebral degeneration seems to fit quite nicely. That’s probably our most fitting diagnosis for this situation.

> 好的，让我们一步一步分析这个问题。我们有一位45岁的男性，突然出现了几个相当具体的 症状：构音困难、步态蹒跚和意向性震颤。这提示着运动控制可能出现了问题，很可能是小 脑或其连接受到了影响。现在有趣的是，他有饮酒史，但过去10年一直戒酒。酒精对小脑有 很大的影响，可能导致退化，而且显然这种影响可能会持续很长时间，甚至在戒酒后仍然会 出现。乍一看，这些症状看起来像是某种慢性退行性病变，也许像是酒精性小脑退行性病变，但稍等一下，这看起来有所不同。症状突然出现。慢性退行性病变通常是逐渐开始的。 好吧，让我们重新考虑一下这种突然的性质。这让我想到一些更急性的、起病迅速的疾病。 嗯，如果我们回顾他的饮酒史，可能存在一些损伤，导致营养缺乏，比如缺乏硫胺素。维尔 尼克脑病就是一个经典的可能性。这个病状是由于硫胺素缺乏引起的，通常发生在有饮酒史 的人群中。它的典型症状包括共济失调、意识混乱和眼球运动问题。然而，他并没有表现出 完整的三联症；没有提到混乱或眼部问题，所以这可能不完全符合。哦，等一下，也许我们 漏掉了更简单的情况。考虑到症状的急性性质，也许这更像是中风。突然发作肯定可以提示 血管事件。考虑到他的饮酒史，他有高血压的风险，这会增加中风的风险。特别是，腔隙性 中风可能会影响运动协调、言语等。这些小的中风虽然很小，但足够显著，能引发这些确切 的症状：构音困难、震颤和蹒跚步态。但等等，是否还有其他角度我们没考虑到？慢性酒精 影响可能仍然起着作用，只是以不同的方式。有一种叫做获得性肝脑病的疾病。这种情况可 能发生在有酒精引起的肝病史的人身上，即使他们已经戒酒多年。它可能会导致类似的症状，因为体内的物质沉积在大脑中。回到我们的患者，他的酒精史可能导致了一些肝脏问题，而现在这些症状突然出现了。这是有道理的，对吧？所以，综合考虑所有因素，获得性 肝脑病似乎非常符合。这个诊断可能是我们最合适的结论。

.  Response ：Considering the clinical presentation of sudden onset dysarthria, shuffling gait, and intention tremors in a 45-year-old man with a history of alcohol use who has been abstinent for the past 10 years, the most likely diagnosis is acquired hepatocerebral degeneration. This condition is associated with chronic liver disease, which can often be a consequence of long-term alcohol use. Despite the patient's abstinence from alcohol for a decade, previous alcohol use may have led to underlying liver dysfunction. This dysfunction, even if subclinical, can cause encephalopathy due to the accumulation of neurotoxic substances that affect the brain. The sudden onset of these neurological symptoms aligns with how acquired hepatocerebral degeneration can manifest, making it a probable diagnosis in this scenario.

考虑到这位45岁男性突发构音困难、蹒跚步态和意向性震颤的临床表现，以及他有饮酒史且 过去10年未再饮酒，最可能的诊断是获得性肝脑病。该病与慢性肝病相关，而慢性肝病通常 是长期饮酒的后果。尽管患者已经戒酒十年，但之前的饮酒可能导致了潜在的肝功能障碍。  即使这些障碍是亚临床的，它也能通过神经毒性物质的积聚引起脑病。症状的突然出现与获 得性肝脑病的表现方式相符，因此它是最符合的诊断。

## **三、** **DeepSeek** **R1模型微调代码实战**                            &#x20;

本部分见Jupyter文件。

> 此部分内容已经转为在线文档导入社区➡️请点击🔗：[ DeepSeek R1 Distill高效微调入门实战](https://kq4b3vgg5b.feishu.cn/wiki/WWzZwHVNei8zsWkieJ3cyHzWnFe?from=from_copylink)

