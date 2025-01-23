## <center> 《2025大模型原理与训练实战》体验课
## <center> Day 1.DeepSeek v3 深度原理与架构精讲

基础配置——

```
torch==2.4.1
triton==3.0.0
transformers==4.46.3
safetensors==0.4.5
```

背景脚本、可在助教处回复【大模型原理】进行领取——

<center><img src="https://skojiangdoc.oss-cn-beijing.aliyuncs.com/2024LLM/training/74.png" alt="image-20250107200502389" style="zoom:100%;" />

今日我们为大家中点解读的是上述代码中的model.py代码。


```python
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
```

## 1 DeepSeekv3架构图解与基本参数配置

根据DeepSeekv3开源架构以及技术报告，我们绘制了DeepSeekv3基本架构图——

<center><img src="https://skojiangdoc.oss-cn-beijing.aliyuncs.com/2024LLM/training/72-.png" alt="image-20250107200502389" style="zoom:50%;" />

整体架构明显是一个类llama的decoder-only架构、具体架构板块如下——

- **1. 输入嵌入层**：模型以 **ParallelEmbedding**（并行嵌入层）作为起点，将输入的 token 映射到高维向量空间。
<br><br>
- **2. 旋转位置编码 (Rotary Positional Encodings, RoPE)**：使用 **RoPE** 为输入嵌入添加位置信息，使模型能够有效地捕捉序列中 token 的顺序关系。
<br><br>
- **3. RMSNorm 层归一化**：每个注意力层和前馈层前均使用 **RMSNorm**（均方根归一化）进行规范化。
<br><br>
- **4. 多头注意力机制 (Multi-Head Attention) 与 KV 缓存**：这是带有KV缓存加速的多头注意力机制、提供两种缓存策略：<br><br>
    1. **Naive Cache**（简单缓存）：适合较短序列或常规场景。
    2. **Absorb Cache**（密集缓存）：优化内存使用和推理效率，特别适用于长序列处理。
<br><br>
- **5. 前馈网络 (FFN) 和混合专家模型 (MoE)**：在注意力模块之后，模型在以下两种模块之间进行 **互斥选择**：<br><br>
  1. **FFN（前馈网络）**：使用 **SiLU（Sigmoid Linear Unit）** 作为激活函数，结合并行计算实现高效特征转换。
  2. **MoE（专家混合模型）**：通过门控机制动态路由输入到部分专家网络。这种设计在不增加显著计算成本的前提下，提升了模型的参数效率和灵活性。
<br><br>
- **6. 堆叠的 Transformer Blocks**：模型由多个 **Transformer Blocks** 堆叠而成（图中标注为 *N）。每个 Transformer 块包含注意力模块、RMSNorm 层以及 FFN 或 MoE。
<br><br>
- **7. 输出层**：堆叠的 Transformer 块输出经过、使用**RMSNorm** 再次进行规范化、**ColumnParallelLinear**（列并行线性层）将特征映射到词汇表的概率分布、**Softmax** 层计算最终的概率分布。

在开源时，DeepSeekv3提供了**三种不同规模的模型**——

| **参数名称**         | **含义描述**                                | **16B**           | **236B**           | **671B**           |
|----------------------|---------------------------------------------|-------------------|--------------------|--------------------|
| **vocab_size**       | 词汇表大小，支持的最大 token 数量            | 102400            | 102400             | 129280             |
| <font color="red">**dim**              | 词向量维度，每个 token 的表示维度              | 2048              | 5120               | 7168               |
| <font color="red">**inter_dim**        | FFN 的中间层维度                            | 10944             | 12288              | 18432              |
| <font color="red">**moe_inter_dim**    | MoE 模型中专家的中间层维度                  | 1408              | 1536               | 2048               |
| <font color="red">**n_layers**         | Transformer 块的层数                        | 27                | 60                 | 61                 |
| **n_dense_layers**   | 使用 FFN 的 Transformer 块数量               | 1                 | 1                  | 3                  |
| <font color="red">**n_heads**          | 注意力头的数量                              | 16                | 128                | 128                |
| <font color="red">**n_routed_experts** | MoE 模型中所有专家的总数量                  | 64                | 160                | 256                |
| **n_shared_experts** | MoE 模型中共享专家的数量                    | 2                 | 2                  | 1                  |
| <font color="red">**n_activated_experts** | 每个输入激活的专家数量                   | 6                 | 6                  | 8                  |
| **n_expert_groups**  | 专家分组的数量                              | -                 | 8                  | 8                  |
| **n_limited_groups** | 路由到的分组数量限制                        | -                 | 3                  | 4                  |
| **route_scale**      | 专家路由得分的缩放因子                      | 1.0               | 16.0               | 2.5                |
| **q_lora_rank**      | Q 投影的低秩分解维度                        | 0                 | 1536               | 1536               |
| **kv_lora_rank**     | KV 投影的低秩分解维度                       | 512               | 512                | 512                |
| **qk_nope_head_dim** | 无位置嵌入的 Query/Key 头维度               | 128               | 128                | 128                |
| **qk_rope_head_dim** | 使用旋转位置编码的 Query/Key 头维度          | 64                | 64                 | 64                 |
| **v_head_dim**       | Value 投影的头维度                          | 128               | 128                | 128                |
| **mscale**           | RoPE 编码的缩放因子，用于扩展序列长度        | 0.707             | -                  | -                  |
| **score_func**       | 专家路由的打分函数类型（如 softmax/sigmoid）  | -                 | -                  | sigmoid            |
| **dtype**            | 数据类型（如 bf16、fp8）                     | -                 | -                  | fp8                |


在整个模型代码的最初进行参数配置可以使关键超参数显而易见，方便开发者快速了解模型的整体设置，而无需额外打开配置文件。这种方式对于调试或学习模型代码尤为重要。一些参数（如 `world_size`、`rank` 等分布式相关配置）需要与具体代码逻辑或硬件环境动态结合，因此直接在代码中定义更加直观，也能避免额外的动态加载过程。

- **理解分布式技术中的基础概念速览**

想象一下你现在要从东南亚运输车厘子到北京、假设你有10吨车厘子、你可以指定一辆超大货车帮你运送所有车厘子到北京，同时你也可以选择——

1. 一辆车运输太慢、所以你选择了一个具有16辆小型车的车队、将车厘子分成了16个批次、给16两车同时运输、这样可以将你的整体的运输速度提升16倍。

2. 你发现、一个车队运输太慢、所以你选择了10个车队同步运输、每个车队包含16辆车、你将车厘子分成了160个批次、给到160辆车同步运输。

3. 然而你发现、虽然你有10个车队、但是10个车队都挤到了同一条道路上、运行很缓慢、于是你让10个车队选择了10支不同的道路行驶、分别将车厘子运送到北京。由于不同共享道路、因此运输的速度又进一步提升了、最终你的运输速度提升了160倍。

在这个过程里——
- 运输10吨车厘子就是你的一个具体程序任务<br><br>
- 车厘子就是你的数据、将车厘子分成16批这就是“数据分片”<br><br>
- 用16辆车的车队同步运输就是在同一个任务下启用了**多线程**（16个线程并行）<br><br>
- 从1个车队拓展到10个车队，就是**同时运行多个进程**（10个进程并行）<br><br>
- 将10个车队分配到10条不同的道路上，就是**为每个进程分配不同的节点**、即分配不同的GPU，能找到几条道路、就是你有几张显卡。道路的上限（资源的上限），也就是你运输速度/程序执行速度的上限。

多个车队分担了整个运输任务，形象地说明了 分布式计算的基本思想、即**通过增加计算单元来提高任务处理效率**、就像你通过增加道路、增加车队、增加每个车队里货车的数量来提升运输效率一样。

在这个过程中、一条路上可以有多个车队、所以一个节点里可以有多个进程。，就像在进程中、进程与进程之间互相独立、无法共享资源

- **节点** 是分布式系统中的物理或虚拟计算单元，通常对应一台物理服务器或虚拟机（如云端实例）。每个节点具有独立的硬件资源（如 CPU、GPU、内存等），是分布式计算的基础单位。

- **进程** 是在节点上运行的独立任务实例，负责执行分布式任务的一部分。理论上一个节点可以有多个独立的进程，但是大部分时候为了避免拥挤，**我们通常在一张 GPU 上分配一个独立的进程**，而一台服务器可能包含多张 GPU，因此可以运行多个进程。进程之间彼此独立，拥有各自的内存空间和系统资源。类比到车队、就像每个车队都有自己的车队首领和沟通方式、但是车队具体的司机和司机之间可能不熟悉。

- **线程** 是进程内部的执行单元，用于提升进程内部的并行效率。一个进程中可以启动多个线程，这些线程共享进程的内存和资源，但有独立的执行栈和寄存器状态。线程间通信方便，适用于需要快速协作的场景。类比到车队，线程就是一个车队内部的一辆单一的车、由于所有线程（所有车）都属于同一个车队、所以内部交换资源很容易。

在分布式系统中，节点、进程和线程 通过 层级结构（节点 > 进程 > 线程）和 协作机制（如网络通信、内存共享）共同作用，高效地完成大规模的计算任务。

```python
# 定义初始配置变量
world_size = 1  # 分布式训练的世界大小，默认为单机
rank = 0  # 当前进程的 rank，默认为主进程
block_size = 128  # 块大小，用于某些量化和计算优化
gemm_impl: Literal["bf16", "fp8"] = "bf16"  # GEMM（通用矩阵乘法）的实现方式，可选 "bf16" 或 "fp8"
attn_impl: Literal["naive", "absorb"] = "absorb"  # 注意力机制实现方式，可选 "naive" 或 "absorb"

**1. `world_size = 1`**：`world_size` 表示分布式训练的 **世界大小**，即并行运行的进程总数、可比喻为“车队总数”，由于一张GPU上通常只有一个进程、所以这个车队总数也通常对应为道路总数、即有多少个独立的计算单元（如 GPU）。
  - 如果 `world_size = 1`，说明系统中只有一个进程运行，相当于只有一个车队在运输车厘子。
  - 如果 `world_size = 10`，表示有 10 个独立的进程在运行，相当于有 10 个车队同时运输车厘子。

- **深度解释**：
  - 在多 GPU 或多节点环境中，`world_size` 决定了整个分布式任务的规模。增加 `world_size` 相当于增加车队数量，能够并行完成更多任务。
  - 当 `world_size = 1` 时，相当于只有一个车队，这种情况下任务是单机单 GPU 执行，性能提升有限。

---

**2. `rank = 0`**：可被比喻为当前车队的编号、`rank` 表示当前进程的唯一编号，用于标识当前进程在整个分布式系统中的位置。
  - 如果有 10 个车队（`world_size = 10`），`rank` 就是每个车队的编号，从 `0` 到 `9`。
  - `rank = 0` 通常表示主车队，负责调度任务（类似于主进程）。

- **深度解释**：
  - 分布式系统中，每个进程都有自己的 `rank`，用于分配任务或数据。例如：
    - `rank 0` 负责处理车厘子的第 1 批。
    - `rank 1` 负责处理车厘子的第 2 批。
    - 依此类推，直到分配完所有任务。
  - 通过 `rank`，系统可以确保任务和数据不会重复或遗漏。

---

**3. `block_size = 128`**：可比喻为单辆卡车的装载能力。
  - `block_size` 是分布式计算中定义的块大小，用于量化、分片或优化计算的基本单位。
  - 它可以理解为每辆车一次运输的货物量（即车队中每辆卡车的装载能力）。

- **深度解释**：
  - 在分布式训练中，大多数计算（如矩阵乘法、量化）需要将任务划分为小块来优化处理。
  - 比如：
    - 如果总任务是 10 吨车厘子（数据矩阵大小为 1024×1024），而 `block_size = 128`，那么会将数据划分为 128×128 的小块进行逐步计算或传输。
    - 小块越小（`block_size` 越小），可以适应更多的硬件限制，但通信和管理开销也可能增加。
  - **优化作用**：
    - 增加 `block_size` 可以减少块间通信，适合高性能硬件。
    - 减小 `block_size` 可以更细粒度地分配任务，适合资源受限的环境。

---

**4. `gemm_impl: Literal["bf16", "fp8"] = "bf16"`**：精度选择。

**5. `attn_impl: Literal["naive", "absorb"] = "absorb"`**：kv缓存的方式选择。

```python
# 使用 dataclass 定义模型参数的类
@dataclass
class ModelArgs:
    """
    定义模型参数和超参数的数据类。

    Attributes:
        max_batch_size (int): 最大批量大小。
        max_seq_len (int): 最大序列长度。
        dtype (Literal["bf16", "fp8"]): 用于计算的数据类型。
        vocab_size (int): 词汇表大小。
        dim (int): 模型的表示维度。
        inter_dim (int): FFN（前馈网络）的中间层维度。
        moe_inter_dim (int): MoE（专家混合）层的中间层维度。
        n_layers (int): Transformer 层的总数。
        n_dense_layers (int): 使用 FFN 的 Transformer 层数量。
        n_heads (int): 注意力头的数量。
        n_routed_experts (int): MoE 层中所有专家的数量。
        n_shared_experts (int): MoE 层中共享专家的数量。
        n_activated_experts (int): MoE 层中每个输入激活的专家数量。
        n_expert_groups (int): 专家分组数量。
        n_limited_groups (int): MoE 路由的受限分组数量。
        score_func (Literal["softmax", "sigmoid"]): 用于 MoE 路由的评分函数。
        route_scale (float): 路由分数的缩放因子。
        q_lora_rank (int): 查询投影的 LoRA 低秩分解维度。
        kv_lora_rank (int): 键值投影的 LoRA 低秩分解维度。
        qk_nope_head_dim (int): 没有位置嵌入的 Query/Key 头的维度。
        qk_rope_head_dim (int): 使用旋转位置编码的 Query/Key 头的维度。
        v_head_dim (int): Value 投影的头维度。
        original_seq_len (int): 原始序列长度。
        rope_theta (float): 用于旋转位置编码的基数。
        rope_factor (float): 扩展序列长度的缩放因子。
        beta_fast (int): 快速 beta 校正因子。
        beta_slow (int): 慢速 beta 校正因子。
        mscale (float): 扩展注意力的缩放因子。
    """

    max_batch_size: int = 8  # 最大批量大小，默认为 8
    max_seq_len: int = 4096 * 4  # 最大序列长度，默认为 4096 的 4 倍
    dtype: Literal["bf16", "fp8"] = "bf16"  # 数据类型，默认为 "bf16"
    vocab_size: int = 102400  # 词汇表大小
    dim: int = 2048  # 模型的维度（每个 token 的表示维度）
    inter_dim: int = 10944  # FFN 的中间层维度
    moe_inter_dim: int = 1408  # MoE 的中间层维度
    n_layers: int = 27  # Transformer 层的总数
    n_dense_layers: int = 1  # 使用 FFN 的 Transformer 层数量
    n_heads: int = 16  # 注意力头的数量

    # MoE 参数
    n_routed_experts: int = 64  # MoE 中专家的总数量
    n_shared_experts: int = 2  # MoE 中共享专家的数量
    n_activated_experts: int = 6  # 每个输入激活的专家数量
    n_expert_groups: int = 1  # 专家分组的数量
    n_limited_groups: int = 1  # MoE 路由的受限分组数量
    score_func: Literal["softmax", "sigmoid"] = "softmax"  # MoE 路由的评分函数，默认为 "softmax"
    route_scale: float = 1.0  # 路由分数的缩放因子

    # MLA（多头潜在注意力）参数
    q_lora_rank: int = 0  # 查询投影的 LoRA 低秩分解维度，默认为 0
    kv_lora_rank: int = 512  # 键值投影的 LoRA 低秩分解维度
    qk_nope_head_dim: int = 128  # 没有位置嵌入的 Query/Key 头的维度
    qk_rope_head_dim: int = 64  # 使用旋转位置编码的 Query/Key 头的维度
    v_head_dim: int = 128  # Value 投影的头维度

    # Yarn（位置编码和序列长度扩展）参数
    original_seq_len: int = 4096  # 原始序列长度
    rope_theta: float = 10000.0  # 用于旋转位置编码的基数
    rope_factor: float = 40  # 扩展序列长度的缩放因子
    beta_fast: int = 32  # 快速 beta 校正因子
    beta_slow: int = 1  # 慢速 beta 校正因子
    mscale: float = 1.0  # 扩展注意力的缩放因子
```

## 2. 分布式并行化嵌入与映射

在PyTorch中，`nn.Embedding`层是用于处理离散数据（如单词或类别）的关键组件，特别常见于自然语言处理（NLP）和推荐系统等任务。它的主要功能是将输入的整数索引映射到连续的高维向量空间中，即将**索引**转化为**嵌入向量**。

```python
class ParallelEmbedding(nn.Module):
    """
    支持分布式并行的嵌入层。

    参数:
        vocab_size (int): 词汇表大小。
        dim (int): 嵌入维度。
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size  # 保存词汇表大小
        self.dim = dim  # 嵌入维度
        assert vocab_size % world_size == 0  # 确保词汇表大小可以被 world_size 整除
        self.part_vocab_size = (vocab_size // world_size)  # 每个进程负责的部分词汇表大小
        self.vocab_start_idx = rank * self.part_vocab_size  # 当前进程负责的词汇起始索引
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size  # 当前进程负责的词汇结束索引
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))  # 嵌入层权重

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        并行嵌入层的前向传播。

        参数:
            x (torch.Tensor): 输入张量，包含 token 索引。

        返回:
            torch.Tensor: 嵌入后的表示。

        异常:
            ValueError: 如果 world_size 未定义。
        """
        if world_size > 1:
            # 检查 token 是否超出当前进程负责的词汇范围
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx  # 调整索引到当前进程范围
            x[mask] = 0  # 超出范围的 token 索引设为 0
        y = F.embedding(x, self.weight)  # 嵌入操作
        if world_size > 1:
            y[mask] = 0  # 对超出范围的 token 嵌入设为 0
            dist.all_reduce(y)  # 聚合所有进程的嵌入结果
        return y
```

```python
def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    实现线性变换 y = xA^T + b。
    支持量化权重和张量格式的特殊实现。

    参数:
        x (torch.Tensor): 输入张量。
        weight (torch.Tensor): 权重张量，可能经过量化，需要解量化。
        bias (Optional[torch.Tensor]): 偏置张量，默认为 None。

    返回:
        torch.Tensor: 线性变换的结果，可能涉及量化相关的计算。

    注意:
        - 如果权重已量化（如 `element_size() > 1`），需要解量化后计算。
        - 如果 `gemm_impl == "bf16"`，则进行 bf16 精度的 GEMM 操作。
        - 在其他情况下，对输入进行量化并使用 `fp8_gemm` 进行计算。
    """
    if weight.element_size() > 1:
        # 如果权重未量化，直接使用标准线性操作
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        # 如果是 bf16 模式，解量化权重并执行线性操作
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        # 否则对输入量化，并使用 fp8_gemm 进行计算
        x, scale = act_quant(x, block_size)  # 输入量化
        y = fp8_gemm(x, scale, weight, weight.scale)  # fp8 矩阵乘法
        if bias is not None:
            y += bias  # 加上偏置
        return y


class Linear(nn.Module):
    """
    自定义线性层，支持量化权重和可选的偏置。

    参数:
        in_features (int): 输入特征的数量。
        out_features (int): 输出特征的数量。
        bias (bool): 是否包含偏置项，默认为 False。
        dtype (optional): 层的数据类型，默认为 `torch.bfloat16`。
    """
    dtype = torch.bfloat16  # 默认数据类型为 bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数
        # 创建权重张量
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            # 如果权重被量化，则初始化量化比例
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)  # 不需要量化比例
        if bias:
            self.bias = nn.Parameter(torch.empty(self.part_out_features))  # 初始化偏置
        else:
            self.register_parameter("bias", None)  # 无偏置

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        线性层的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 线性变换后的张量。
        """
        return linear(x, self.weight, self.bias)  # 调用自定义线性变换函数

在代码中，**行并行（Row Parallelism）** 和 **列并行（Column Parallelism）** 都是针对 **特征维度（Feature Dimension）** 进行并行化的，它容易被误解成是样本、而不是特征的并行化。

在深度学习的线性层中，通常的计算公式为：
$$
Y = XW^T + b
$$
其中：
- $X $ 是输入数据，形状为 $(\text{batch\_size}, \text{in\_features})$。
- $W $ 是权重矩阵，形状为 $(\text{out\_features}, \text{in\_features})$。
- $b $ 是偏置向量，形状为 $(\text{out\_features},)$。
- $Y $ 是输出，形状为 $(\text{batch\_size}, \text{out\_features})$。

这里的 **行和列** 对应的是矩阵 $W$ 的 **特征维度**：
- **行并行**: 将矩阵 $W$ 的行（输入特征）分片。
- **列并行**: 将矩阵 $W$ 的列（输出特征）分片。

```python
class ColumnParallelLinear(Linear):
    """
    列并行的线性层，将输出特征分布到多个分布式进程中。

    参数:
        in_features (int): 输入特征的数量。
        out_features (int): 总输出特征数量。
        bias (bool): 是否包含偏置项，默认为 False。
        dtype (optional): 数据类型，默认为 `torch.bfloat16`。
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0  # 确保输出特征数可被 world_size 整除
        self.part_out_features = out_features // world_size  # 每个进程负责的输出特征数量
        super().__init__(in_features, self.part_out_features, bias, dtype)  # 调用父类构造函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        列并行线性层的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过列并行计算的张量。
        """
        y = linear(x, self.weight, self.bias)  # 调用自定义线性变换函数
        return y


class RowParallelLinear(Linear):
    """
    行并行的线性层，将输入特征分布到多个分布式进程中。

    参数:
        in_features (int): 总输入特征数量。
        out_features (int): 输出特征数量。
        bias (bool): 是否包含偏置项，默认为 False。
        dtype (optional): 数据类型，默认为 `torch.bfloat16`。
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0  # 确保输入特征数可被 world_size 整除
        self.part_in_features = in_features // world_size  # 每个进程负责的输入特征数量
        super().__init__(self.part_in_features, out_features, bias, dtype)  # 调用父类构造函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        行并行线性层的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过行并行计算的张量。
        """
        y = linear(x, self.weight)  # 调用自定义线性变换函数
        if world_size > 1:
            dist.all_reduce(y)  # 聚合所有进程的输出结果
        if self.bias is not None:
            y += self.bias  # 加上偏置
        return y
```

## 3. RMS Norm层

在Transformer结构中，Layer Normalization（层归一化）是一个至关重要的部分，它是一种特定的归一化技术，与Batch Normalization（批归一化）不同，Layer Normalization不是对一个批次（batch）中的样本进行归一化，而是独立地对每个样本中的所有特征进行归一化（也就是对单一词向量、单一时间点的所有embedding维度进行归一化）。具体来说，对于每个样本，Layer Normalization会在特定层的所有激活上计算均值和方差，然后用这些统计量来归一化该样本的激活值。

- **LN与BN的差别**

BN 和 LN 的差别就在$u_i$和 $\sigma_i$这里，前者在某一个 Batch 内统计某特定神经元节点的输出分布（跨样本），后者在某一次迭代更新中统计同一层内的所有神经元节点的输出分布（同一样本下）。
![Alt text](https://skojiangdoc.oss-cn-beijing.aliyuncs.com/2023DL/transformer/image-29.png)

![](https://skojiangdoc.oss-cn-beijing.aliyuncs.com/2024LLM/19.png)

> * NLP任务中经常会处理长度不同的句子，使用LN时可以不考虑其它样本的长度。<br><br>
> * 在某些情况下，当可用的内存有限或者为了加速训练而使用更小的batch时，BN因为batch数量不足而受到了限制。<br><br>
> * 在某些NLP任务和解码设置中，模型可能会一个接一个地处理序列中的元素，而不是一次处理整个batch。这样BN就不是很适用了。<br><br>
> * 在Transformer模型中有很深的层次和自注意机制。通过对每一层的输入进行规范化，可以防止值的爆炸或消失，从而帮助模型更快地收敛。

![](https://skojiangdoc.oss-cn-beijing.aliyuncs.com/2024LLM/08.webp)

- **LN与RMSNorm的差别**

**RMSNorm** 和 **普通 LayerNorm** 的主要区别在于归一化的计算方式：

1. **归一化公式**：  
   - **LayerNorm** 会对输入特征的均值和方差进行计算，然后用这些统计量对每个特征进行归一化。
   - **RMSNorm** 只计算输入特征的均方根 (RMS, Root Mean Square)，而不考虑均值。因此，RMSNorm 去掉了均值的计算，直接利用每个特征的均方根进行归一化。
<br><br>
2. **计算简化**：  
   - **LayerNorm** 需要同时计算均值和方差，涉及更多的计算步骤。
   - **RMSNorm** 只需要计算输入的均方根，计算量更小。

- **更少的计算开销**：由于去除了对均值的计算，RMSNorm 的计算开销相比 LayerNorm 更小，尤其在大规模模型中表现更为高效。
- **训练稳定性**：RMSNorm 保留了归一化的效果，能够稳定训练过程，同时在某些场景下表现出更好的收敛性。
- **适合大模型**：RMSNorm 因其简化的计算过程，特别适合像 LLaMA 这样的大模型，可以提高训练和推理的效率。
- **统计概念不同**：均方根 (RMS) 是数据点平方的均值再开平方，表达的是数据的“绝对大小”，忽略了数据的符号，反映的是整体数据的幅度或能量，均值和方差则更多代表波动，更关注数据的分布，包括数据的中心位置和离散程度。

```python
class RMSNorm(nn.Module):
    """
    均方根层归一化（RMSNorm）。

    参数:
        dim (int): 输入张量的维度。
        eps (float): 为了数值稳定性引入的小值，默认值为 1e-6。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim  # 保存输入张量的维度
        self.eps = eps  # 设置数值稳定性的小偏移值
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的归一化权重，初始值为 1

    def forward(self, x: torch.Tensor):
        """
        RMSNorm 的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过归一化后的张量，形状与输入相同。
        """
        # 调用 PyTorch 的 rms_norm 实现进行计算
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)
```

## 4. 旋转位置编码ROPE

在 **LLaMA (Large Language Model Meta AI)** 中，使用了一种称为 **旋转位置编码** (RoPE, Rotary Position Embedding) 的技术来引入位置信息。这种编码方式不同于传统的固定位置编码或可学习的位置编码，通过使用**旋转矩阵**将位置信息嵌入到序列中。

- 旋转位置编码 (RoPE) 的原理：
RoPE 的核心思想是将**位置编码**嵌入到每个输入的特征维度中，而不是像传统的绝对位置编码那样为每个位置生成单独的向量。具体而言，RoPE将输入特征通过一个与位置相关的**旋转变换**，在不同位置上通过旋转不同角度来表达位置信息。

1. **嵌入位置信息**：  
   通过使用旋转矩阵，RoPE能够在同一特征空间中嵌入位置信息，并且这种旋转变换可以是连续的，使得模型可以处理不同长度的序列输入，而不依赖于绝对位置编码的长度限制。

2. **特征维度的分组旋转**：  
   RoPE 会将输入特征维度两两分组，并将每对特征维度进行角度旋转，旋转角度根据序列中的相对位置来调整。随着序列位置的变化，每个特征都会以不同的旋转角度进行变化，从而实现位置的编码。

3. **优点**：
   - **相对位置感知**：RoPE 自然具备相对位置感知能力（因为它具有一定的循环性），因此模型可以更好地处理较长序列中的相对位置信息。
   - **长度灵活性**：相比于绝对位置编码，RoPE 可以更加灵活地处理不同长度的序列，而不会受到编码长度的限制。
   - **平滑的位置信息传递**：通过旋转变换的方式嵌入位置信息，使得位置信息在整个特征空间中平滑地传递，避免了绝对位置编码的离散性。

- **旋转位置编码的具体流程**

**Rotary Positional Embedding**（旋转位置编码）的流程可以如下解释——
1. **x1 和 x2 是 token 的原始编码值**。
2. **θ1（theta1）** 是一个常数，为每两维度的编码设置。我们将[$\theta_1, \theta_2...\theta_{d/2}$]这个序列总称为“频率”。
3. **m 是 position（位置）**，表示当前 token 在序列中的位置。
4. 通过**m * θ** 计算角度，并将 **x1 和 x2** 按照这个角度进行旋转，得到新的编码 **x'1 和 x'2**。

这个过程的核心是通过旋转操作引入位置相关的信息，这种方法可以使得模型对相对位置更加敏感，同时保持旋转不变性。

![](https://skojiangdoc.oss-cn-beijing.aliyuncs.com/2024LLM/07.webp)

- **极坐标表示**

**极坐标表示**是表示二维平面上点的一种方式，与我们常用的**直角坐标系**不同。在极坐标系中，一个点的位置由**模长**（半径）和**角度**来确定。
- **直角坐标系**：一个点的位置用两个值 `(x, y)` 来表示，`x` 是水平轴（x轴）的距离，`y` 是垂直轴（y轴）的距离。
- **极坐标系**：一个点的位置用两个值 `(r, θ)` 来表示，其中：
  - **`r`** 是从原点到这个点的距离（模长或半径），它表示点到原点的距离。
  - **`θ`** 是这个点与极轴（通常是正x轴）之间的夹角，称为**角度**或**方位角**。

- <font color="red">**相对位置表示**</font>

通过这样的旋转，不难发现一个现象——当两个token距离很近时，他们旋转后的向量之间的夹角也会是很小的锐角。当两个token距离很远时，他们旋转后的向量的夹角角度就会更大。在注意力机制计算的过程中，我们是会计算词向量之间的点积来判断token与token的相关性，而两个向量的夹角是锐角时，向量之间的相关性就强，两个向量的夹角越靠近直角时，向量之间的相关性就弱。在旋转位置编码中，距离差距越大、两个向量旋转后的角度越大，因此**旋转位置编码通过在自注意力机制中旋转查询和键向量，使得随着两个词之间距离增加时，两个词之间的注意力分数（点积值）会逐渐衰减，相关性会减弱**，这有助于更好地模拟自然语言中远距离词语关联度降低的现象。

那如果两个词距离太远，出现钝角、甚至出现反向的锐角怎么办？当两个向量之间出现钝角时，其相关性也比直角时要强，更不要提反向锐角了！为了避免距离非常遥远的两个token之间反而相关性强，RoPE设计时内嵌了随相对距离增加点积递减的机制，那就是**在计算频率时，位置值越大的词拿到的频率越小**。尽管在极坐标上旋转有可能使一些远离的词向量“接近”或“反向”，但是由于RoPE对长距离的相对位置引入了额外的衰减因子，这种情况下即使形成了反向的锐角，点积值也会因为衰减而减少，从而降低它们的注意力权重。这种情况下，即便在极坐标下形成反向锐角、或者钝角，旋转引入的这种衰减效应也会防止这些词在计算中获得过高的权重。

```python
def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    预计算旋转位置编码（Rotary Positional Embeddings, RoPE）所需的频率和复数指数值。

    参数:
        args (ModelArgs): 模型的参数，包含位置编码的相关设置。

    返回:
        torch.Tensor: 预计算的用于位置编码的复数指数值。
    """
    dim = args.qk_rope_head_dim  # 位置编码作用的维度
    seqlen = args.max_seq_len  # 最大序列长度
    beta_fast = args.beta_fast  # 快速旋转的频率因子
    beta_slow = args.beta_slow  # 缓慢旋转的频率因子
    base = args.rope_theta  # 旋转位置编码的基数，用于生成频率
    factor = args.rope_factor  # 用于扩展序列长度的缩放因子

    # 定义一个函数，用于计算旋转编码的修正维度
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        计算旋转编码的修正维度。

        参数:
            num_rotations (float): 要计算旋转的频率数量。
            dim (int): 嵌入空间的维度。
            base (float): 指数计算的基数。
            max_seq_len (int): 最大序列长度。

        返回:
            float: 根据输入参数计算的修正维度。
        """
        # 根据旋转频率和维度，计算修正维度
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    # 定义一个函数，用于计算旋转编码的修正范围
    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        计算旋转编码的修正范围。

        参数:
            low_rot (float): 低频旋转的下界。
            high_rot (float): 高频旋转的上界。
            dim (int): 嵌入空间的维度。
            base (float): 指数计算的基数。
            max_seq_len (int): 最大序列长度。

        返回:
            Tuple[int, int]: 修正范围的下界和上界（低，高），并限制在合法索引范围内。
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))  # 修正范围下界
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))  # 修正范围上界
        return max(low, 0), min(high, dim - 1)  # 确保范围在合法维度内

    # 定义一个函数，用于生成线性渐变的修正因子
    def linear_ramp_factor(min, max, dim):
        """
        计算线性渐变因子，用于在范围内平滑修正值。

        参数:
            min (float): 渐变范围的最小值。
            max (float): 渐变范围的最大值。
            dim (int): 渐变因子的维度。

        返回:
            torch.Tensor: 形状为 (dim,) 的张量，其值在 [0, 1] 之间线性插值。
        """
        if min == max:
            max += 0.001  # 防止分母为 0
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)  # 线性函数
        ramp_func = torch.clamp(linear_func, 0, 1)  # 限制值在 [0, 1] 之间
        return ramp_func

    # 生成初始频率张量
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:  # 如果序列长度超出原始长度，进行扩展
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)  # 计算修正范围
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)  # 生成平滑修正因子
        freqs = freqs / factor * (1 - smooth) + freqs * smooth  # 修正频率值

    t = torch.arange(seqlen)  # 生成时间步长
    freqs = torch.outer(t, freqs)  # 计算频率的外积
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 生成复数指数值
    return freqs_cis  # 返回预计算结果
```

```python
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    将旋转位置编码（RoPE）应用于输入张量。

    参数:
        x (torch.Tensor): 输入张量，带有需要添加位置编码的值。
        freqs_cis (torch.Tensor): 预计算的复数指数值，用于位置编码。

    返回:
        torch.Tensor: 应用位置编码后的张量。
    """
    dtype = x.dtype  # 获取输入张量的数据类型
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))  # 将张量视为复数
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))  # 调整频率张量的形状以便广播
    y = torch.view_as_real(x * freqs_cis).flatten(3)  # 进行旋转编码，并恢复为实数
    return y.to(dtype)  # 返回与输入相同数据类型的结果
```

## 4. DeepSeek v3的KV缓存机制

**KV缓存（Key-Value Cache）**机制在Transformer模型的自回归生成任务中（如文本生成）是一种重要的加速技术，尤其是在处理长序列时。它能够减少重复计算，从而加速推理过程。这种机制主要应用于**解码器**架构，常见于GPT系列模型等自回归模型。在自回归生成任务中，模型逐步生成序列中的每个token。例如，在文本生成中，每一步生成一个新的token，然后将这个token与之前的所有tokens一起重新输入模型，预测下一个token。对于每一步的生成，模型会重新计算所有tokens的注意力（self-attention），包括所有历史tokens（即已经生成的tokens）和当前生成的token。

在Transformer的每一层中，注意力机制会基于输入生成**查询（Query）**、**键（Key）**和**值（Value）**。计算公式如下：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

- `Query` 表示当前token的信息，它用于寻找与其他tokens相关的内容。
- `Key` 和 `Value` 分别代表历史tokens的信息，它们在每个生成步骤中被重新计算。

这种重新计算会导致计算量成倍增加。例如，如果生成了100个token，每一步都会重新计算前面99个token和当前token的注意力，这样的重复计算是非常耗时的。**KV缓存机制**的核心思想是避免重复计算注意力层（self-attention）中的键（Key）和值（Value）。KV缓存通过将生成过程中每个token的Key和Value保存在缓存中，只在第一次生成时计算一次Key和Value。在生成后续token时，模型只需要计算新token的Query，而可以直接使用缓存中已经存储的Key和Value，避免了对历史tokens的重复计算。

- **工作原理**

以下是KV缓存机制的详细工作原理。注意，**KV缓存大部分时候适用于推理，因此这个流程大部分时候是自回归的**。

1. **初始化**：
   - 当生成开始时，模型计算输入序列的Key和Value，并将这些计算结果缓存起来，保存在内存中。
     
   - **大部分时候，每个注意力层都会有一对Key-Value缓存，这个缓存是在自回归的每次循环中共用。然而，有时我们也可以在多头注意力机制中、只保留一个头或两个头上的KV值，并共享给所有的头使用**。其中，每次循环中的kv缓存，是推理中常见的用法。多头共享kv的缓存，可以被用于训练和推理两个流程，但可能伤害训练的结果。
     
   - 这些缓存的Key和Value会存储到KV缓存中，并作为后续生成步骤中的参考。

2. **生成过程中**：
   - 当生成下一个token时，模型不需要重新计算前面已经生成的token的Key和Value。它会直接使用之前缓存的Key和Value。
   - 只需要计算当前token的Query，并将它与已经缓存的Key进行点积计算，得出注意力分数。
   - 这些注意力分数会结合缓存的Value来计算当前token的输出。

3. **更新缓存**：
   - 对于每一个生成步骤，模型还会将当前生成的token的Key和Value加入缓存，确保缓存中的Key和Value始终保持更新，包含所有已经生成的tokens。
   - 缓存的大小会逐渐增加，最终会包含所有生成序列的Key和Value。

4. **加速效果**：
   - 由于每个生成步骤只需要计算当前token的Query，而不需要重新计算整个序列的Key和Value，这大大减少了计算量。
   - 随着序列长度增加，缓存的使用能够显著减少时间复杂度，使生成过程更快。

- **类定义与初始化**

```python
class MLA(nn.Module):
    """
    多头注意力层（MLA）。

    属性：
        dim (int): 输入特征的维度。
        n_heads (int): 注意力头的数量。
        n_local_heads (int): 在分布式系统中，本地处理的注意力头数量。
        q_lora_rank (int): Query 低秩投影的秩。
        kv_lora_rank (int): Key/Value 低秩投影的秩。
        qk_nope_head_dim (int): 非位置 Query/Key 投影的维度。
        qk_rope_head_dim (int): 旋转位置编码 Query/Key 投影的维度。
        qk_head_dim (int): Query/Key 投影的总维度。
        v_head_dim (int): Value 投影的维度。
        softmax_scale (float): softmax 计算中的缩放因子。
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim  # 输入特征维度
        self.n_heads = args.n_heads  # 注意力头数量
        self.n_local_heads = args.n_heads // world_size  # 当前进程处理的本地注意力头数量
        self.q_lora_rank = args.q_lora_rank  # Query 低秩投影的秩
        self.kv_lora_rank = args.kv_lora_rank  # Key/Value 低秩投影的秩
        self.qk_nope_head_dim = args.qk_nope_head_dim  # 非位置编码的 Query/Key 投影维度
        self.qk_rope_head_dim = args.qk_rope_head_dim  # 旋转位置编码的 Query/Key 投影维度
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim  # Query/Key 投影的总维度
        self.v_head_dim = args.v_head_dim  # Value 投影的维度

        # 如果 Query 的 LoRA 秩为 0，则直接初始化线性层
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            # 否则，使用 LoRA 的两步线性投影
            self.wq_a = Linear(self.dim, self.q_lora_rank)  # 第一步投影
            self.q_norm = RMSNorm(self.q_lora_rank)  # RMSNorm 归一化
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)  # 第二步投影

        # 初始化 Key/Value 的线性投影和归一化
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))

        # 初始化输出层
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)

        # 计算 softmax 的缩放因子
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:  # 如果序列长度扩展，则调整缩放因子
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # 初始化缓存
        if attn_impl == "naive":  # 使用基础实现时，分别缓存 Key 和 Value
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:  # 优化实现时，缓存合并的 Key/Value 和位置编码
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
```

---

- **forward**

```python
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        MLA 的前向传播。

        参数：
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, dim)。
            start_pos (int): 序列开始位置，用于缓存。
            freqs_cis (torch.Tensor): 预计算的旋转位置编码。
            mask (Optional[torch.Tensor]): 掩码张量，用于排除某些位置的注意力。

        返回：
            torch.Tensor: 输出张量，形状与输入相同。
        """
        bsz, seqlen, _ = x.size()  # 获取 batch 大小、序列长度、特征维度
        end_pos = start_pos + seqlen  # 计算序列结束位置

        # 计算 Query
        if self.q_lora_rank == 0:
            q = self.wq(x)  # 如果 LoRA 秩为 0，直接线性投影
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))  # 否则通过 LoRA 进行两步投影
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)  # 调整形状以匹配多头注意力
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)  # 分离非位置和旋转位置部分
        q_pe = apply_rotary_emb(q_pe, freqs_cis)  # 应用旋转位置编码

        # 计算 Key/Value
        kv = self.wkv_a(x)  # 初步线性投影
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)  # 分离 Key 和位置部分
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)  # 应用旋转位置编码

        if attn_impl == "naive":  # 基础实现
            q = torch.cat([q_nope, q_pe], dim=-1)  # 合并 Query 的非位置和位置部分
            kv = self.wkv_b(self.kv_norm(kv))  # 进一步线性投影
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)  # 调整形状
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)  # 分离 Key 和 Value
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)  # 合并 Key 的非位置和位置部分
            self.k_cache[:bsz, start_pos:end_pos] = k  # 缓存 Key
            self.v_cache[:bsz, start_pos:end_pos] = v  # 缓存 Value
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale  # 计算注意力分数
        else:  # 优化实现
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)  # 解量化权重
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)  # 调整权重形状
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])  # 计算非位置部分
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)  # 缓存 Key/Value
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)  # 缓存位置编码
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +  # 非位置分数
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale  # 位置分数

        if mask is not None:
            scores += mask.unsqueeze(1)  # 应用掩码
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)  # 计算注意力分布

        # 根据注意力分布计算输出
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])  # 乘以缓存 Value
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])  # 优化实现
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])  # 重投影
        x = self.wo

(x.flatten(2))  # 通过输出层重投影
        return x  # 返回最终结果
```

<center><img src="https://skojiangdoc.oss-cn-beijing.aliyuncs.com/2024LLM/training/73-.png" alt="image-20250107200502389" style="zoom:50%;" />


```python

```
