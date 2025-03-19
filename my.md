LLaVA代码解读 - 余日秋山的文章 - 知乎
https://zhuanlan.zhihu.com/p/694730956

作者：余日秋山
链接：https://zhuanlan.zhihu.com/p/694730956



4 位置编码

在 LLaVA 的代码中，并未找到对于 position embedding 的相关修改，那么说明位置编码方式与 LLaVA 中使用的 Vicuna相同。

## 数据
1 预训练（特征对齐）数据


2 指令微调数据

## 模型
class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)


class LlavaLlamaModel(LlavaMetaModel, LlamaModel)


LlavaLlamaForCausalLM  ⬅️ 继承  ⬅️  LlamaForCausalLM, LlavaMetaForCausalLM
│
└── (🟣 组合) self.model = LlavaLlamaModel  ⬅️ 继承  ⬅️  LlavaMetaModel, LlamaModel
                            │
                            ├── (🟢 继承) LlavaMetaModel  ⬅️ 视觉特征提取
                            ├── (🟢 继承) LlamaModel  ⬅️ 语言模型


## 前向传播部分

## 训练

    训练时的 forward 函数：用于计算损失并更新模型参数，训练时不会通过递归生成每个token，而是通过真实的token数据来加速训练。

    生成时的 generate 函数：用于处理生成任务，涉及多步的token预测，并使用不同的解码策略（如greedy search, beam search等）来确保生成文本的质量。解码过程比训练中的预测过程更复杂，通常需要更多的计算。

因此，generate 函数用于生成文本时会比 forward 函数更加复杂，因为它需要处理自回归生成过程和复杂的解码算法。
