最近经常被人求助多模态大模型代码方面的相关问题，由于本人是费曼学习法的坚实拥护者，之前学习的时候也有记录对于LLaVA代码的理解，因此将代码部分的学习记录贴到专栏里，来帮助更多入门者学习代码，阅读建议是对照本文和源代码，本文叙述结构偏向于总体思路-->局部细节的叙述，建议对应代码看的时候可以先找到总体思路中各个部分对应的代码段再分块理解。
数据部分

主函数train.py关于数据集构建的调用语句：data_module = make_supervsed_data_module(tokenizer=tokenizer, data_args=data_args)，内含dataset和data_collector的构建
dataset

对于其中dataset train_dataset = LazySupervisedDataset

(tokenizer=tokenizer,data_path=data_args.data_path, data_args=data_args)，则LazySupervisedDataset关键要素如下：

    类成员变量需要数据存储路径data_list以供__getitem__从中按照下标或者文件名提取对应样本。对于LLaVA构建的视觉指令数据集，在磁盘里以.json格式存储，json文件中每一个条目为一条single image-conversation pair，具体如下格式：

{
 "id": 0,
 "image": "llava_image/00453/004539375.jpg",
 "conversations": [
      {
 "from": "human",
 "value": "<image>\nRender a clear and concise summary of the photo."
      },
      {
 "from": "gpt",
 "value": "select luxury furniture 3 - inch gel memory foam mattress topper"
      }
    ]
  }

    dataset的另一个关键要素就是__getitem__()函数，负责从data_list中按照data_sampler采样得到的样本序号i 取出数据样本并且进行处理，得到模型需要的输入。
        从def __getitem__(self, i) -> Dict[str, torch.Tensor]: 可以看出json中一条数据pair其组织形式为Dict，含有三个keys，分别为input_ids、labels、image matrix。
        内部需要进行的操作主要包含下面5块：判断是否含有图片、有图片则对图片进行预处理得到image matrix、对文本进行预处理、将图片和文本进行拼接、构建label
            判断是否含有图片：json样本条目里是否含有'image'字段
            对图片进行预处理得到image matrix：
                按照路径属性进行读取Image.open(path).convert('RGB')；
                对图片进行预处理process_image()，具有两种逻辑：pad、anyres
                    在pad逻辑下，首先会进行expand2square()将图片按照更长边pad成正方形，pad的值为每个channel预处理操作对应的mean，会保证原始的图像内容放置在正中心。这样保持了图像的长宽比例不变（1:1），这样不会导致图像在某一维度上的失真或变形。而 336x224 到 224x224 改变了图像的长宽比例，可能会使图像在水平方向上被拉伸或压缩，导致模型学习到不自然的特征。在 336x336 到 224x224 的缩放过程中，图像的每一维度都被均匀缩小，减少了每个方向上的像素点。这使得图像信息丢失更加均衡。相对而言，336x224 到 224x224 的缩放会在水平方向上更显著地压缩信息，导致可能重要的视觉特征被模糊或丢失。得到匹配vision encoder分辨率的原图后，直接利用的是vision encoder对应的图像预处理操作，包括resize，center_crop，然后normalize。resize会到CLIP ViT

                    预训练的图像分辨率大小如224。此处会进行normalize，使得之前pad位置的值变成0。
                    anyres逻辑主要是用于高分辨率图像处理，主要涉及的函数为process_anyres_image()：主要包含两个操作，一个是挑选能够被整数切分的resize分辨率，第二个是将高分辨率图片按照vision encoder预训练分辨率分割成子图。对于第一个操作，需要从预定义的grid resolution中去挑选最合适的分辨率。对每个分辨率，计算当前图片大小resize到该分辨率的缩放系数，缩放系数由更短的边决定。然后计算当前预定义分辨率减去原图缩放后的像素数量，得到被浪费的像素数。从所有与定义的分辨率中选择像素数浪费最少预定义分辨率。然后就可以resize&pad到该分辨率，再按照vision encoder预训练分辨率均匀切分成若干个patch，和直接resize高分辨率图片组合成该样本对应的image list，此时image的输入不再是单个tensor，而是多个tensor在新的堆叠。

def select_best_resolution(original_size, possible_resolutions):
 """
    Selects the best resolution from a list of possible resolutions based on the original size.
    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
 original_width, original_height = original_size
 best_fit = None
 max_effective_resolution = 0
 min_wasted_resolution = float('inf')
​
 for width, height in possible_resolutions:
 scale = min(width / original_width, height / original_height)
 downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
 effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
 wasted_resolution = (width * height) - effective_resolution
​
 if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
 max_effective_resolution = effective_resolution
 min_wasted_resolution = wasted_resolution
 best_fit = (width, height)
​
 return best_fit

            对文本进行预处理，输入序列需要包含system prompt、image、formatted conversation string，且遵循了之前的语言模型的范式需要都处理成token_ids sequence，
                对于图像输入，由于图像是连续的表征没有离散化的token ids，所以在此处是采用占位符表示，也说明了为什么通常前向里会有一个"prepare_inputs_labels_for_multimodal()"。图像占位符会保证防止在文本序列的最开头，用DEFAULT_IMAGE_TOKEN=<image>表示图像位置。另外会根据mm_use_im_start_end判断是否需要为图像序列配备开始和结束符DEFAULT_IM_START_TOKEN=<im_start>、DEFAULT_IM_END_TOKEN=<im_end>
                将含有图像占位符的text conversations list转化为一段以正常文本形式展现的完整对话，其中图像为占位符形式，该过程借助conversation_lib对conversations list进行拼接和tokenize处理。
                    对每个{from,value}条目，调用conversation中的conv.append_message()，具体操作就是将from和value对应的值组合成单个list，添加到conversation对应的self.message变量中。
                    最后通过conv.get_prompt()组合成输入大模型的input sequence，其中图像仍然为占位符形式。具体操作是按照sep_style将system_prompt、role、conversation content拼接起来。不同的conversation主要是system prompt设置不同、分隔符不同。
                    对组合后的input sequence进行tokenizer，主要是对text部分进行分词，tokenizer_image_token()：由于其中图像仍然为占位符形式，而词表里没有该单词，所以需要依据占位符截开，分别进行tokenize。prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')] 
            构建label：targets用上述文本预处理后结果进行clone（next-token prediction常规操作，计算CE只需要进行简单偏移即可），此时input_ids的图片仍然为占位符，需要被mask掉预测的部分有
                图像占位符部分
                instruction部分
                pad部分

data_collator

对于其中data_collator部分，其输入instance为上述Dataset定义的若干个dict输出组成的Sequence，要将里面单独的样本组合成一个个batch的形式，以便并行处理，如每个样本单独的label，应该打成batch_label。因此会先从每个dict中取出对应的input_ids和label，组成序列，然后pad成统一长度。

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
​
    tokenizer: transformers.PreTrainedTokenizer
​
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 按照keys将单独样本的各个类型的输入数据取出来打成batch
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels")) 
        # label和input_ids都需要pad到同样长度
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]


        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
​    
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
​
        return batch

模型部分

模型结构：理解该部分便于快速定位自己需要对模型改进，处理的位置应该定位在何处。

    多模态大模型主要由三个部分组成：multimodal_encoder、multimodal_projector、language_model分别对应到下图三个文件夹中
        multimodal_encoder：装各种模态的encoder的实现，对于vision来说，主要含有“clip_encoder.py”
        multimodal_projector：装各种模态的projector的实现，如果是有新的projector实现方式，应该往这里面装，包括projector class具体实现、创建方式（build_from_xxx）
        language_model：装language model的大脑，内含有“llava_llama.py”、“llava_mistral.py”、“llava_mpt.py”

llava model组织结构

    点开llava_llama.py可以看见多模态大模型的class——LlavaLlamaForCausalLM
        其继承了LlamaForCausalLM, LlavaMetaForCausalLM这两个类，成员变量中主要包含用于处理输入提取特征的feature extractor和用于预测token概率分布的lm_headm。由于有额外的图像模态输入，所以LlamaForCausalLM中重写了中的forward和generate函数。
        其中用于feature extractor的llm model对应的class为LlavaLlamaModel

        ，仅作为一个抽象的组合类，组合文本和图像在进入llm之前的特征处理过程，该class继承了LlavaMetaModel和LlamaModel，其中LlavaMetaModel用于注入额外的图像分支处理逻辑到feature extract逻辑中。
        简而言之层级关系为：LlavaLlamaForCausalLM(model、lm_head)-->LlavaLlamaModel-->LlavaMetaModel、LlamaModel。其中LlavaLlamaForCausalLM负责整体逻辑的统筹，LlavaMetaModel控制vision branch功能函数来完成细节的图像特征提取，LlamaModel就是正常的LLM通用的功能。
        其中比较关键的，可能需要自行设计的为LlavaMetaModel和LlavaMetaForCausalLM，均位于llava_arch.py中

LlavaLlamaForCausalLM

从它的主要成员函数可以看出，需要负责统筹特征提取和预测部分，所以其内部主要有实现图像特征提取encode_images()的逻辑（不涉及具体细节）和prepare_inputs_labels_for_multimodal()的逻辑。而，就是负责将图像的特征（来自于self.encode_images的调用）替换掉图像的占位符，构成完整的特征输入（具体流程在前向传播部分具体介绍）

class LlavaMetaForCausalLM(ABC):
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)

    def prepare_inputs_labels_for_multimodal(...)

LlavaMetaModel

在上面我们说了，LlavaMetaModel主要是负责图像分支的特征信息提取，涉及到图像具体处理代码。图像模态处理主要包括两个步骤，第一个是通过vision encoder去得到图像特征，第二个是用projector将图像特征进行对齐。所以内部要完成vision encoder的创建和projector的创建

class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

或者通过调用initialize_vision_modules()来完成。

具体的vision_tower和vision_projector的对应实现在对应的build_xxx()中，对应于其调用的class。如果需要替换vision encoder / projector，应该在build_xxx中添加上对应的class创建与实例化，和在对应的文件夹下实现具体class。
初始化过程

在train.py主函数里，模型创建对应的语句为

model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args,)

其中model_name_or_path在脚本中通常指定的都是一些语言模型路径，完成LlavaLlamaForCausalLM中与LLM相关的初始化。预训练的语言模型的config文件中不可能包含"mm_vision_tower"，因此依据之前提到的LlavaMetaModel的init函数，此时实例化的LlavaLlamaForCausalLM model中不会含有vision_tower和vision_projector。

我们在主函数中可以发现有额外一段逻辑判断，通过利用initialize_vision_modules()完成视觉部分的创建。

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

前向传播部分

    从数据部分我们可以看出，数据处理丢入图片仍然为占位符模式，还需要将图片和文本embedding进行拼接，这点功能的实现主要在LlavaMetaForCausalLM的def prepare_inputs_labels_for_multimodal()中，具体流程逻辑如下
        对图片进行特征提取
            可以看到有一个条件判断为：if type(images) is list or images.ndim == 5，该条件对应于我们数据部分对于高分辨率图片处理的介绍，多个子图会在新的维度上堆叠，使得images的dim等于5，或者以list形式组合子图。此条件下的特征提取对应于高分辨率的图像特征提取。
            高分辨率图片切分得到的一系列子图经过vision encoder得到的token将会依据 mm_patch_merge_type的参数值进行组合。
                如果是“flattern”，则相当于将所有子图的feature直接头尾拼接
                如果是“spatial”，则会将所有子图的token以从左至右、从上至下顺序处理成1D作为替换占位符的image feature，并且会依据图像的2D结构为每一行拼接上特殊的token表示图像行分割符，
        对文本token ids进行处理。按照图像占位符进行split，split得到的text token ids部分进行token_embed()

cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))

        文本embedding和image feature进行拼接。将image feature来替换原有的占位符。
            可以看到image_token_indices可能是一个序列，这是因为输入可能是图文交替形式，也可能是多图形式，需要按顺序拼接上对应的图像embedding。单图的话就应该只有一个，不需要遍历image_token_indices
        Truncate拼接后的序列到max_length
        pad到最大长度，其中position_ids按token idx进行设置（图像也会是这种一维的顺序位置编码）
    输入模型：调用父类的forward函数，根据继承顺序，会调用LlamaForCausalLM的forward进行普通LLM的前向传播，即过堆叠的Transformer block。