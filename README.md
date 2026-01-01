# Atri_Project_2

在半个学期多的自学之下，感觉还是学了不少东西的，基本的概念和逻辑搭建的差不多了，可以开始一遍看论文或者其他人的东西，一边写我自己的东西了，看看论文，有用的东西就直接在这个工程里面使用，用处不大的也可以复现然后放在其他仓库里面作为技术积累。希望这个个人的ATRI项目可以督促我熟悉会看会写会发论文，并且是真正的有价值的东西而不是随便水水。

我希望我的代码是拿到下载好依赖直接就能跑的类型，而不需要特别仔细的折腾，但是又能够在一定程度上自定义。

这里放一个野生的[Atri官网](https://atri-mdm.com/)。

笔记/博客网站：[星锡丅的后宅](https://xingxixia.github.io/)（hexo框架，简单好用），[星锡丅の后宅](https://www.xingxixia.top/)（建设中，使用worldpress搭建）

## Atri_Project_2的目标是：

- 实现一个能跑能对话的模型。
- 在对话时实现人格扮演，认为自己就是亚托莉，能用亚托莉的语气等。
- 能正确称呼我。
- 世界观正确，不至于把现实世界观当成游戏世界观。
- 输出语音，和文字对齐。
- 大体上算流畅，而不是转几秒才能给输出，这意味着要么电脑性能强，要么模型小。

## 整体规划：

Atri_1放置一些Atri官方相关的东西，无关乎完成不完成，一直在做。

Atri_2希望在我大三结束前也就是2026年7月份之前完成。当然了，早点完成更好。

Atri_3我已经有思路了，要加入记忆模块，这个有雏形但是缺个底子。

Atri_4是大脑不同的模块，这个我之前写过，只是失败了，当时不知道怎么把drl套进我的理论里面，和ai讲也讲不明白，chatgpt老是对我的想法做很多预设然后说这是错的，然后给出一个一眼就和我想实现的效果相违背的东西。可能会用到isaac sim之类的来仿真，也可能是个单纯的ai只是会用到摄像头之类的。

Atri_5则是实体机器，真正的让Atri降临在这个世界上，思路和大体规划很多，只是缺基底和历练。希望全部的Atri项目可以在我本科或是研究生结束前完成吧。



## 临时开发日记：

### 2025年12月23日：

- Check_Env.py是检查依赖的文件，不保成，想起来了就往里面塞一点，等整个Atri_2完成了再整体上修改。
- Atri_2_1.0.py是最初始的文件，里面就只是从[国内镜像](https://hf-mirror.com)下载加载[Qwen/Qwen2.5-3B-Instruct ](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)模型。

### 2025年12月24日：

- Atri_2_1.1.py使用[Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)，是加了prompt的文件，输出只能说一坨，显然只是加prompt是不够的。接下来是应该是该用数据集训练一下，目的是仿人格文字输出方式，让我研究研究怎么做。
- 看了看网上有很多模型，感觉可以捞一个，但是呢，毕竟不是自己的，修改起来也麻烦，可以参考，但还是自定义。
- 文本数据集来自[ATRI && GINKA游戏文本分享#GAL#【galgame吧】](https://tieba.baidu.com/p/9078681496?pid=150538365370&cid=#150538365370)，作者:[星痕之弦](https://tieba.baidu.com/p/9078681496?pid=150538365370&cid=#150538365370)，数据集:[百度网盘](https://pan.baidu.com/s/1FTp5jsL2trGWkK7NF0dfwA?pwd=ATRI)。

### 2025年12月25日

- 参考了[Qwen2.5-3b Lora微调一个你自己的ATRI机器人_4070能lora微调多大的模型](https://blog.csdn.net/2501_92532948/article/details/150932768)，感觉还行，不过代码使用没有说的很清楚，至少我运行的时候报错还是不少的，可能是环境问题吧。对文本处理的有些粗暴了，原文有很多诸如

  - “我们在学校造了一个发电机。没想到从理科实验拓展出来的超纲活动，产生了如此意外的效果。（原文中“从”前面多加了“从到”）

  这样的注释，这里没有处理直接扔进去了，我处理了一下，把这种注释基本都删了，虽然应该影响不大但是还是删了吧，而且是在linux上训练的，我暂时不想碰linux，就不用它的方法了，不过既然它训练出效果了，说明还是有很大的参考价值，还是看看lora微调是个什么东西吧。不过可以的话，其实我是想要对整个模型训练的，少于十一个小时我都能接受。

### 2025年12月26日

- 更新了2.1.2，今天看了一下lora的论文，然后修改了[Qwen2.5-3b Lora微调一个你自己的ATRI机器人_4070能lora微调多大的模型](https://blog.csdn.net/2501_92532948/article/details/150932768)里的代码，变成windows里面点击就能跑的代码，成功训练了，确实可以看到效果。只是嘛，不知道是不是我不会写prompt的缘故，这模型说他智障都是抬举它了。

  ![Atri2.1.2.2.png (1730×602)](https://tc.xingxixia.top/images/Atri_Project/Atri2/Atri2.1.2.2.png)

  ![Atri2.1.2.5.png (1695×575)](https://tc.xingxixia.top/images/Atri_Project/Atri2/Atri2.1.2.5.png)

  真是令人无语。等后续在处理吧，文本可能也能仔细调一调。这个版本就这样吧先，暂时也没法训练。

### 2025年12月30日

- 这几天忙着写ppt和大报告，lora论文都没咋搞，真烦，咋这么多报告，还有个ros控制机械臂的东西没写，看来新年别想好好过了。Atri_2_1.2/chat3将对话窗口挪到终端了，唉终端就终端吧，代码比较简洁，什么是当前信息，什么是传入模型的数据一眼就能看出来。
- 老是把我叫成夏生，我迟早把训练文本里的夏生都改成我的名字再训练一遍。
- 这两天写一些lora论文的笔记吧，然后看一看flylora然后看一看Nested Learning和deepseek org（听说这个能压缩文本，似乎是图像领域的patch_embedding然后模型压缩直接代替文字输入，不知道是直接patch切割完直接就结束了还是说要进模型处理。将图像信息近似成文本信息，和我的想法很像啊，我的最终实现里理论上是可以没有任何文本输入的，一定是他抄袭了我脑子里的想法.jpg，改天看看到底是什么个原理）。

### 2025年12月31日

- 今年2025年最后一天啊，真是令人感叹。今天做了Atri_2_1.2/test6，本质就是chat3的代码，然后把接口换了换，把输入变成(inputs_id--------imputs_embedding)+(history_embedding/prefix_embedding)了,这样一来就不会再走embedding查询层，方便对history进行压缩。缺点是后续处理会延长推理时间。坏了，我考虑这个干嘛，现阶段直接对其语音然后直接输出就行了，不纠结记忆了。然后调整一下文本，保证称呼和基本世界观正确就行。这样就差不多可以Atri_2结档了，感觉还是很简单的，如果寒假猛猛做有可能寒假都没过完就能做完atri2。
- 不过对齐语音应该也是需要打开embedding接口的，不过这样的话就有点麻烦了，或许我应该找个更小的模型，3B的没法使用8g的4060训练啊或许我可以延长一下层？虽然可能会损失点效率，可毕竟现在这个是文本-文本的模型，想加语音的话就得搞一个空间对齐。
- 或许我可以加一个比较小的语音模型，正好我打开了embedding层，可以将语音模型的输出（非文字）直接对其到Atri文本模型，不过即便是这样显存也有点太吃紧了，啊啊啊啊啊我要是很有钱就好了。最近需要的电脑配件全都在涨价，涨的太猛了点，烦死了。
- 把lora笔记写一写，lora这块儿就可以过了，然后看看flylora和Nested Learning好了，然后再去看deepseek org，不对应该先看看ai语音领域的论文。



## 电脑配置：

- Laptop: HP Shadow Elf 9
- GPU: RTX 4060 8GB
- CPU: AMD Ryzen 7 7840H w/Radeon 780M Graphics 
- RAM: 16GB

## System:

- Windows 11

## 依赖：

- python version: 3.10
- Pytorch version: 2.8.0+cu129
- CUDA version: 12.9
- transformers version: 4.57.3
- accelerate version: 1.12.0
- gradio version: 6.2.0
- tokenizers version: 0.22.1
- sentencepiece version: 0.2.1
- tiktoken version: 0.12.0
- chromadb version: 1.3.7

## License：

This project is licensed under the Apache License 2.0.

If you use this code or model, please attribute:
Original work by 星锡丅 (https://github.com/xingxixia/Atri_Project_2)
