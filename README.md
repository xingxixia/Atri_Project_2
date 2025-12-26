# Atri_Project_2



我希望我的代码是拿到下载好依赖直接就能跑的类型，而不需要特别仔细的折腾，但是又能够在一定程度上自定义。

这里放一个野生的[Atri官网](https://atri-mdm.com/)。

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

Atri_4是大脑不同的模块，这个我之前写过，只是失败了，当时不知道怎么把drl套进我的理论里面，和ai讲也讲不明白，chatgpt老是对我的想法做很多预设然后说这是错的，然后给出一个一眼就和我想实现的效果相违背的东西，可能会用到isaac sim之类的来仿真，也可能是个单纯的ai只是会用到摄像头之类的。

Atri_5则是实体机器，真正的让Atri降临在这个世界上，思路和大体规划很多，只是缺基底和历练。希望全部的Atri项目可以在我本科或是研究生结束前完成吧。



## 开发日志：

### 2025年12月23日：

- Check_Env.py是检查依赖的文件，不保成，想起来了就往里面塞一点，等整个Atri_2完成了再整体上修改。
- Atri_2_1.0.py是最初始的文件，里面就只是从[国内镜像](https://hf-mirror.com)下载加载[Qwen/Qwen2.5-3B-Instruct ](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)模型。

### 2025年12月24日：

- Atri_2_1.1.py使用[Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)，是加了prompt的文件，输出只能说一坨，显然只是加prompt是不够的。接下来是应该是该用数据集训练一下，目的是仿人格文字输出方式，让我研究研究怎么做。
- 看了看网上有很多模型，感觉可以捞一个，但是呢，毕竟不是自己的，修改起来也麻烦，可以参考，但还是自定义。
- 文本数据集来自[ATRI && GINKA游戏文本分享#GAL#【galgame吧】](https://tieba.baidu.com/p/9078681496?pid=150538365370&cid=#150538365370)，作者:[星痕之弦](https://tieba.baidu.com/p/9078681496?pid=150538365370&cid=#150538365370)，数据集:[百度网盘](https://pan.baidu.com/s/1FTp5jsL2trGWkK7NF0dfwA?pwd=ATRI)。

### 2025年12月15日

- 参考了[Qwen2.5-3b Lora微调一个你自己的ATRI机器人_4070能lora微调多大的模型](https://blog.csdn.net/2501_92532948/article/details/150932768)，感觉还行，不过代码使用没有说的很清楚，至少我运行的时候报错还是不少的，可能是环境问题吧。对文本处理的有些粗暴了，原文有很多诸如

  - “我们在学校造了一个发电机。没想到从理科实验拓展出来的超纲活动，产生了如此意外的效果。（原文中“从”前面多加了“从到”）

  这样的注释，这里没有处理直接扔进去了，我处理了一下，把这种注释基本都删了，虽然应该影响不大但是还是删了吧，而且是在linux上训练的，我暂时不想碰linux，就不用它的方法了，不过既然它训练出效果了，说明还是有很大的参考价值，还是看看怎么使用lora微调吧。不过可以的话，其实我是想要对整个模型训练的，少于十一个小时我都能接受。

### 2025年12月16日

- 今天看了一下lora的论文，然后修改了[Qwen2.5-3b Lora微调一个你自己的ATRI机器人_4070能lora微调多大的模型](https://blog.csdn.net/2501_92532948/article/details/150932768)里的代码，变成windows里面点击就能跑的代码，成功训练了，确实可以看到效果。只是嘛，不知道是不是我不会写prompt的缘故，这模型说他智障都是抬举它了。

  ![Atri2.1.2.2.png (1730×602)](https://tc.xingxixia.top/images/Atri_Project/Atri2/Atri2.1.2.2.png)

  ![Atri2.1.2.5.png (1695×575)](https://tc.xingxixia.top/images/Atri_Project/Atri2/Atri2.1.2.5.png)

  真是令人无语。等后续在处理吧，文本可能也能仔细调一调。这个版本就这样吧先，暂时也没法训练。





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
