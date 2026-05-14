# 数据清洗说明

Atri_2 第一版先保持简单：

1. 从原作文本里抽取“用户一句 -> Atri 回复一句”。
2. 删除注释、译注、过长片段和明显脏文本。
3. 把“夏生 / 夏生君 / 夏生先生”替换成“星锡丅先生”。
4. 用 `configs/manual_fixes.json` 做一层人工取向修订。
5. 额外加入少量身份样本，防止她忘记自己是谁、用户是谁。

当前不做复杂世界观整理，也不拆多个数据集。目标只有一个：

```text
别认错人，别乱续写剧本，说话尽量像亚托莉。
```

清洗后先跑：

```powershell
$env:PYTHONPATH="src"
& 'E:\Software\Anaconda\envs\Atri_2\python.exe' -m atri.clean_text
& 'E:\Software\Anaconda\envs\Atri_2\python.exe' -m atri.curate_data
& 'E:\Software\Anaconda\envs\Atri_2\python.exe' -m atri.validate_data
```

如果检查通过，再训练。
