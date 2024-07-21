# IPGCN_Code

## 概述

此仓库包含由**骆勇**开发的基于图卷积网络的敏感用户识别（IPGCN）的代码和数据。IPGCN旨在解决如何在用户-商品二分网络中有效识别出敏感用户。敏感用户通常指的是在特定领域或平台上表现出较高敏锐能力的用户，他们能在商品的生命周期早期识别出商品群中的高质量商品，他们往往能够提供有价值的反馈和建议，对提升服务质量和用户体验具有重要意义。通过开发IPGCN方法，充分利用图卷积网络的优势，结合用户评分行为，准确识别出敏感用户，为后续的个性化推荐、用户画像构建等应用提供有力支持。

## 开始使用

### 要求

要运行此代码，您需要安装以下软件/库：

- 软件：Pycharm
- 编程语言：Python

### 设置

1. 克隆仓库：

   ```bash
   git clone https://github.com/mgyllll/IPGCN_Code.git
   ```

2. 导航到克隆的目录：

   ```bash
   cd IPGCN_Code
   ```

3. 安装必需的依赖项（如果尚未安装）：

   ```bash
   pip install -r requirements.txt
   ```

## 使用数据和方法

### 数据

本研究使用的数据集位于`data`目录中。数据格式为CSV。我们鼓励未来的研究人员根据自己的需求对数据进行预处理。

### 方法

IPGCN方法实现在`src`文件夹中。要训练和评估模型，请运行

`baseline_TraditionalMachineLearning.py`——传统机器学习复现文件

`main_IPGCN_L.py`——不同的L参数下IPGCN方法的性能

`main_IPGCN_epochs.py`——不同的epochs参数下IPGCN方法的性能

`main_IPGCN_learnRate.py`——不同的learnRate参数下IPGCN方法的性能

脚本。该脚本接受必要的参数，如数据集路径、模型配置等。有关脚本用法的更多详细信息，请运行：

```bash
python train.py --help
```

## 引用

如果您在研究中使用了此代码或数据，请引用以下论文：

Qiang Guo, Yong Luo, Yang Ou, Min Liu, Jian-Guo Liu. Identification of perceptive users based on the graph convolutional network, ESWA, 2024 (投稿中)

## 联系

如有任何问题或疑问，请随时通过[骆勇/ly1121581284@163.COM]与我们联系。
