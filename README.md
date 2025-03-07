# NLP-25 课程代码仓库
## 简介
本仓库包含了 NLP-25 课程的相关代码和练习，旨在帮助学生更好地理解和实践自然语言处理（NLP）的相关知识和技术。

## 课程内容
- 讲座资料：每个讲座对应的代码示例和笔记。
- 练习作业：课程中布置的练习和作业代码。

## 目录结构

```
nlp-25/
├── figs                       # 图片
├── lecture-01                 # 第一讲代码
├── lecture-02                 # 第二讲代码
├── ...                        # 其他讲座和练习代码
└── README.md                  # 本说明文档
```

## 环境要求
Python 3.x

必要的 Python 库的安装方式会在.ipynb中提供。

## 使用方法

```bash
git clone https://github.com/yourusername/nlp-25.git
cd nlp-25
```

运行代码： 打开 Jupyter Notebook 或 Jupyter Lab 并运行相应的 .ipynb 文件。

如果你发现代码中有错误或有改进的建议，欢迎提交 Pull Request 或创建 Issue。

## 联系信息
如果你有任何问题或建议，请通过以下方式联系我们：

邮箱: bbhuang24@m.fudan.edu.cn

GitHub: Rick7117

## Q&A

1. Q: nlkt下载时``nltk.download()``报错或是无法下载？

   A: 挂载VPN重新尝试。
2. Q: kenlm的安装命令只有linux和macos版本，windows电脑如何安装？

   A: 安装适用于Linux的Windows子系统 WSL

   ```bash 
    wsl  --install  # 安装wsl
    wsl --list --online # 查看可以安装的linux发行版
    wsl --install -d Ubuntu # 一般而言推荐安装ubuntu
   ```

3. Q: kenlm的安装报错？

   A: 安装相关依赖，如gcc等。

   ```bash
    sudo snap install cmake --classic
    sudo apt update
    sudo apt install gcc g++
    sudo apt install build-essential libboost-all-dev libeigen3-dev zlib1g-dev libbz2-dev
    git clone https://github.com/kpu/kenlm.git
    cd kenlm
    mkdir -p build
    cd build
    cmake ..
    make -j 4
   ```

