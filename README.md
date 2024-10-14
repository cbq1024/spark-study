# 优质图书推荐系统

欢迎来到 **优质图书推荐系统**！本项目旨在利用数据分析和交互式可视化来为用户提供图书推荐服务。通过本系统，用户可以发现最受欢迎的书籍和个性化推荐，帮助用户找到符合兴趣的好书。

[![GitHub](https://img.shields.io/github/stars/cbq1024/ai-dev?style=social)](https://github.com/cbq1024/ai-dev)

## 目录

- [项目简介](#项目简介)
- [安装指南](#安装指南)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [依赖库](#依赖库)
- [贡献](#贡献)

## 项目简介

**优质图书推荐系统** 是一个基于 Streamlit 的应用程序，结合了各种数据分析工具，旨在为用户提供交互式图书推荐体验。用户可以通过这个平台浏览最受欢迎的图书、根据个人偏好进行筛选和排序，从而找到自己感兴趣的书籍。

系统使用来自 Kaggle 的公开数据集，并经过数据清洗和处理，以确保推荐的准确性和实用性。

## 安装指南

要运行此项目，请按照以下步骤进行操作：

1. 克隆本项目的 GitHub 仓库：

   ```bash
   git clone https://github.com/cbq1024/ai-dev.git
   cd ai-dev
   ```

2. 安装项目所需的 Python 库：

   建议在虚拟环境中安装依赖：

   ```bash
   python3 -m venv env
   source env/bin/activate   # Linux/macOS
   # 或者
   env\Scripts\activate    # Windows
   ```

   然后安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. 运行应用程序：

   ```bash
   ./start.sh
   ```

   或者直接通过 `streamlit run Hello.py` 启动应用。

## 使用方法

启动应用程序后，你可以在浏览器中访问该服务。在页面中，你可以：

- 浏览和搜索书籍的详细信息。
- 使用 **Book Rank** 页面查看推荐的热门书籍。
- 使用交互式的可视化工具，探索图书数据。

## 项目结构

项目结构如下：

```
├─📁 .streamlit/
│  ├─📄 config.toml
│  └─📄 secrets.toml
├─📁 data/
│  ├─📁 cleaned/
│  │  └─📄 books-kaggle-mohamadreza-momeni.csv
│  ├─📁 image/
│  │  └─📄 poland_ball.jpg
│  └─📁 uncleaned/
│    └─📄 books-kaggle-mohamadreza-momeni.csv
├─📁 pages/
│  └─📄 1_👀_Book_Rank.py
├─📄 .gitignore
├─📄 Hello.py
├─📄 requirements.txt
└─📄 start.sh
```

### 文件说明

- **.streamlit/**: Streamlit 配置文件目录。
  - `config.toml` 和 `secrets.toml` 用于配置和密钥管理。
- **data/**: 数据文件夹，包含清洗后的数据、原始数据和图像。
  - `cleaned/` 和 `uncleaned/` 文件夹存放清洗后的和未清洗的图书数据。
  - `image/` 存放图像资源。
- **pages/**: 包含不同页面的 Python 脚本，用于实现不同功能模块，例如图书排名。
- **Hello.py**: 主应用脚本，启动应用的入口点。
- **requirements.txt**: 项目所需的依赖库列表。
- **start.sh**: 启动脚本，方便在 Unix 系统上快速启动应用。

## 依赖库

本项目依赖以下 Python 库，这些库主要用于数据处理、可视化、网页开发等用途。

- **Streamlit**: `streamlit==1.39.0`，用于构建交互式的网页应用。
- **Pandas**: `pandas==2.2.3`，用于数据处理和分析。
- **Altair**: `altair==5.4.1`，用于绘制交互式数据可视化。
- **BeautifulSoup**: `beautifulsoup4==4.12.3`，用于从 HTML 页面中提取数据。
- 其他依赖库包括 **matplotlib**、**requests**、**plotly** 等，详见 `requirements.txt`。

## 贡献

如果你对本项目感兴趣并希望做出贡献，欢迎提交 Pull Request 或报告 Bug。你也可以通过 Issue 提出改进建议。

1. Fork 本项目
2. 创建你的 Feature 分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送到分支：`git push origin feature/AmazingFeature`
5. 打开 Pull Request

## 许可证

本项目遵循 MIT 许可证。详情请查看 [LICENSE](LICENSE) 文件。

---

感谢你对本项目的关注与支持，希望你在使用 **优质图书推荐系统** 的过程中能找到你喜爱的书籍！