# RebuttalAgent: AI-Powered Academic Paper Rebuttal Assistant

<p align="center">
  <a href="https://arxiv.org/abs/2601.xxxxx" target="_blank"><img src="https://img.shields.io/badge/arXiv-2601.xxxxx-red"></a>
  <a href='https://mqleet.github.io/Paper2Rebuttal_ProjectPage/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
  <a href="https://huggingface.co/spaces/Mqleet/RebuttalAgent" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-red%22"></a>
  <a href="https://huggingface.co/papers/2601.xxxxx" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Daily Papers-red"></a>

</p >


<p align="center">
<strong><big>If you find our work useful, please consider giving us a star 🌟</big></strong>
</p>

## 🔎 Overview

**RebuttalAgent** is an AI-powered multi-agent system that helps researchers craft high-quality rebuttals for academic paper reviews. The system analyzes reviewer comments, searches relevant literature, generates rebuttal strategies, and produces formal rebuttal letters, all through an interactive human-in-the-loop workflow.


### Key Features

- 📄 **Automatic Paper Parsing**: Converts PDF papers to structured text using Docling
- 🔍 **Issue Extraction**: Breaks down reviewer comments into actionable issues with priority levels
- 📚 **Literature Search**: Automatically searches arXiv for relevant supporting papers
- 💡 **Strategy Generation**: Creates data-driven rebuttal strategies (not sophistry!)
- ✍️ **Rebuttal Writing**: Generates formal, conference-ready rebuttal letters
- 🔄 **Human Feedback Loop**: Iteratively refine strategies based on author input



**We recommend to use [Hugginface Space](https://huggingface.co/spaces/Mqleet/RebuttalAgent) to have a quick try!**

If you want to deploy our work locally, following the instructions below:

## 🛠️ Installation

### Setup

**Environment**
```bash
conda create -n rebuttal python=3.10
conda activate rebuttal
pip install -r requirements.txt
```



## 🚀 Quick Start


### ⚙️ Configuration

**API Key**

Copy the example environment file and fill in your API key for your chosen provider:

```bash
cp .env.example .env
```

Then edit `.env` and add your API key:

```bash
# OpenRouter (supports many models)
OPENROUTER_API_KEY=<your_openrouter_api_key>

# Qwen (Alibaba DashScope)
QWEN_API_KEY=<your_qwen_api_key>

# DeepSeek
DEEPSEEK_API_KEY=<your_deepseek_api_key>

# OpenAI
OPENAI_API_KEY=<your_openai_api_key>

# Gemini (Google)
GEMINI_API_KEY=<your_gemini_api_key>

# ZhiPu (GLM)
ZHIPUAI_API_KEY=<your_zhipuai_api_key>
```

> **Note:** Only configure the provider you plan to use.

### Start with Gradio🤗

> **Note (Windows):** On Windows, you may need to run the terminal as Administrator for the first run, as the PDF parser (Docling) needs to download AI models that require symlink permissions. Alternatively, you can enable [Developer Mode](https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development).

Launch the interactive web interface:

```bash
python app.py --port 8080
```



**With GPU acceleration (for PDF parsing):**
```bash
python app.py --device cuda --port 8080
```

> **Note:** The first run may be slower as Docling needs to download AI models from Hugging Face. Subsequent runs will be much faster.

### Workflow

1. **Upload Files**: Upload your paper PDF and review text file
2. **Initial Analysis**: The system parses your paper and extracts reviewer issues
3. **Strategy Review**: For each issue, review the AI-generated rebuttal strategy
4. **Provide Feedback**: Refine strategies through natural language feedback
5. **Generate Rebuttal**: Download the final rebuttal letter and strategy summary

## 📋 Agent Pipeline

The system uses a multi-agent pipeline:

| Agent | Role |
|-------|------|
| **Semantic Encoder** | Compresses paper into high-density format |
| **Issue Extractor** | Extracts and merges reviewer concerns |
| **Literature Retrieval** | Decides if external papers are needed |
| **Reference Filter** | Filters relevant references |
| **Reference Analyzer** | Analyzes supporting literature |
| **Strategy Generator** | Creates rebuttal strategy and to-do list |
| **Strategy Reviewer** | Reviews and improves strategy |
| **Rebuttal Writer** | Writes formal rebuttal letter |
| **Rebuttal Reviewer** | Polishes final rebuttal |


## ✒️ Citation

If you find RebuttalAgent useful for your research, please consider citing:

```bibtex

```

## ❤️ Acknowledgement

### ✍️ RebuttalAgent Contributers

>[Qianli Ma](https://mqleet.github.io/), [Chang Guo](https://github.com/MrC-crm), [Zhiheng Tian](https://github.com/ElysiaTT)

### 🤗 Open-Sourced Repositories

This project builds upon the following open-source projects:
- [Docling](https://github.com/docling-project/docling) - PDF parsing

Thanks to the contributors for their great work!


## 📄 License
This project is licensed under the MIT License.
