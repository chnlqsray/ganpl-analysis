# GaN PL 智能分析平台 · GaN PL Intelligent Analysis Platform

一套面向半导体光谱研究的 AI 增强分析工作台，将博士课题中分散的实验处理脚本整合为统一的 Streamlit 界面，覆盖从原始文件解析到 AI 辅助论文写作的完整链路。约 6,200 行代码，由本人主导需求与架构，通过与 AI 协作完成开发，未直接编写一行代码。

An AI-augmented analysis workspace for semiconductor photoluminescence research. This tool consolidates previously fragmented experimental processing scripts into a unified Streamlit interface, covering the full chain from raw file parsing to AI-assisted scientific writing. The ~6,200-line codebase was delivered through structured LLM collaboration based on my own requirements and architecture design, without writing any code directly.

---

## 研究背景 · Research Context

本平台基于英国曼彻斯特大学电子电气工程博士课题开发，课题方向为**离子注入对氮化镓（GaN）光致发光性质的影响**。平台直接对接 Thorlabs CCS200 光谱仪输出的 SPF2 二进制文件，数据来源于真实实验测量。

This platform was developed as part of a PhD project at the University of Manchester, investigating **the effects of ion implantation on the photoluminescence properties of gallium nitride (GaN)**. It interfaces directly with the binary SPF2 output format of the Thorlabs CCS200 spectrometer, processing real experimental measurement data.

---

## 核心功能 · Features

### Tab 1 · 批量处理器 · Batch Processor

自动扫描文件夹内所有 SPF2 文件，通过字节级逆向工程从文件头读取曝光时间（字节偏移 900，float64，毫秒），批量执行光谱校正（乘法/除法校正曲线）、坏点线性插值修复与归一化，输出校正后绝对强度文件与归一化文件。

Automatically scans a folder for SPF2 files, reads the integration time directly from the binary header (byte offset 900, float64, milliseconds) via reverse engineering, then batch-applies spectral correction curves (multiply/divide roles), dead-pixel linear interpolation, and normalisation, outputting both corrected-absolute and normalised spectrum files.

### Tab 2 · 光谱查看器 · Spectrum Viewer

多文件交互式 Plotly 叠加图，支持波长（nm）与能量（eV）双横轴切换、图例位置控制与自定义标签，并自动生成峰位与最大强度摘要表格。

Interactive multi-file Plotly overlay with switchable wavelength (nm) / energy (eV) x-axis, customisable legend labels and positions, and an auto-generated peak position and maximum intensity summary table.

### Tab 3 · 峰拟合 · Peak Fitting

在能量空间对单条光谱执行多峰高斯拟合（含 λ² 雅可比校正），自动初始化拟合参数，输出各分量的峰位（eV/nm）、半高宽（FWHM）、面积与 R²，支持导出 xlsx。

Performs multi-peak Gaussian fitting on a single spectrum in energy space with Jacobian correction (I·λ²). Auto-initialises fit parameters and outputs peak position (eV/nm), FWHM, area, and R² for each component, with CSV/TXT export.

### Tab 4 · 功率系列分析 · Power Series Analysis

批量拟合一组功率依赖光谱，在 Log-log 图中提取各发光峰斜率以判断发光机制；进一步通过速率方程全局优化（differential evolution）求解辐射寿命 τ₁（BB/UVB）、τ₂（YB）与辐射效率 η，输出四面板诊断图。

Batch-fits a series of power-dependent spectra; extracts Log-log slopes per emission band to identify emission mechanisms. Further applies global rate-equation optimisation (differential evolution) to solve radiative lifetimes τ₁ (BB/UVB), τ₂ (YB), and radiative efficiency η, with a four-panel diagnostic output.

### Tab 5 · 寿命对比 · Lifetime Compare

加载多个样品条件的速率方程拟合结果，生成跨样品趋势图（τ₁、τ₂、η），专为对比不同退火温度梯度下的载流子动力学规律而设计。

Loads rate-equation fitting results from multiple sample conditions and plots cross-sample trend figures (τ₁, τ₂, η), designed specifically to compare carrier dynamics across annealing temperature gradients.

### Tab 6 · CIE 色坐标图 · CIE Diagram

标准 CIE 1931 色度图，支持全色域填色背景，可手动输入多个样品的 x,y 色坐标并叠加显示，用于对比不同条件下 GaN 发光的色坐标偏移。

Standard CIE 1931 chromaticity diagram with optional full-gamut colour-fill background. Accepts manual x,y coordinate input for multiple samples to visualise emission chromaticity shifts across conditions.

### Tab 7 · Mapping 热图 · Mapping Heatmap

读取二维扫描强度数据，生成交互式 Plotly 热图，内置**离子注入区域自动检测**功能——设定强度百分位或绝对阈值后，自动在热图上用圆圈标注发光被抑制的注入区域边界。若加载完整光谱字典（spec_dict.pkl），可点选任意位点查看该处 PL 光谱。

Reads 2D scan intensity data and renders an interactive Plotly heatmap. Built-in **ion-implanted region auto-detection**: set a percentile or absolute intensity threshold and the platform automatically annotates suppressed-emission regions with circle markers. When a full spectral dictionary (spec_dict.pkl) is loaded, any map point can be selected to display its full PL spectrum.

### Tab 8 · AI Copilot

调用 Groq 大模型（llama-3.3-70b / qwen3-32b），采用"Python 计算 + LLM 写作"分层架构：Python 提取物理描述符（寿命、辐射效率、R²、色坐标、空间统计量）并构造结构化 prompt，LLM 生成固定标签输出（`[FIGURE_CAPTION]`、`[RESULTS_PARAGRAPH]`、`[SPATIAL_SUMMARY]` 等）及 Results & Discussion 草稿，配备多层容错 rescue call 机制，可导出为 .txt 或 .md 文件。

Calls Groq models (llama-3.3-70b / qwen3-32b) using a layered architecture where Python handles computation and the LLM handles writing. Python first extracts physical descriptors (lifetime, radiative efficiency, R², chromaticity, spatial statistics) and assembles structured prompts; the LLM then generates fixed-tag outputs (`[FIGURE_CAPTION]`, `[RESULTS_PARAGRAPH]`, `[SPATIAL_SUMMARY]`, etc.) and Results & Discussion drafts, with a multi-layer rescue call fallback. Output is exportable as .txt or .md.

---

## 技术栈 · Tech Stack

| 类别 | 依赖 |
|------|------|
| 界面框架 | Streamlit |
| 数值计算 | NumPy, SciPy (curve_fit, differential_evolution) |
| 数据处理 | Pandas |
| 可视化 | Plotly |
| 图像处理 | Pillow |
| AI 层 | Groq API via langchain-openai (ChatOpenAI + base_url) |

---

## 运行方式 · Running Locally

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 Groq API Key（可选，仅 Tab 8 AI Copilot 需要）
# 在项目根目录创建 .streamlit/secrets.toml：
# [secrets]
# GROQ_API_KEY = "your_groq_api_key_here"

# 3. 运行
streamlit run app.py
```

**注意 · Note**：实验数据文件（SPF2 原始光谱、校正曲线 txt 等）涉及未发表研究成果，不随代码一并发布。运行时需自行准备兼容格式的光谱数据。

Experimental data files (raw SPF2 spectra, correction curves, etc.) are not included in this repository as they contain unpublished research results. You will need to provide compatible spectral data files to run the platform.

---

## 关于本项目 · About This Project

本平台由本人主导需求定义、系统架构与功能设计，通过与 Claude、Gemini 等大模型持续协作完成开发，未直接编写任何代码。这是"工程师思维 + AI 工具放大产出"方法论的实践案例。

This platform was designed and directed by me — covering requirements, system architecture, and feature specification — and implemented entirely through structured collaboration with Claude and Gemini. No code was written by hand. It is a practical demonstration of the methodology: engineering thinking amplified by AI tooling.

---

*PhD project · University of Manchester · Electrical & Electronic Engineering · 2022–2026*
