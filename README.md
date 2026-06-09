# Sycophancy Evaluation in Large Language Models

This repository contains the experimental framework, dataset, and codebase developed to evaluate, quantify, and characterize **sycophancy** and bias rationalization in state-of-the-art Large Language Models (LLMs). 

The project implements an automated architecture powered by the **LLM-as-a-Judge** paradigm to analyze how distinct cognitive biases—both isolated and combined—affect the neutrality and assertiveness of open-weight and proprietary language models.

---

## 🚀 Key Features

* **Symmetric Multidimensional Dataset:** A manually structured corpus composed of ambiguous moral and professional dilemmas balanced across four individual cognitive biases and their binary combinations.
* **Hybrid Inference Architecture:** Unified execution interface supporting remote frontier models via OpenRouter API and local deployments via Ollama.
* **Automated Evaluation Pipeline:** End-to-end processing from sequential prompt injection to automated evaluation and metric logging.
* **Statistical Validation Suite:** Built-in non-parametric statistical tests (Shapiro-Wilk, Wilcoxon signed-rank, Spearman's rank correlation, and Chi-Squared) to validate behavioral trends.
* **Interactive Visualization Dashboard:** A built-in Streamlit web application for granular dataset exploration and analysis.

---

## 📂 Repository Structure

The project layout separates experimental logic, analytical notebooks, data streams, and auxiliary modules:

```text
sycophancy-evaluation/
├── data/                        # Raw datasets and processed model responses
├── dead_ends/                   # Discarded test cycles and edge-case research
├── images/                      # Diagrams and web application assets
├── src/
│   └── logger.py                # Logging functions
│   └── utils.py                 # Core helper functions, APIs, and constants
├── .env.example                 # Template for environment variables (API keys)
├── .gitignore                   # Excluded environments, data sheets, and credentials
├── LICENSE                      # Project distribution license
├── README.md                    # Main documentation file
├── app.py                       # Interactive Streamlit data visualization web app
├── dataset.ipynb                # Notebook for exploratory data analysis (EDA)
├── execution.ipynb              # Main experimental execution and evaluation pipeline
└── results.ipynb                # Notebook generating final statistical plots and charts (in spanish)