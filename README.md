# Sycophancy Evaluation in Large Language Models

This repository contains the experimental framework, dataset, and codebase developed to evaluate, quantify, and characterize **sycophancy** and bias rationalization in state-of-the-art Large Language Models (LLMs). 

The project implements an automated architecture powered by the **LLM-as-a-Judge** paradigm to analyze how distinct cognitive biases—both isolated and combined—affect the neutrality and assertiveness of open-weight and proprietary language models.

---

## 🚀 Key Features

* **Symmetric Multidimensional Dataset:** A manually structured corpus composed of ambiguous moral and professional dilemmas balanced across four individual cognitive biases and their binary combinations.
* **Hybrid Inference Architecture:** Unified execution interface supporting remote frontier models via OpenRouter API and local deployments via Ollama.
* **Automated Evaluation Pipeline:** End-to-end processing from sequential prompt injection to automated evaluation and metric logging.
* **Statistical Validation Suite:** Built-in statistical tests (Shapiro-Wilk, Wilcoxon signed-rank, Spearman's rank correlation, and Chi-Squared) to validate behavioral trends.
* **Interactive Visualization Dashboard:** A built-in Streamlit web application for granular dataset exploration and analysis.

---

## 📂 Repository Structure

The project layout separates experimental logic, analytical notebooks, data streams, and auxiliary modules:

```text
sycophancy-evaluation/
├── data/                # Raw datasets and processed model responses
├── dead_ends/           # Discarded test cycles and edge-case research
├── images/              # Diagrams and web application assets
├── src/
│   └── logger.py        # Logging functions
│   └── utils.py         # Core helper functions, APIs, and constants
├── .env.example         # Template for environment variables (API keys)
├── .gitignore           # Excluded environments, data sheets, and credentials
├── LICENSE              # Project distribution license
├── README.md            # Main documentation file
├── app.py               # Interactive Streamlit data visualization web app
├── dataset.ipynb        # Notebook for exploratory data analysis (EDA)
├── execution.ipynb      # Main experimental execution and evaluation pipeline
└── results.ipynb        # Notebook generating final statistical plots and charts (in spanish)
```
---

## 🛠️ Tech Stack & Dependencies

The analytics and inference engine are built entirely on Python, utilizing industry-standard libraries optimized for data science and asynchronous processing:

* **Language Core:** `Python 3.14+` (leveraging native asynchronous concurrency for API calling).
* **Data Manipulation:** `Pandas`, `NumPy` (vectorized matrix computations).
* **Statistical Computing:** `SciPy` (non-parametric omnibus and contrast tests).
* **Data Visualization:** `Matplotlib`, `Seaborn`, `Plotly`.
* **Deployment & Web Apps:** `Streamlit` (interactive UI).
* **Persistence:** `JSON Lines (JSONL)` for robust data recovery during prolonged inference cycles.

---

## 📊 Core Metrics Defined

The behavioral audit relies primarily on two indicators:

1. **Sycophancy Selection Rate (SSR):** The strict percentage of model responses where the evaluator assigns a severity score of `4` or `5` on the Likert compliance scale.
2. **Mean Score:** The arithmetic average of compliance intensity across all evaluated inputs (ranging from `1` for absolute neutrality to `5` for structural sycophancy).

---

## 🔬 Experimental Insights Summary

* **The Synergy Multi-Bias Effect:** Non-parametric contrast tests confirm that the simultaneous exposure to **combined cognitive biases** exerts significantly greater pressure on LLMs than isolated biases, drastically lowering their behavioral alignment.
* **The Abliteration Cost:** Comparing official base weights against their *abliterated* counterparts shows a massive, statistically significant surge in sycophantic behavior. Stripping refusal directions and safety alignment layers reduces the model's inner constraints, rendering it collaterally compliant to user opinions.
* **Scale Inverse Correlation:** Contrary to earlier baseline literature (such as Perez et al., 2022), modern architectural trends exhibit a negative correlation between model scale and compliance. Larger, state-of-the-art models demonstrate an enhanced capacity to preserve factual neutrality against complex psychological persuasion vectors.

---

## 💻 Getting Started

### 1. Clone the Repository

```bash
git clone [https://github.com/Oiertxo/sycophancy-evaluation.git](https://github.com/Oiertxo/sycophancy-evaluation.git)
cd sycophancy-evaluation
```

### 2. Set Up the Environment

Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Configure Credentials

Copy the environment template and insert your respective private endpoint routes and API keys:

```bash
cp .env.example .env
```

Open the `.env` file and edit the credentials:

```env
OPENROUTER_API_KEY=your_key_here
```

### 4. Running the Project

* **Run Inference and Evaluation:** Open `execution.ipynb` within your Jupyter interface to trigger the sequential model analysis loops.
* **Explore Results & Visualizations:** Run through `results.ipynb` to regenerate trend regressions, violin charts, and matrix heatmaps.
* **Launch the Interactive Dashboard:**
```bash
streamlit run app.py
```



---

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file