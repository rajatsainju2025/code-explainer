# 🐍 Python Code Explainer

A simple web application that uses a fine-tuned `distilgpt2` language model to generate human-readable explanations for Python code snippets.



---

## Overview

This project is an end-to-end demonstration of fine-tuning a pre-trained Large Language Model (LLM) for a specific task. The model was trained on a small, custom dataset of code-explanation pairs to learn the relationship between Python syntax and its plain-English description.

## Features

-   **AI-Powered Explanations:** Get instant explanations for Python code.
-   **Simple Web UI:** An interactive interface built with Gradio.
-   **Self-Contained:** Includes the training script, inference app, and a small built-in dataset.

## Tech Stack

-   **Model:** `distilgpt2` from Hugging Face
-   **Frameworks:** PyTorch, Transformers
-   **UI:** Gradio
-   **Libraries:** Datasets, Accelerate

---

## Setup and Installation

To run this project locally, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/code-explainer.git](https://github.com/YOUR_USERNAME/code-explainer.git)
    cd code-explainer
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

The project is split into two main parts: training the model and running the application.

### 1. Training the Model

First, you need to run the training script. This will fine-tune the `distilgpt2` model on the custom dataset and save the result in a `./results` directory.

```bash
python train.py