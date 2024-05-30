## Synthetic Data Generation with LangChain and LLMs

This repository provides tools for generating synthetic data using either OpenAI's GPT-3.5-turbo or Ollama's Llama 3-8B. You can use any model from ollama but I tested with llama3-8B in this repository.

**Features:**

* **Flexible Model Selection:** Choose between OpenAI's GPT-3.5-turbo or Ollama's Llama 3 models for your data generation needs.
* **Customizable Data Generation:** Easily modify the provided model modules to generate specific data formats and structures.
* **Scalable Generation:** Specify the number of runs (data generation attempts) to control the volume of synthetic data produced.

**Prerequisites:**

1. **Python 3.7+:** Ensure you have Python installed on your system.
2. **Virtual Environment:** Create a virtual environment for managing project dependencies.
3. **Requirements:** Install the necessary libraries using the provided `requirements.txt` file.

**Installation:**

1. Clone the repository:
   ```bash
   git clone https://github.com/yazanrisheh/Synthetic-Data-Generation-with-LangChain
   ```
2. Navigate to the project directory:
   ```bash
   cd Synthetic-Data-Generation-with-LangChain
   ```
3. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   ```
4. Activate the virtual environment:
   ```bash
   source .venv/bin/activate 
   ```
5. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**Running the Scripts:**

1. **Ollama:**
   * Download and install Ollama from: [https://ollama.com/download/windows](https://ollama.com/download/windows)
   * Install a Llama 3 model from the Ollama Library: [https://ollama.com/library/llama3](https://ollama.com/library/llama3)
   * Navigate to the `ollama` directory:
     ```bash
     cd ollama
     ```
   * Run the script:
     ```bash
     python ollama_synthetic_generation.py
     ```
2. **OpenAI:**
   * Obtain an API key from OpenAI: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   * Navigate to the `openai` directory:
     ```bash
     cd openai
     ```
   * Run the script:
     ```bash
     python openai_synthetic_generation.py
     ```

**Configuration:**

* **Model Selection:**
   * Modify the `ollama_model.py` or `openai_model.py` file to customize the data generation prompts and structure for your desired output.
* **Number of Runs:**
   * Adjust the `runs` variable in the respective script to control the number of data generation attempts.


**Example Usage:**

* **Generate 100 product descriptions using OpenAI's GPT-3.5-turbo:** 
   * Update `openai_model.py` with your product description prompts.
   * Set `runs = 100` in `openai_synthetic_generation.py`.
   * Run `python openai_synthetic_generation.py`.

* **Generate 50 customer reviews for a specific product using Ollama's Llama 3:**
   * Modify `ollama_model.py` to include customer review prompts and format.
   * Set `runs = 50` in `ollama_synthetic_generation.py`.
   * Ensure your Ollama settings are configured correctly.
   * Run `python ollama_synthetic_generation.py`.


**Important Notes:**
* **Note that the models do not always generate the same amount of data points specificed "runs" especially using ollama** 

* **OpenAI Tests:** Running 50 generations cost anywhere from $0.01 to $0.07, taking around 64 seconds each time.
* **Ollama Performance:** Running 100 generations using the Llama 3 model produced 571 outputs in 8299 seconds (2.3 hours). 