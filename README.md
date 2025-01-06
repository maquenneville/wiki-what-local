

# wiki-what-local

**wiki-what-local** is a lightweight, offline-capable **Retrieval-Augmented Generation (RAG)** system for question-answering. It uses:

- **Qdrant** for vector-based context storage and retrieval.
- A **Hugging Face Transformer** (via Optimum Quanto) for question-answering and text generation.
- A **local Wikipedia gatherer** to download, parse, and store relevant Wikipedia articles for your queries (inspired by OpenAI's implementation: https://cookbook.openai.com/examples/embedding_wikipedia_articles_for_search).

When you query the system, it:
1. Retrieves the most relevant text chunks from Qdrant.
2. Summarizes or feeds them to a local quantized language model for efficient inference.
3. Generates a final answer with the help of the retrieved context.

## Features

- **Local/offline**: Once the Wikipedia chunks are gathered, the QA pipeline does not require an external API.  No API keys!  (unless using a gated model, in which case follow Huggingface guidelines to access).
- **Customizable**: Easily switch out language models and bit-depth quantization (2, 4, or 8 bits).  Quantization is mandatory, to keep things as lightweight as possible.
- **Configurable**: Set your favorite LLMs, summarization models, or tokenizers by updating `config.ini`.


## Getting Started

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```
   Make sure you have the following libraries (among others):
   - **qdrant-client**
   - **transformers**
   - **optimum-quanto**  
   - **mwclient**  
   - **mwparserfromhell**  

2. **Set Up Configuration**

   Edit `config.ini` to specify:
   - `preferred_llm`: Path or name of your main language model (e.g., a local folder or a Hugging Face model ID).
   - `preferred_sum_llm`: Model for summarization (optional).
   - `tokenizer_llm`: Tokenizer path/name for the selected LLM (should be the same as base model entered before quantization).
   - `collection`: Name of the Qdrant collection you want to store data in.
   - `storage_path`: Path on disk for Qdrant’s local storage (leave blank for in-memory).
     
   The config.ini is currently set with good defaults.  'meta-llama/Llama-3.2-3B-Instruct' has performed the best in my tests, in terms of tradeoff between quality and processing time, but is a gated model.

3. **Run the Application**

   ```bash
   python main.py
   ```
   You’ll be prompted to enter a Wikipedia page title. The program will:
   1. Gather pages related to that title (via `WikiGather`).
   2. Store their text chunks in Qdrant (via `QdrantMemory`).
   3. Allow you to ask questions about the topic (and all other topics stored), retrieving relevant context from Qdrant and generating answers with `SLMChatbot`.

---

## Usage Flow

1. **Enter a Wikipedia page** (e.g., `"George Washington"`).  Must be exact, so copy/paste is recommended.
2. **Wait** while it gathers relevant pages and chunks them into Qdrant.  Processing time varies considerably with the complexity/'broadness' of the topic.
3. **Ask questions** about that topic:
   - The system fetches context from Qdrant and calls the quantized LLM to generate an answer.  Response time will also vary, depending on the size of the LLM.
4. **Switch to a new topic** or **switch models** by typing the corresponding command.

---

## License

[MIT](https://opensource.org/licenses/MIT)

---

Enjoy! If you have any questions or issues, just open an **Issue** in this repository or reach out.

## TODO

- Enable CUDA flow after testing, currently only performs CPU inference
