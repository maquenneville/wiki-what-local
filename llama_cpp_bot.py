import os
import time

from llama_cpp import Llama  # Make sure to install 'llama-cpp-python'
from utils import load_config


class LlamaCPPChatbot:
    """
    A simplified chatbot class using llama-cpp-python for local inference.
    No quantization or summarization logic — just direct calls to a llama.cpp model.
    """

    def __init__(self, args):
        """
        Initialize the SLMChatbot with paths/names from 'args'.
        We ignore any summarization flags/LLMs, since llama-cpp is used for all inference.

        Args:
            args: An object/dict with attributes, including:
                  - preferred_llm: Path to the local llama.cpp model (GGML/GGUF file)
                  - tokenizer_llm: (optional) Can be used for naming, or skip entirely
        """
        self.history = []
        self.fast_model_name = "bartowski/Llama-3.2-3B-Instruct-GGUF"
        self.fast_filename = "Llama-3.2-3B-Instruct-Q8_0.gguf"
        self.smart_model_name = "lmstudio-community/Llama-3.3-70B-Instruct-GGUF"
        self.smart_filename ="Llama-3.3-70B-Instruct-Q3_K_L.gguf"
        if args.preferred_llm == "fast":
            self.active_model = self.fast_model_name
            self.active_file = self.fast_filename
        elif args.preferred_llm == "smart":
            self.active_model = self.smart_model_name
            self.active_file = self.smart_filename
        else:
            raise ValueError("preferred_llm must be 'fast' or 'smart'")
        self.active_model = None
        # You can ignore 'tokenizer_llm' if you don't need a separate tokenizer file
        self.token_name = getattr(args, "tokenizer_llm", None) or self.model_name
        

        # Load llama-cpp model
        self._load_llama()

    def _load_llama(self):
        """
        Loads the llama.cpp model from self.model_name using llama-cpp-python.
        Adjust n_ctx, n_gpu_layers, temperature, etc. as needed.
        """
        print(f"Loading llama-cpp model from: {self.active_model}")
        self.llama = Llama.from_pretrained(
            repo_id=self.active_model,
            filename=self.active_file,
            local_dir="quantized",
            verbose=False,
            n_ctx=100000

        )
        print(f"Llama model loaded successfully from '{self.model_name}'.")

    def clear_history(self):
        """
        Clears stored conversation history.
        """
        self.history = []

    def get_gen_response(self, prompt: str) -> str:
        """
        Generate text from a prompt using llama-cpp's create_completion call.
        """
        # You can tune these parameters or make them class-level defaults
        response = self.llama(
            prompt=prompt,
            max_tokens=8000,
            top_p=0.95,
            stop=[],
            echo=False
        )

        # The result is typically in response["choices"][0]["text"]
        text = response["choices"][0]["text"]
        # Optional: If you're using a prompt format that includes "Answer:", remove that
        if "Answer:" in text:
            text = text.split("Answer:", 1)[1].strip()

        return text.strip()

    def chat(self, query: str, context_chunks: list = None) -> str:
        """
        A simple chat method. Combines context chunks + user query into a single prompt,
        calls get_gen_response, updates chat history, and returns the model's answer.
        """
        start_time = time.perf_counter()

        context_chunks = context_chunks or []
        context = " ".join(context_chunks)

        # Basic prompt format for question answering
        prompt = (
            "You are my Question Answering Assistant. Use the provided context to answer.\n"
            f"Question: {query}\n\n"
            f"Context: {context}\n\n"
            "Answer:"
        )

        # Generate
        response = self.get_gen_response(prompt)

        # Update history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})

        elapsed_time = time.perf_counter() - start_time
        hrs, rem = divmod(elapsed_time, 3600)
        mins, secs = divmod(rem, 60)
        print(f"Response time: {int(hrs):02}:{int(mins):02}:{secs:.2f}")

        return response

    def switch_model(self, new_model_name: str):
        """
        Switch to a different llama.cpp model. Clears history and reloads the model.
        """
        self.history.clear()
        if new_model_name == "fast":
            self.active_model = self.fast_model_name
            self.active_file = self.fast_filename
        elif new_model_name == "smart":
            self.active_model = self.smart_model_name
            self.active_file = self.smart_filename
        else:
            raise ValueError("Must respond with 'fast' or 'smart'")
        self._load_llama()



