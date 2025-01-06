import os
import time
import configparser
import torch

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    QuantoConfig,
    CompileConfig,
)
from optimum.quanto import QuantizedModelForCausalLM, qint2, qint4, qint8


class SLMChatbot:

    def __init__(self, args):

        self.history = []  # store conversation if needed
        self.token_name = args.tokenizer_llm
        self.model_sum = args.preferred_sum_llm
        self.model_name = args.preferred_llm
        self.do_summarize = bool(args.summarize_context)

        # If not already quantized, do an 8-bit quantization by default (can handle 2, 4, 8 bit)
        if "-quanto" not in self.model_name:
            self.quantize_model(bits=8)

        self._load_pipeline()
        self.tokenizer = AutoTokenizer.from_pretrained(self.token_name)

    def _load_pipeline(self):

        if self.do_summarize:
            self._load_summarization_pipeline()

        self._load_generation_model()

    def _load_summarization_pipeline(self):

        self.summary_pipeline = pipeline(
            "summarization",
            model=self.model_sum,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Loaded summarization pipeline with model: {self.model_sum}")

    def _load_generation_model(self):

        quant_config = QuantoConfig(weights="int8")
        qmodel = QuantizedModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=quant_config,
        )
        self.gen_model = qmodel
        print(f"Loaded quantized text-generation model: {self.model_name}")

    def quantize_model(self, bits: int):

        if "-quanto" in self.model_name:
            print(f"Model '{self.model_name}' is already quantized.")
            return

        bits_map = {2: qint2, 4: qint4, 8: qint8}

        quantized_folder_name = f"{os.path.basename(self.model_name)}-quanto-{bits}"
        self.quantize_dir = os.path.join("quantized", quantized_folder_name)

        if os.path.exists(self.quantize_dir):
            print(f"Found existing quantized directory: {self.quantize_dir}")
            self.model_name = self.quantize_dir
            print(f"Using previously quantized model: '{self.model_name}'.")
        else:
            print(f"Quantizing model '{self.model_name}' to {bits}-bit...")
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            qmodel = QuantizedModelForCausalLM.quantize(
                model, weights=bits_map[bits], exclude="lm_head"
            )

            os.makedirs(self.quantize_dir, exist_ok=True)
            qmodel.save_pretrained(self.quantize_dir)

            self.model_name = self.quantize_dir
            print(f"Model quantized to {bits}-bit and saved in: '{self.model_name}'.")

        try:

            config = configparser.ConfigParser()
            config.read("config.ini")

            if not config.has_section("CONFIG"):
                config.add_section("CONFIG")

            config.set("CONFIG", "preferred_llm", self.model_name)

            with open("config.ini", "w") as configfile:
                config.write(configfile)

            print(f"Updated config.ini [CONFIG] -> preferred_llm = {self.model_name}")

        except Exception as e:
            print(f"Failed to update config.ini with new quantized model: {e}")

    def count_tokens(self, text: str) -> int:

        return len(self.tokenizer.tokenize(text))

    def switch_model(self, new_model_name: str):

        self.history.clear()
        self.model_name = new_model_name
        if "-quanto" not in self.model_name:
            self.quantize_model(bytes=8)
        self._load_pipeline()

    def clear_history(self):

        self.history = []

    def get_gen_response(self, prompt: str) -> str:

        if not hasattr(self, "gen_model"):
            raise AttributeError(
                "Generation model not loaded. Use task='text-generation' or load it manually."
            )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        compile_config = CompileConfig(dynamic=True)

        gen_kwargs = {
            "do_sample": True,
            "top_p": 0.95,
            "max_new_tokens": 500,
            "cache_implementation": "static",
            "compile_config": compile_config,
        }

        with torch.no_grad():
            outputs = self.gen_model.generate(**inputs, **gen_kwargs)

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in text:
            text = text.split("Answer:")[1]
        return text

    def get_summary_response(self, prompt: str) -> str:

        if not hasattr(self, "summary_pipeline"):
            raise AttributeError(
                "Summarization pipeline not loaded. Use task='summarization'."
            )
        results = self.summary_pipeline(prompt, do_sample=True, top_p=0.95)
        return results[0]["summary_text"]

    def chat(self, query: str, context_chunks: list = None) -> str:

        start_time = time.perf_counter()
        context_chunks = context_chunks or []
        context = " ".join(context_chunks)

        # print(f"User query: {query}\nContext: {context}\n")

        prompt = (
            "You are my Question Answering Assistant. Use the provided context to answer.\n"
            f"Question: {query}\n\n"
            f"Context: {context}\n\n"
            "Answer:"
        )

        if self.count_tokens(prompt) > 2056:
            raise ValueError(
                "Input too long; reduce the context length or question size."
            )

        response = self.get_gen_response(prompt)

        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})

        elapsed_time = time.perf_counter() - start_time
        hrs, rem = divmod(elapsed_time, 3600)
        mins, secs = divmod(rem, 60)
        print(f"Response time: {int(hrs):02}:{int(mins):02}:{secs:.2f}")

        return response
