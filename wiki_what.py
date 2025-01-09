# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 23:26:03 2023

@author: marca
"""

from qdrant_memory import QdrantMemory
from wiki_gather import WikiGather
from llama_cpp_bot import LlamaCPPChatbot
from utils import Spinner, load_config


args = load_config()


def main():
    print("Welcome to WikiWhat!")
    exit_program = False
    spinner = Spinner()

    # Initialize WikiGather, QdrantMemory, and SLMChatbot
    wiki_gather = WikiGather(args)
    memory = QdrantMemory(args)
    bot = LlamaCPPChatbot(args)


    print(f"\nUsing Qdrant for context and {args.preferred_llm} for answering questions.\n")

    while not exit_program:
        title = input(
            "Please enter a Wikipedia page title (type 'exit' to quit, 'skip' to skip data gathering): "
        )
        if title.lower() == "exit":
            break

        skip_data_gathering = title.lower() == "skip"

        if not skip_data_gathering:
            # Gather data, calculate embeddings, and store them in Chroma
            print(
                "\n\nGathering the background data for this chat, calculating its embeddings, and loading them into Qdrant. This could take some time depending on the topic's complexity.\n"
            )
            wiki_gather.gather(title)
            wiki_chunks = wiki_gather.dump()
            memory.store(wiki_chunks)

            print(f"\n\nOk, I'm ready for your questions about {title}.\n\n")

        while True:
            command = input(
                "Enter a question or a command (enter 'help' for additional commands): "
            ).lower()

            if command == "exit":
                exit_program = True
                break

            if command == "switch topic":
                break

            if command == "switch model":
                new_model = input("Enter the new model name (copied from HuggingFace or quantized\\ folder): ")
                bot.switch_model(new_model)


            if command == "help":
                print(
                    """
                    Commands:
                        switch topic: takes you back to enter a new Wikipedia page
                        exit: quit program
                    """
                )
                continue

            # Fetch context from Chroma and add it to the SLMChatbot
            context_chunks = memory.fetch_context(command)
            
            # Generate the answer using the SLMChatbot
            spinner.start()
            try:
                answer = bot.chat(command, context_chunks=context_chunks)
            except Exception as e:
                answer = f"An error occurred: {e}"
            spinner.stop()
            print(f"\n\nAnswer: {answer}\n\n")

        if not exit_program:
            print(f"\n\nI hope you learned something about {title}!\n\n")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
