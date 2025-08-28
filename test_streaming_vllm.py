import asyncio
from openai import AsyncOpenAI

# Point to the local vLLM server
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed" # API key is not needed for local server
)

async def main():
    """
    This function simulates a long conversation to test the Streaming-LLM feature.
    It continuously sends prompts to the model, accumulating the conversation history.
    We can observe that the conversation can proceed even when the total number of
    tokens exceeds the model's context window.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Please keep your responses concise.",
        }
    ]

    # A series of questions to simulate a long conversation
    questions = [
        "What is the history of the Great Wall of China?",
        "Summarize the main points for me.",
        "Who was the first emperor to start the construction?",
        "What were the primary materials used in its construction?",
        "How long did it take to build the entire wall?",
        "Are there any famous legends or stories associated with the Great Wall?",
        "What is its total length and where does it start and end?",
        "How has the wall been preserved over the centuries?",
        "What is its significance in modern-day China?",
        "Can you recommend some sections of the wall to visit for tourists?",
        # Add more questions to extend the conversation further
        "Tell me about the Terracotta Army.",
        "Where was it discovered and by whom?",
        "What was the purpose of creating the Terracotta Army?",
        "How many soldiers are there in total?",
        "Is every soldier's face unique?",
        "What other figures besides soldiers were found?",
        "How does this discovery impact our understanding of the Qin Dynasty?",
        "What are the ongoing conservation efforts for the Terracotta Army?",
        "Let's switch topics. Tell me about the Python programming language.",
        "What are its main advantages?",
        "What is the GIL and how does it affect multi-threading?",
        "Explain the difference between a list and a tuple.",
        "What are decorators in Python and what are they used for?",
        "This conversation is getting long. Can you summarize our discussion so far?"
    ]

    print("Starting long conversation test for Streaming-LLM...\n")

    for i, question in enumerate(questions):
        print(f"--- Turn {i+1} ---")
        print(f"User: {question}")

        # Add user's question to the message history
        messages.append({"role": "user", "content": question})

        try:
            # Create a chat completion stream
            stream = await client.chat.completions.create(
                model="/home/liangzhongkai/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B",
                messages=messages,
                stream=True,
                max_tokens=200,  # Limit response length for each turn
            )

            full_response = ""
            print("Assistant: ", end="", flush=True)
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    full_response += content
            
            print("\n") # Newline after assistant response

            # Add assistant's response to the message history
            messages.append({"role": "assistant", "content": full_response})

            # Optional: Get token count to observe context growth
            # Note: This requires the `tiktoken` library to be installed
            try:
                import tiktoken
                # The tokenizer name might need adjustment depending on the model.
                # For Qwen, 'cl100k_base' is a common choice.
                encoding = tiktoken.get_encoding("cl100k_base")
                total_tokens = sum(len(encoding.encode(msg["content"])) for msg in messages)
                print(f"(Approx. total tokens in context: {total_tokens})\n")
            except ImportError:
                print("(Install 'tiktoken' to see token count estimates)\n")

        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    asyncio.run(main())

