import os
import time
import random
import html
import re
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from datasets import Dataset
from datasets import load_dataset


def clean_context(text):
    text = text.replace("\\n", " ")
    text = text.replace("\u00a0", " ")
    text = text.replace("\\/", "/")
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def safe_run_agent(agent, prompt, retries=3):
    for attempt in range(retries):
        try:
            return agent.run(prompt)
        except Exception as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"⚠️ Retry in {wait:.2f}s after error: {e}")
            time.sleep(wait)
    raise Exception("❌ Failed after retries")


def generate_prompt(text_chunk):
    return f"""
You are a helpful assistant. Based on the following content, generate **3 question-answer pairs** that a reader might ask after reading it.

Text:
\"\"\"{text_chunk}\"\"\"

Return the output in this JSON format (no explanations or extra text):

[
  {{
    "question": "...",
    "answer": "..."
  }},
  ...
]
"""


def main():
    # Load environment variables
    load_dotenv()

    agent = Agent(
        model=Gemini(id=os.getenv("GEMINI_MODEL")),
        markdown=True,
    )

    # Load text
    with open("scraped_section.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = [full_text[i:i + 500] for i in range(0, len(full_text), 500)]

    # Resume from checkpoint
    checkpoint_file = "checkpoint.txt"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_index = int(f.read().strip())
    else:
        start_index = 0

    print(f"⏳ Resuming from chunk index: {start_index}")

    qa_pairs = []
    BATCH_SIZE = 10

    for batch_start in range(start_index, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]

        for idx, chunk in enumerate(batch):
            chunk_idx = batch_start + idx
            print(f"⏳ Processing chunk {chunk_idx + 1}/{len(chunks)}...")
            prompt = generate_prompt(chunk)
            try:
                response = safe_run_agent(agent, prompt)
                content = response.content.strip().strip("`").replace("```json", "").replace("```", "").strip()

                if content.startswith("json\n"):
                    content = content[5:].strip()

                pairs = eval(content)

                for pair in pairs:
                    qa_pairs.append({
                        "question": pair.get("question", ""),
                        "answer": pair.get("answer", ""),
                        "context": clean_context(chunk)
                    })

                time.sleep(5)

            except Exception as e:
                print(f"❌ Error in chunk {chunk_idx + 1}: {e}")

        # Save checkpoint
        with open(checkpoint_file, "w") as f:
            f.write(str(batch_start + BATCH_SIZE))
        print(f"✅ Batch {batch_start // BATCH_SIZE + 1} complete. Checkpoint saved.")
        time.sleep(60)

    if qa_pairs:
        dataset = Dataset.from_dict({
            "question": [item["question"] for item in qa_pairs],
            "answer": [item["answer"] for item in qa_pairs],
            "context": [item["context"] for item in qa_pairs]
        })

        dataset.save_to_disk("gemini_qa_dataset")
        dataset.to_csv("qa_pairs.csv")
        dataset.to_json("qa_pairs.jsonl")

        dataset.push_to_hub("rajeshthangaraj1/uae-banking-rulebook-qa", token=os.getenv("HF_TOKEN"))
        print("✅ Dataset pushed to Hugging Face successfully.")
    else:
        print("⚠️ No QA pairs generated.")

    ds = load_dataset("rajeshthangaraj1/uae-banking-rulebook-qa")
    print(ds["train"][0])


if __name__ == "__main__":
    main()
