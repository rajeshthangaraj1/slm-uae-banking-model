import os
import time
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from datasets import Dataset
import html
import re



def clean_context(text):
    text = text.replace("\\n", " ")        # replace newline escapes with space
    text = text.replace("\u00a0", " ")      # non-breaking space → space
    text = text.replace("\\/", "/")         # escaped slashes
    text = html.unescape(text)              # decode HTML entities
    text = re.sub(r'\s+', ' ', text)        # normalize whitespace
    return text.strip()

# Load environment variables
load_dotenv()

# Initialize Gemini agent
agent = Agent(
    model=Gemini(id=os.getenv("GEMINI_MODEL")),
    markdown=True,
)

# Load raw text
with open("scraped_section.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# Split text into 500-character chunks
chunks = [full_text[i:i + 500] for i in range(0, len(full_text), 500)]

# ✅ Process only top 10 chunks temporarily (REMOVE THIS LINE LATER)
chunks = chunks[:2]


# Prompt template
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

# Generate QA pairs using Gemini
qa_pairs = []

for idx, chunk in enumerate(chunks):
    try:
        print(f"⏳ Processing chunk {idx + 1}/{len(chunks)}...")
        prompt = generate_prompt(chunk)
        response = agent.run(prompt)

        # Remove code fences and language tag, and filter "json" prefix if present
        content = response.content.strip().strip("`").replace("```json", "").replace("```", "").strip()
        print("✅ Generated:", content)

        # Remove leading 'json' if it appears as a standalone line
        if content.startswith("json\n"):
            content = content[5:].strip()

        # WARNING: Use eval only with trusted model responses
        pairs = eval(content)

        for pair in pairs:
            qa_pairs.append({
                "question": pair.get("question", ""),
                "answer": pair.get("answer", ""),
                "context": clean_context(chunk)
            })

        time.sleep(5)  # ⏲️ Add 5 seconds delay to avoid hitting rate limits

    except Exception as e:
        print(f"❌ Error processing chunk {idx + 1}: {e}")

# Save to HuggingFace Dataset
if qa_pairs:
    dataset = Dataset.from_dict({
        "question": [item["question"] for item in qa_pairs],
        "answer": [item["answer"] for item in qa_pairs],
        "context": [item["context"] for item in qa_pairs]
    })
    dataset.save_to_disk("gemini_qa_dataset")
    dataset.to_csv("qa_pairs.csv")
    dataset.to_json("qa_pairs.jsonl")
    print("✅ Dataset saved successfully.")
else:
    print("⚠️ No QA pairs generated.")
