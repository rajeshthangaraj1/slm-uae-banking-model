from haystack.schema import Document
from haystack.nodes import PromptNode
from haystack.pipelines import Pipeline
from datasets import Dataset

# Load your scraped text
with open("scraped_section.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into manageable chunks (adjust as needed)
chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# Create Haystack Documents
documents = [Document(content=chunk) for chunk in chunks]

# Initialize PromptNode (choose a supported model)
prompt_node = PromptNode(
    model_name_or_path="google/flan-t5-base",  # You can change to better QA models
    max_length=128,
    model_kwargs={"use_auth_token": False},  # if using HuggingFace models locally
    default_prompt_template="question-generation"
)

# Optional: wrap in pipeline if needed
pipeline = Pipeline()
pipeline.add_node(component=prompt_node, name="QuestionGenerator", inputs=["PromptNodeInput"])

# Generate questions using the prompt node
qa_pairs = []

for doc in documents:
    prompt = f"Generate 3 questions and answers from this passage:\n{doc.content}"
    response = prompt_node.prompt(prompt)

    # Parse the response (may vary by model)
    for qa in response.split("\n"):
        if "?" in qa:
            parts = qa.split("?")
            question = parts[0].strip() + "?"
            answer = parts[1].strip() if len(parts) > 1 else ""
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "context": doc.content
            })

print(f"✅ Generated {len(qa_pairs)} QA pairs")

# Convert to Hugging Face dataset
dataset = Dataset.from_dict({
    "question": [pair["question"] for pair in qa_pairs],
    "answer": [pair["answer"] for pair in qa_pairs],
    "context": [pair["context"] for pair in qa_pairs]
})

# Save in multiple formats
dataset.save_to_disk("my_qa_dataset")
dataset.to_csv("qa_pairs.csv")
dataset.to_json("qa_pairs.jsonl")
