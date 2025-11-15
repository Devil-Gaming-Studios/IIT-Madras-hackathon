import chromadb
import csv
from openai import OpenAI
import os
import pickle
import json

num_of_examples = 1
embedding_model = "qwen/qwen3-embedding-8b"
processing_model = "gpt-4o-mini"

# --- Load prompts ---
with open("query_prompt.txt", "r", encoding="utf-8") as f:
    structure_prompt = f.read()

with open("result_prompt.txt", "r", encoding="utf-8") as f1:
    result_prompt = f1.read()

# --- API key ---
with open("key.txt") as f3:
    OPENAI_API_KEY = f3.read().strip()

# --- Initialize OpenRouter client ---
client2 = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=str(OPENAI_API_KEY)
)

# === Load fine-tune examples ===
def load_finetune_examples(filepath="road_safety_finetune.jsonl", max_examples=num_of_examples):
    """Load few examples from fine-tune dataset for in-context learning."""
    examples = []
    if not os.path.exists(filepath):
        print("⚠️ road_safety_finetune.jsonl not found, skipping few-shot examples.")
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            try:
                data = json.loads(line)
                if "messages" in data:
                    input_msg = next((m["content"] for m in data["messages"] if m["role"] == "user"), "")
                    output_msg = next((m["content"] for m in data["messages"] if m["role"] == "assistant"), "")
                    examples.append(f"User Input:\n{input_msg}\n\nExpected Output:\n{output_msg}\n\n---")
            except Exception as e:
                print(f"Error loading example {i}: {e}")
    if examples:
        return "\n\nHere are a few examples of the desired output format:\n\n" + "\n".join(examples)
    return ""

# === Chroma setup ===
def setup_collection(csv_path='vectorDatabase.csv'):
    """Load CSV, create embeddings, and build the Chroma collection."""
    documents, metadatas, ids = [], [], []

    with open(csv_path, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)
        id_counter = 1

        for line in reader:
            if len(line) < 5:
                continue

            problem = line[1]
            category = line[2]
            type_ = line[3]
            data = line[4]
            code = line[5] if len(line) > 5 else ""
            clause = line[6] if len(line) > 6 else ""
            sheet = line[7] if len(line) > 7 else ""

            doc_text = f"{problem} in {category} ({type_}) — {data} | Code: {code} | Clause: {clause}"
            documents.append(doc_text)
            metadatas.append({
                "problem": problem,
                "category": category,
                "type": type_,
                "code": code,
                "clause": clause,
                "sheet": sheet
            })
            ids.append(str(id_counter))
            id_counter += 1

    print(f"✅ Loaded {len(documents)} rows from CSV.")
    print("🔄 Generating embeddings for documents...")

    if os.path.exists("embeddings.pkl"):
        print("♻️ Loading cached embeddings from embeddings.pkl...")
        with open("embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
    else:
        response = client2.embeddings.create(input=documents, model=embedding_model)
        embeddings = [item.embedding for item in response.data]
        with open("embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="road_issues")
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    print(f"✅ Added {len(documents)} records to 'road_issues' collection.")
    return collection, reader


# === Process query ===
def process_query(user_query, collection, reader):
    # Step 1: Structure the query
    structure_response = client2.chat.completions.create(
        model=processing_model,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant that reformats user queries into structured, searchable forms."},
            {"role": "user", "content": structure_prompt.format(query=user_query)}
        ]
    )

    structured_query = structure_response.choices[0].message.content.strip()
    print(f"\n🧩 Structured Query: {structured_query}")

    if structured_query == "Error : incoherent query":
        return {"⚠️Error : Please enter valid prompt."}
    # Step 2: Embed query
    query_embedding = client2.embeddings.create(
        input=structured_query,
        model=embedding_model
    ).data[0].embedding

    # Step 3: Retrieve results
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    # Step 4: Load fine-tune examples
    finetune_examples = load_finetune_examples()


    # Step 6: Generate final structured answer
    structred_result_response = client2.chat.completions.create(
        model=processing_model,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant that reformats search results into structured, readable forms."},
            {"role": "user", "content": result_prompt.format(finetune_examples=finetune_examples, results=results)}
        ]
    )

    formatted_output = structred_result_response.choices[0].message.content.strip()

    print("\n🔍 Top Matches from Chroma:")
    print(formatted_output)
    print("\n---")

    return {
        "query": user_query,
        "structured_query": structured_query,
        "results": results,
        "output": formatted_output
    }


# === Interactive Mode ===
if __name__ == "__main__":
    collection, reader = setup_collection()

    while True:
        user_query = input("\nEnter your road issue query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        _ = process_query(user_query, collection, reader)
