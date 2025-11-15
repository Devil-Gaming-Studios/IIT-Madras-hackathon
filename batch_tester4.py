# --- tester.py ---
import random
import re
import csv
import statistics
import concurrent.futures
from openai import OpenAI
from main_engine import setup_collection, process_query

# --- Initialize OpenRouter clients ---
f3 = open("key.txt")
OPENAI_API_KEY = f3.read()

client_eval = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=str(OPENAI_API_KEY)
)
client_generate = client_eval  # same key for generating queries

# --- Step 1: Load Chroma collection ---
collection, reader = setup_collection("vectorDatabase.csv")

# --- Step 2: Load CSV data for query generation ---
csv_data = []
with open("vectorDatabase.csv", newline="", encoding="utf-8") as f:
    csv_reader = csv.reader(f)
    header = next(csv_reader)
    for row in csv_reader:
        if len(row) >= 3:
            csv_data.append({
                "problem": row[1],
                "category": row[2],
                "type": row[3] if len(row) > 3 else ""
            })

# --- Step 3: Ask GPT to generate 100 realistic queries based on CSV data ---
problems_text = "\n".join(
    [f"- {item['problem']} ({item['category']}, {item['type']})" for item in random.sample(csv_data, min(25, len(csv_data)))]
)

query_prompt = f"""
You are a domain expert in road and traffic infrastructure safety.

Below are examples of real-world problems from a dataset:
{problems_text}

Generate 100 diverse, natural-language queries that a civil engineer or road safety auditor might ask
to find solutions to these problems. The queries should vary in phrasing but remain technical and realistic.
Output them as a numbered list (1–100), one query per line.
"""

print("🧠 Generating 100 test queries from dataset...")
query_response = client_generate.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a civil engineering domain expert that generates test cases."},
        {"role": "user", "content": query_prompt}
    ]
)

raw_queries = query_response.choices[0].message.content.strip()
test_queries = [re.sub(r"^\d+\.\s*", "", line).strip() for line in raw_queries.split("\n") if line.strip()]
test_queries = test_queries[:100]
print(f"✅ Generated {len(test_queries)} queries for testing.\n")

# --- Step 4: Define evaluator ---
def evaluate_response(query, response_text):
    eval_prompt = f"""
    You are an evaluator AI.

    Rate the following response based on:
    1. Relevance: Accuracy in matching interventions to the described safety issues and context (0–10).
    2. Comprehensiveness: Ability to provide detailed and appropriate intervention options (0–10).

    Each criterion is scored 0–10. Then average the two scores for a final_score (1–10 scale).

    Query:
    {query}

    Response:
    {response_text}

    Return only this JSON:
    {{
    "relevance": <number>,
    "comprehensiveness": <number>,
    "final_score": <average>
    }}
    """


    rating_response = client_eval.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a strict grader AI that evaluates responses objectively."},
            {"role": "user", "content": eval_prompt}
        ]
    )

    rating_text = rating_response.choices[0].message.content.strip()
    match = re.search(r'"final_score"\s*:\s*(\d+(\.\d+)?)', rating_text)
    if match:
        return float(match.group(1))
    return 0.0

# --- Step 5: Define a single test runner ---
def run_single_test(query):
    try:
        result = process_query(query, collection, reader)
        score = evaluate_response(query, result["output"])
        return (query, score)
    except Exception as e:
        return (query, f"Error: {e}")

# --- Step 6: Run 100 tests in parallel (10 threads) ---
print("🚀 Running 100 evaluations (parallel mode)...\n")
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    for query, score in executor.map(run_single_test, test_queries):
        results.append((query, score))
        if isinstance(score, (int, float)):
            print(f"🎯 {query[:60]}...  →  {score}/10")
        else:
            print(f"⚠️  {query[:60]}...  →  {score}")

# --- Step 7: Summary report ---
valid_scores = [s for _, s in results if isinstance(s, (int, float))]
if valid_scores:
    avg_score = round(statistics.mean(valid_scores), 2)
    print("\n" + "=" * 60)
    print(f"✅ Completed {len(valid_scores)} successful evaluations.")
    print(f"📊 Average Score: {avg_score}/10")
    print(f"📈 Best Score: {max(valid_scores):.1f}/10")
    print(f"📉 Lowest Score: {min(valid_scores):.1f}/10")
    print("=" * 60)
else:
    print("❌ No valid scores generated.")
