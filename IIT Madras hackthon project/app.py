from flask import Flask, render_template, request, jsonify
import threading
from main_engine import setup_collection, process_query
import json, ast, re

app = Flask(__name__)
from flask_cors import CORS
CORS(app)

collection = None
reader = None
init_lock = threading.Lock()


# ==========================================
# ✅ JSON Repair + Parser Function
# ==========================================
def fix_and_parse_output(text):
    """
    Fixes broken JSON returned by the model.
    Accepts both dict/list and badly formatted strings.
    """

    # Case 1: Already valid JSON (list/dict)
    if isinstance(text, (list, dict)):
        return text

    # Case 2: Try direct JSON load
    try:
        return json.loads(text)
    except:
        pass

    # Case 3: Extract JSON-like part
    matches = re.findall(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if matches:
        text = matches[0]

    # Cleanup common errors
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    # Fix missing brackets
    text += "]" * (text.count("[") - text.count("]"))
    text += "}" * (text.count("{") - text.count("}"))

    # Case 4: Last fallback → safe python literal
    try:
        return ast.literal_eval(text)
    except:
        pass

    # Still broken
    return {"error": "Could not parse JSON", "raw": text}


# ==========================================


def initialize_once():
    global collection, reader
    with init_lock:
        if collection is None:
            print("🚀 Initializing Chroma collection...")
            collection, reader = setup_collection()
            print("✅ Initialization complete.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    global collection, reader
    collection, reader = setup_collection()

    data = request.get_json()
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    try:
        print(f"\n🧠 Received query: {user_query}")
        result = process_query(user_query, collection, reader)

        raw_output = result["output"]

        # 🔥 FIX the output using the new function
        output_list = fix_and_parse_output(raw_output)

        # Select second item (index 1)
        selected_output = output_list

        # Save the list properly
        with open("query_results.json", "w", encoding="utf-8") as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)

        return jsonify({"output": selected_output})

    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
