import os
import json
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
import faiss
import time

# ========= CONFIGURATION ==========
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "kg1_2345"
FAISS_DIR = "faiss_index"
GROQ_API_KEY = "API key"
TAVILY_API_KEY = "API key"  # 🔑 <-- ADD YOUR TAVILY API KEY HERE
THRESHOLD = 0.55  # distance threshold for determining relevance
# ==================================

# ---- Step 0: Set up environment variable for Tavily ----
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# ---- Step 1: Connect to Neo4j ----
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ---- Step 2: Load existing data ----
with driver.session() as session:
    records = session.run("""
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN id(n) AS id, n.name AS name, n.embedding AS embedding
    """).data()

if not records:
    print("❌ No nodes found in Neo4j with embeddings. Please populate your KG first.")
    exit()

print(f"✅ Loaded {len(records)} nodes with embeddings from Neo4j.")

# ---- Step 3: Initialize FAISS index ----
dim = len(records[0]['embedding'])
embeddings_np = np.array([r['embedding'] for r in records]).astype('float32')

if not os.path.exists(FAISS_DIR):
    os.makedirs(FAISS_DIR, exist_ok=True)
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)
    faiss.write_index(index, os.path.join(FAISS_DIR, "index.faiss"))
    with open(os.path.join(FAISS_DIR, "meta.json"), "w") as f:
        json.dump([{"id": r["id"], "name": r["name"]} for r in records], f)
    print("🧩 Created new FAISS index.")
else:
    index = faiss.read_index(os.path.join(FAISS_DIR, "index.faiss"))
    with open(os.path.join(FAISS_DIR, "meta.json"), "r") as f:
        metadata = json.load(f)
    print("📦 Loaded existing FAISS index.")

# ---- Step 4: Initialize Embeddings + LLM + Web Search ----
embedder = SentenceTransformer('all-MiniLM-L6-v2')
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0)
search_tool = TavilySearchResults(k=3)

# ---- Step 5: Chat Loop ----
while True:
    query = input("\n💬 Ask your farming-related question (type 'exit' to quit): ").strip()
    if query.lower() == "exit":
        print("👋 Exiting chatbot...")
        break

    # ---- Step 6: Retrieve from KG ----
    query_vector = embedder.encode(query).astype('float32').reshape(1, -1)
    k = 5
    distances, indices = index.search(query_vector, k)
    top_nodes = [metadata[i] for i in indices[0]]

    print("\n🔍 Top matches from KG:")
    for i, node in enumerate(top_nodes):
        print(f"{i+1}. {node['name']} (distance={distances[0][i]:.4f})")

    # ---- Step 7: Check if KG contains relevant info ----
    if distances[0][0] > THRESHOLD:
        print("\n⚠️ No strong match found in KG. Searching the web for fresh information...")
        web_results = search_tool.run(query)

        extract_prompt = f"""
        You are an expert at extracting farming knowledge into structured JSON.

        From the text below, extract key entities and relationships.
        Return **only** valid JSON — nothing else.

        Example output:
        [
          {{"source": "Rainfall", "relation": "affects", "target": "Crop Yield"}}
        ]

        Text:
        {web_results}
        """

        try:
            response = llm.invoke(extract_prompt)
            raw_output = response.content.strip()

            # --- Robust JSON extraction ---
            if not raw_output.startswith("["):
                print("⚠️ LLM did not return valid JSON, trying to clean output...")
                start = raw_output.find("[")
                end = raw_output.rfind("]")
                if start != -1 and end != -1:
                    raw_output = raw_output[start:end+1]
                else:
                    print("❌ Could not find JSON array in response. Skipping.")
                    continue

            new_kg_data = json.loads(raw_output)

        except Exception as e:
            print("❌ Failed to extract structured data:", e)
            print("🔎 Raw model output:\n", response.content)
            continue

        if not new_kg_data:
            print("⚠️ Could not extract any structured knowledge from web data.")
            continue

        # ---- Step 8: Insert new facts into Neo4j ----
        with driver.session() as session:
            for fact in new_kg_data:
                session.run("""
                    MERGE (a:Entity {name: $source})
                    MERGE (b:Entity {name: $target})
                    MERGE (a)-[:RELATION {type: $relation, source:'web'}]->(b)
                """, fact)
        print(f"🌱 Added {len(new_kg_data)} new facts from the web into Neo4j.")

        # ---- Step 9: Update embeddings + FAISS ----
        new_nodes = list(set([f["source"] for f in new_kg_data] + [f["target"] for f in new_kg_data]))
        new_embeddings = embedder.encode(new_nodes).astype('float32')
        index.add(new_embeddings)
        faiss.write_index(index, os.path.join(FAISS_DIR, "index.faiss"))

        with open(os.path.join(FAISS_DIR, "meta.json"), "r") as f:
            metadata = json.load(f)
        metadata.extend([{"id": len(metadata)+i, "name": n} for i, n in enumerate(new_nodes)])
        with open(os.path.join(FAISS_DIR, "meta.json"), "w") as f:
            json.dump(metadata, f)

        print("🔄 Updated FAISS index with new web knowledge.")
        continue

    # ---- Step 10: If relevant KG info found, retrieve context ----
    node_ids = [n['id'] for n in top_nodes]
    with driver.session() as session:
        context_records = session.run("""
            MATCH (a)-[r]->(b)
            WHERE id(a) IN $ids
            RETURN a.name AS source, type(r) AS relation, b.name AS target
        """, ids=node_ids).data()

    context_text = "\n".join(
        [f"{r['source']} -[{r['relation']}]-> {r['target']}" for r in context_records]
    )

    prompt = f"""
    Use the following graph facts to answer the user's question.

    Graph Context:
    {context_text}

    User Question:
    {query}

    Answer concisely:
    """

    response = llm.invoke(prompt)
    print("\n🌾 Answer:\n", response.content)
    time.sleep(1)

driver.close()
