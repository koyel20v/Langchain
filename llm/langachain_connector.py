import os
import json
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq  # ✅ using Groq instead of OpenAI
import faiss

# ========= CONFIGURATION ==========
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "kg1_2345"
FAISS_DIR = "faiss_index"
GROQ_API_KEY = "API key"  # 👈 replace with yours
# ==================================

# ---- Step 1: Connect to Neo4j and pull embeddings ----
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
with driver.session() as session:
    records = session.run("""
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN id(n) AS id, n.name AS name, n.embedding AS embedding
    """).data()

print(f"✅ Loaded {len(records)} nodes with embeddings from Neo4j.")

# ---- Step 2: Prepare FAISS index ----
dim = len(records[0]['embedding'])
embeddings_np = np.array([r['embedding'] for r in records]).astype('float32')

if not os.path.exists(FAISS_DIR):
    os.makedirs(FAISS_DIR, exist_ok=True)
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)
    faiss.write_index(index, os.path.join(FAISS_DIR, "index.faiss"))
    with open(os.path.join(FAISS_DIR, "meta.json"), "w") as f:
        json.dump([{"id": r["id"], "name": r["name"]} for r in records], f)
    print("🧩 Created new FAISS index and metadata.")
else:
    index = faiss.read_index(os.path.join(FAISS_DIR, "index.faiss"))
    with open(os.path.join(FAISS_DIR, "meta.json"), "r") as f:
        metadata = json.load(f)
    print("📦 Loaded existing FAISS index.")

# ---- Step 3: Prepare embedder + LLM ----
embedder = SentenceTransformer('all-MiniLM-L6-v2')
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0)

# ---- Step 4: Continuous Query Loop ----
print("\n🤖 Chatbot ready! Type your questions below.")
print("Type 'exit' to quit.\n")

while True:
    query = input("You: ").strip()
    if query.lower() == "exit":
        print("👋 Exiting chatbot. Goodbye!")
        break

    # Embed the query
    query_vector = embedder.encode(query).astype('float32').reshape(1, -1)
    k = 5
    distances, indices = index.search(query_vector, k)
    top_nodes = [metadata[i] for i in indices[0]]

    print("\n🔍 Top matches from KG:")
    for i, node in enumerate(top_nodes):
        print(f"{i+1}. {node['name']} (distance={distances[0][i]:.4f})")

    # Fetch relationships for context
    with driver.session() as session:
        node_ids = [n['id'] for n in top_nodes]
        context_records = session.run("""
            MATCH (a)-[r]->(b)
            WHERE id(a) IN $ids
            RETURN a.name AS source, type(r) AS relation, b.name AS target
        """, ids=node_ids).data()

    context_text = "\n".join(
        [f"{r['source']} -[{r['relation']}]-> {r['target']}" for r in context_records]
    )

    # Ask Groq LLM
    prompt = f"""
    Use the following graph facts to answer the user's question.

    Graph Context:
    {context_text}

    User Question:
    {query}

    Answer in concise form:
    """

    response = llm.invoke(prompt)
    print("\n💬 LLM Answer:\n", response.content)
    print("-" * 80)

driver.close()

