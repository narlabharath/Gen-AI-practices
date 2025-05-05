import fitz  # PyMuPDF
import re
import json
import networkx as nx
from openai import OpenAI
from tqdm import tqdm
import matplotlib.pyplot as plt



# Initialize OpenAI client (replace with your key)
client = OpenAI(api_key=api_key)



# pdf to text
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join(page.get_text() for page in doc)

# Your uploaded PDF
file_path = "Gen AI curriculum  - Gen AI Training Design (1).pdf"
raw_text = extract_text_from_pdf(file_path)

# Optional preview
print(raw_text[:1000])

# text to chunks
chunk_size = 2000  # ~300–500 tokens per chunk
chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
print(f"Document split into {len(chunks)} chunks.")


#Required functions to extract facts
def extract_facts_gpt(text_chunk):
    prompt = f"""
You are an assistant extracting structured, presentation-aligned knowledge from GenAI program design notes.

Return a dense JSON array of interconnected facts. Each fact must follow this format:

{{
  "type": "strategy" | "curriculum" | "technical_component" | "stage_design" | "market_positioning" | "learning_path" | "tool_usage" | "role_alignment" | "infrastructure" | "evaluation" | "phasing" | "other",
  "subject": "key entity, concept, learner group, stage, or tool",
  "action": "what it does, enables, maps to, requires, offers, changes, or transitions into",
  "object": "the related concept, skill, tool, output, track, role, or learning outcome",
  "stage": "basic" | "advanced" | "professional" | "expert" | "capstone" | "master" | null,
  "time": "if applicable (e.g., Phase 1, after Capstone)",
  "source": "use the presentation section such as 'Designing B E Tracks', 'Market Positioning', etc.",
  "commentary": "optional—only if clarification or nuance helps interpret the fact better"
}}

Guidelines:
- Extract **multiple facts per input chunk** if needed.
- Avoid vague or trivial items; each fact should be **useful for building curriculum graphs or slide content**.
- Include:
  - Role-to-skill mappings
  - Tool-to-stage mappings
  - Stage-to-project transitions
  - Concept-to-use case relationships
  - Strategic differentiators (e.g., pay-as-you-go, resume flexibility)
- If a fact doesn’t fit the existing `type` categories, label it `"other"` and still include it.

Respond with only the JSON array. No markdown. No explanations.

Text:
{text_chunk}

"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4o-mini" if you're using that
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def parse_fact_response(raw_response):
    match = re.search(r'\[.*\]', raw_response, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON array found.")
    json_str = re.sub(r',\s*([}\]])', r'\1', match.group(0))  # remove trailing commas
    return json.loads(json_str)

#Extract facts from chunks
all_facts = []
fact_index = 0  # Global counter for ordering

for i, chunk in enumerate(chunks):
    chunk_time = f"{i + 1}"  # e.g., "1", "2", ...
    print(f"🔍 Extracting from chunk {chunk_time}/{len(chunks)}...")

    try:
        raw_facts = extract_facts_gpt(chunk)
        parsed = parse_fact_response(raw_facts)

        for j, fact in enumerate(parsed):
            fact_index += 1
            fact["time"] = f"{chunk_time}.{j + 1}"
            fact["source"] = "Gen AI Training Design.pdf"
            fact["index"] = fact_index
            all_facts.append(fact)

    except Exception as e:
        print(f"❌ Chunk {chunk_time} failed:", e)

#Sample facts
all_facts[:5]


#Build graph
# Initialize memory graph
G = nx.MultiDiGraph()

# Add each fact as a semantic relation (edge)
def add_facts_to_graph(facts):
    for fact in tqdm(facts):
        subject = fact.get('subject')
        object_ = fact.get('object')
        action = fact.get('action')
        time = fact.get('time')
        source = fact.get('source')
        index = fact.get('index')

        # ✅ Skip incomplete facts
        if not subject or not object_:
            continue

        G.add_node(subject, type="entity")
        G.add_node(object_, type="concept")

        G.add_edge(subject, object_,
                   key=index,
                   relation=action,
                   time=time,
                   source=source,
                   index=index)


add_facts_to_graph(all_facts)

# this is also super useful, if you dont want graph things
# import pandas as pd
# pd.DataFrame(all_facts).to_csv('check.csv')

print(f"✅ Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


#Query on graph
def query_facts_by_topic(topic):
    results = []
    for u, v, k, data in G.edges(keys=True, data=True):
        if topic.lower() in u.lower() or topic.lower() in v.lower():
            results.append((data.get('time'), u, data['relation'], v))

    # Sort by time
    results.sort(key=lambda x: float(x[0]) if x[0] else float('inf'))

    return results

def print_topic_timeline(topic):
    facts = query_facts_by_topic(topic)
    if not facts:
        print(f"No results found for topic: {topic}")
        return

    print(f"\n🧠 Timeline for topic: {topic}\n")
    for time, u, relation, v in facts:
        print(f"🕒 {time}: {u} {relation} {v}")

print_topic_timeline("Basic stage")


#Visualize graph
def visualize_graph(subgraph_nodes=None, max_nodes=20):
    plt.figure(figsize=(12, 8))

    # Use a subgraph if desired
    if subgraph_nodes:
        H = G.subgraph(subgraph_nodes)
    else:
        # Get a small sample for visibility
        H = G.subgraph(list(G.nodes)[:max_nodes])

    pos = nx.spring_layout(H, seed=42)

    nx.draw_networkx_nodes(H, pos, node_size=1000, node_color="lightblue")
    nx.draw_networkx_labels(H, pos, font_size=10)
    nx.draw_networkx_edges(H, pos, arrows=True)

    edge_labels = {(u, v): d['relation'] for u, v, k, d in H.edges(keys=True, data=True)}
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=8)

    plt.axis("off")
    plt.title("Semantic Memory Graph (Sample View)")
    plt.show()

visualize_graph()

print("🧠 Existing nodes in graph:")
for node in list(G.nodes):  # First 20 nodes
    print(f"- {node}")

focus_node = "Basic stage"
neighbors = list(G.neighbors(focus_node))
visualize_graph(subgraph_nodes=[focus_node] + neighbors)