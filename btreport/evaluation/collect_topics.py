import os, json, csv
from os.path import join
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import ollama
from tqdm import tqdm
import matplotlib.pyplot as plt
import textwrap

def main(root = "/pscratch/sd/j/jehr/MSFT/BTReport/data/example", report_key="generated_facts", N_debug=80):    
    dirs = [join(root, d) for d in os.listdir(root) if os.path.isdir(join(root, d))]

    all_ref_facts = []

    # Load all reference facts
    for idx, d in enumerate(dirs):
        results_json = join(d, "eval_results.json")
        if not os.path.exists(results_json):
            continue

        with open(results_json, "r") as f:
            results = json.load(f)

        ref_facts = results["tbfact"]["details"][report_key]
        # ref_facts = results["tbfact"]["details"]["reference_facts"]
        for item in ref_facts:
            all_ref_facts.append(item["fact"])

        if idx >= N_debug: 
            break

    print(f"Total facts collected: {len(all_ref_facts)}")


    # embed
    model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight and fast
    emb = model.encode(all_ref_facts, convert_to_tensor=False)

    # cluster
    clustering = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=0.30,
        n_clusters=None
    ).fit(emb)

    labels = clustering.labels_
    print(f"Num clusters found: {len(set(labels))}")


    # build cluster dict
    clusters = defaultdict(list)
    for label, fact in zip(labels, all_ref_facts):
        clusters[label].append(fact)

    # Sort clusters by size
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)


    # display 
    print("\n========== Top Clusters ==========\n")
    for cluster_id, facts in sorted_clusters[:20]:
        print(f"\n### Cluster {cluster_id}  | {len(facts)} facts")
        for f in facts[:10]:  
            print(" •", f)

    out = []
    for cluster_id, facts in tqdm(sorted_clusters):
        desc = describe_cluster_ollama(facts)
        out.append({
            "cluster_description":desc,
            "cluster_id": int(cluster_id),
            "count": len(facts),
            "facts": facts,
        })

    with open(join(root,"pred_fact_clusters.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved to {join(root,'pred_fact_clusters.json')}")




def describe_cluster_ollama(facts, model="gemma3:27b", max_examples=8):
    """
    Produce a short semantic description of a cluster using Ollama.
    Returns a fallback string if Ollama is unavailable.
    """

    # pick sample facts for summarization
    sample_facts = facts[:max_examples]

    prompt = (
        "You are an expert radiologist summarizing clusters of radiology facts. "
        "Given the example fact statements below, write a short 3-8 word description "
        "that captures the shared semantic meaning.\n\n"
        "FACTS:\n- "
        + "\n- ".join(sample_facts)
        + "\n\nDescription:"
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        desc = response["message"]["content"].strip()
        return desc

    except Exception as e:
        print(f"[WARN] Ollama failed: {e}")
        return "general finding category"





def compute_fact_frequency_distribution(
    cluster_json_path="/pscratch/sd/j/jehr/MSFT/BTReport/data/example/fact_clusters.json",
    output_csv="/pscratch/sd/j/jehr/MSFT/BTReport/data/example/fact_frequency.csv",
    output_json="/pscratch/sd/j/jehr/MSFT/BTReport/data/example/fact_frequency.json",
    plot_path="/pscratch/sd/j/jehr/MSFT/BTReport/data/example/fact_frequency.png",
    top_k=20,
    save_plot=True,
):
    """
    Compute a ranked histogram (Fact Frequency Distribution)
    from the cluster JSON created earlier.
    """

    with open(cluster_json_path, "r") as f:
        clusters = json.load(f)

    # Extract (description, count) pairs
    freq = []
    for c in clusters:
        freq.append((c["cluster_description"], c["count"]))

    # Sort by frequency (descending)
    freq_sorted = sorted(freq, key=lambda x: x[1], reverse=True)

    # Save CSV
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster_description", "count"])
        for desc, count in freq_sorted:
            w.writerow([desc, count])

    # Save JSON
    # with open(output_json, "w") as f:
    #     json.dump(
    #         [{"cluster_description": d, "count": c} for d, c in freq_sorted],
    #         f,
    #         indent=2
    #     )

    # Optional plot


    if save_plot:
        labels = [d for d, _ in freq_sorted[:top_k]]
        labels = [l[0:100] for l in labels]
        counts = [c for _, c in freq_sorted[:top_k]]

        # Wrap long labels to multiple lines
        wrapped_labels = [
            "\n".join(textwrap.wrap(label, width=55))  # adjust width as needed
            for label in labels
        ]

        # Increase figure height based on number of labels
        plt.figure(figsize=(14, 0.4 * top_k + 2))

        plt.barh(wrapped_labels[::-1], counts[::-1])
        plt.xlabel("Frequency (Number of Facts)")
        plt.title(f"Top {top_k} Most Common Radiology Fact Clusters")

        # Reduce font size if needed
        plt.gca().tick_params(axis='y', labelsize=9)

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return freq_sorted



if __name__ == '__main__':
    # main()
    compute_fact_frequency_distribution()