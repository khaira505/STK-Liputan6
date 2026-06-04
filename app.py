import os
import pickle
import string
import itertools
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import gradio as gr

# --- Pengaturan Path untuk Folder Data (Sudah disesuaikan untuk posisi app.py) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data') 

# Load file pickle langsung dari folder data kamu
with open(os.path.join(DATA_PATH, 'articles.pkl'), 'rb') as f:
    paper = pickle.load(f)
with open(os.path.join(DATA_PATH, 'processed_paper.pkl'), 'rb') as f:
    processed_paper = pickle.load(f)
with open(os.path.join(DATA_PATH, 'thesaurus.pkl'), 'rb') as f:
    thesaurus = pickle.load(f)

# Setup Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_factory = StopWordRemoverFactory()
stopword_remover = stop_factory.create_stop_word_remover()

vectorizer = TfidfVectorizer(use_idf=True)

def search_documents(query_text, use_expansion=True, top_k=5):
    if not query_text.strip():
        return "Masukkan query terlebih dahulu."

    query = query_text.lower()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    query = query.translate(remove_punctuation_map)
    query = stopword_remover.remove(query)
    query = query.split()
    query = [stemmer.stem(x) for x in query]

    if not use_expansion:
        x = [' '.join(query)]
        paper_tfidf = vectorizer.fit_transform(x + processed_paper)
        q = paper_tfidf[0]
        result = cosine_similarity(paper_tfidf, q)
        final = [[num, y[0], x] for num, y in enumerate(result) if y[0] > 0.0]
        max_result = sorted(final, key=lambda x: x[1], reverse=True)
    else:
        list_synonym = []
        for w in query:
            if w in thesaurus:
                list_synonym.append(thesaurus[w])
            else:
                list_synonym.append([w])

        qs = []
        for combo in itertools.product(*list_synonym):
            combo = [stemmer.stem(y) for y in combo]
            qs.append([' '.join(combo)])

        max_result = []
        for x in qs:
            paper_tfidf = vectorizer.fit_transform(x + processed_paper)
            q = paper_tfidf[0]
            result = cosine_similarity(paper_tfidf, q)
            final = [[num, y[0], x] for num, y in enumerate(result) if y[0] > 0.0]
            max_result += final
        max_result = sorted(max_result, key=lambda x: x[1], reverse=True)

    set_result = set()
    new_result = []
    for item in max_result:
        if item[0] not in set_result:
            set_result.add(item[0])
            new_result.append(item)

    # Format output Markdown untuk Gradio
    output = ""
    shown = 0
    for item in new_result:
        if item[0] == 0:
            continue
        if shown >= top_k:
            break
        doc_idx = item[0] - 1
        score = item[1]
        judul = paper[doc_idx]['judul']
        text_preview = paper[doc_idx]['text'][:300] + "..."
        url = paper[doc_idx]['url']
        output += f"### {shown+1}. {judul}\n"
        output += f"**Score:** {score:.4f} | **Query:** {item[2][0]}\n\n"
        output += f"{text_preview}\n\n"
        output += f"[Baca selengkapnya]({url})\n\n---\n\n"
        shown += 1

    if not output:
        return "Tidak ada dokumen yang relevan ditemukan."
    return output

def gradio_search(query, mode, top_k):
    use_exp = (mode == "Dengan Query Expansion")
    return search_documents(query, use_expansion=use_exp, top_k=int(top_k))

# --- Antarmuka Gradio ---
with gr.Blocks(title="Sistem Temu Kembali Informasi") as demo:
    gr.Markdown("# 🔍 Sistem Temu Kembali Informasi")
    gr.Markdown("Mencari dokumen dari 50 artikel Liputan6.com menggunakan TF-IDF + Cosine Similarity")

    with gr.Row():
        query_input = gr.Textbox(label="Masukkan Query", placeholder="contoh: perang dunia, presiden, harga minyak")
        mode_input = gr.Radio(
            ["Tanpa Query Expansion", "Dengan Query Expansion"],
            label="Mode Pencarian",
            value="Dengan Query Expansion"
        )

    top_k_input = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Jumlah Hasil")
    search_btn = gr.Button("🔍 Cari", variant="primary")
    output = gr.Markdown(label="Hasil Pencarian")

    search_btn.click(
        fn=gradio_search,
        inputs=[query_input, mode_input, top_k_input],
        outputs=output
    )

# Jalankan Gradio
if __name__ == '__main__':
    demo.launch()