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

# --- Tambahkan library untuk API dan CORS ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- Pengaturan Path untuk Folder Data ---
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

# --- FUNGSI UTAMA: Mengambil Data Mentah dari Model (Dipakai Bersama) ---
def get_search_data(query_text, use_expansion=True, top_k=5):
    if not query_text.strip():
        return []

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

    results_list = []
    shown = 0
    for item in new_result:
        if item[0] == 0:
            continue
        if shown >= top_k:
            break
        doc_idx = item[0] - 1
        score = float(item[1])
        judul = paper[doc_idx]['judul']
        text_preview = paper[doc_idx]['text'][:300] + "..."
        url = paper[doc_idx]['url']
        
        results_list.append({
            "judul": judul,
            "score": score,
            "expanded_query": item[2][0],
            "text_snippet": text_preview,
            "url": url
        })
        shown += 1
        
    return results_list

# --- Fungsi Output Markdown Khusus untuk Tampilan Gradio ---
def search_documents(query_text, use_expansion=True, top_k=5):
    if not query_text.strip():
        return "Masukkan query terlebih dahulu."
        
    results = get_search_data(query_text, use_expansion, top_k)
    
    if not results:
        return "Tidak ada dokumen yang relevan ditemukan."
        
    output = ""
    for i, r in enumerate(results):
        output += f"### {i+1}. {r['judul']}\n"
        output += f"**Score:** {r['score']:.4f} | **Query:** {r['expanded_query']}\n\n"
        output += f"{r['text_snippet']}\n\n"
        output += f"[Baca selengkapnya]({r['url']})\n\n---\n\n"
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

# --- PENGATURAN FASTAPI & CORS (Untuk Backend Vercel) ---
app = FastAPI()

# Mengizinkan Vercel (dan domain mana pun) mengakses API ini tanpa diblokir CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jalur API JSON yang dipanggil oleh fetch() di index.html Vercel
@app.get("/api/search")
def search_api(q: str = "", expand: str = "true"):
    use_exp = expand.lower() == "true"
    results = get_search_data(q, use_expansion=use_exp, top_k=5)
    return {"results": results}

# Satukan Gradio ke dalam FastAPI utama
app = gr.mount_to_fastapi(app, demo, path="/")

# JALANKAN LANGSUNG TANPA KONDISI 'if __name__ == "__main__":'
# Pastikan posisinya rata kiri (tanpa spasi/indentasi di depannya)
uvicorn.run(app, host="0.0.0.0", port=7860)