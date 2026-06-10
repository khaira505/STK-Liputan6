import os
import pickle
import string
import itertools
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = FastAPI()

# Izinkan CORS agar frontend bisa memanggil tanpa diblokir
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Menyesuaikan path folder data sesuai struktur folder VS Code kamu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data')

# Load data pickle lokal
with open(os.path.join(DATA_PATH, 'articles.pkl'), 'rb') as f:
    paper = pickle.load(f)
with open(os.path.join(DATA_PATH, 'processed_paper.pkl'), 'rb') as f:
    processed_paper = pickle.load(f)
with open(os.path.join(DATA_PATH, 'thesaurus.pkl'), 'rb') as f:
    thesaurus = pickle.load(f)

# Setup Sastrawi & Vectorizer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_factory = StopWordRemoverFactory()
stopword_remover = stop_factory.create_stop_word_remover()
vectorizer = TfidfVectorizer(use_idf=True)

# Tambahkan rute fallback ke / agar Vercel mendeteksi backend aktif
@app.get("/")
def read_root():
    return {"status": "Backend SvelteKit/FastAPI aktif dan berjalan lancar"}

# Rute Utama Pencarian yang dipanggil oleh vercel.json dan index.html kamu
@app.get("/search-news")
def search_api(q: str = "", expand: str = "true"):
    if not q.strip():
        return {"results": []}
        
    use_expansion = expand.lower() == "true"
    query = q.lower()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    query = query.translate(remove_punctuation_map)
    query = stopword_remover.remove(query)
    query = query.split()
    query = [stemmer.stem(x) for x in query]

    if not use_expansion:
        x = [' '.join(query)]
        paper_tfidf = vectorizer.fit_transform(x + processed_paper)
        q_vec = paper_tfidf[0]
        result = cosine_similarity(paper_tfidf, q_vec)
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
            q_vec = paper_tfidf[0]
            result = cosine_similarity(paper_tfidf, q_vec)
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
        
        doc_idx = item[0] - 1
        
        results_list.append({
            "judul": paper[doc_idx]['judul'],
            "score": float(item[1]),
            "expanded_query": item[2][0],
            "text_snippet": paper[doc_idx]['text'][:300] + "...",
            "url": paper[doc_idx]['url']
        })
        shown += 1
        
    return {"results": results_list}