let currentQuery = "";
let totalRetrieved = 0;
let currentMode = "QE OFF";
async function searchDoc() {
    const queryInput = document.getElementById("query");
    const expandInput = document.getElementById("expand");
    const resultsDiv = document.getElementById("results");
    const evaluationResult = document.getElementById("evaluation-result");

    const q = queryInput.value.trim();
    currentQuery = q;
    const expand = expandInput.checked;

    const startTime = performance.now();

    // Reset evaluasi
    if (evaluationResult) {
        evaluationResult.innerHTML = "";
    }

    document.getElementById("groundTruth").value = "";
    document.getElementById("truePositive").value = "";

    if (!q) {
        resultsDiv.innerHTML = `
        <div class="result-card">
            <p>⚠️ Silakan masukkan kata kunci terlebih dahulu.</p>
        </div>
    `;
        return;
    }

    resultsDiv.innerHTML = `
    <div class="result-card">
        <p>🔍 Sedang mencari dokumen...</p>
    </div>
`;

    try {
        const response = await fetch(
            `http://127.0.0.1:8000/search-news?q=${encodeURIComponent(q)}&expand=${expand}`
        );

        if (!response.ok) {
            throw new Error(`HTTP Error ${response.status}`);
        }

        const data = await response.json();
        totalRetrieved = data.results.length;
        currentMode = expand ? "Dengan Query Expansion" : "Tanpa Query Expansion";

        const endTime = performance.now();

        const duration = (endTime - startTime).toFixed(0);

        document.getElementById("search-info").innerHTML = `

<div class="stats-bar">

<div class="stat">
    <span>${data.results.length}</span>
    <small>Hasil</small>
</div>

<div class="stat">
    <span>${duration}</span>
    <small>ms</small>
</div>

<div class="stat">
    <span>${expand ? "QE ON" : "QE OFF"}</span>
    <small>Mode</small>
</div>

</div>
`;

        if (!data.results || data.results.length === 0) {
            resultsDiv.innerHTML = `
            <div class="result-card">
                <p>Tidak ada dokumen yang relevan ditemukan.</p>
            </div>
        `;
            return;
        }

        resultsDiv.innerHTML = data.results.map((r, index) => `

<div class="result-card">

<div class="result-number">
    #${index + 1}
</div>

<h3>
    <a href="${r.url}" target="_blank">
        ${r.judul}
    </a>
</h3>

<div class="score">
    ⭐ Similarity Score : ${Number(r.score).toFixed(4)}
</div>

<div class="query-badge">
    🔎 ${r.expanded_query}
</div>

<p class="snippet">
    ${r.text_snippet}
</p>

<a href="${r.url}" target="_blank" class="read-more">
    Baca Selengkapnya →
</a>

</div>

`).join("");


    } catch (error) {
        console.error("Detail Error:", error);

        resultsDiv.innerHTML = `
        <div class="result-card">
            <p>❌ Terjadi kesalahan pada server.</p>
            <small>${error.message}</small>
        </div>
    `;
    }
}

function calculateMetrics() {
    const groundTruth =
        parseInt(document.getElementById("groundTruth").value);

    const TP =
        parseInt(document.getElementById("truePositive").value);

    if (isNaN(groundTruth) || isNaN(TP)) {
        alert("Lengkapi Ground Truth dan TP terlebih dahulu");
        return;
    }

    const FP = totalRetrieved - TP;
    const FN = groundTruth - TP;

    const precision =
        TP / (TP + FP);

    const recall =
        TP / (TP + FN);

    const f1 =
        2 * ((precision * recall) /
            (precision + recall));

    document.getElementById("evaluation-result").innerHTML = `

    <div class="eval-result">

        <h3>Hasil Evaluasi</h3>

        <div class="metric">
            Query : <b>${currentQuery}</b>
        </div>

        <div class="metric">
            Query Mode : <b>${currentMode}</b>
        </div>

        <div class="metric">
            Precision :
            <b>${(precision * 100).toFixed(2)}%</b>
        </div>

        <div class="metric">
            Recall :
            <b>${(recall * 100).toFixed(2)}%</b>
        </div>

        <div class="metric">
            F1 Score :
            <b>${(f1 * 100).toFixed(2)}%</b>
        </div>

        <hr>

        <div class="metric">
            Ground Truth : ${groundTruth}
        </div>

        <div class="metric">
            TP : ${TP}
        </div>

        <div class="metric">
            FP : ${FP}
        </div>

        <div class="metric">
            FN : ${FN}
        </div>

    </div>

    `;
}