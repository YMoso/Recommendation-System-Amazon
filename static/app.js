const API_BASE = "/api";

function showLoading(show) {
    document.getElementById("loading").style.display = show ? "block" : "none";
}

function resetUI() {
    document.getElementById("historyList").innerHTML = "";
    document.getElementById("recommendationsList").innerHTML = "";
    document.getElementById("historySection").style.display = "none";
    document.getElementById("recommendationsSection").style.display = "none";
    document.getElementById("userStatus").innerHTML = "";
}

async function loadRandomUser(type) {
    resetUI();
    showLoading(true);

    const endpoint = type === "active" ? "/random/active" : "/random/cold";
    const res = await fetch(API_BASE + endpoint);
    const data = await res.json();

    document.getElementById("userId").value = data.user_id;
    await getRecommendations();
}

async function getRecommendations() {
    const userId = document.getElementById("userId").value.trim();
    if (!userId) return;

    resetUI();
    showLoading(true);

    const res = await fetch(API_BASE + "/recommend", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({user_id: userId, k: 5})
    });

    const data = await res.json();
    showLoading(false);

    renderHistory(data.history);
    renderRecommendations(data.recommendations);
}

function renderHistory(history) {
    if (!history || history.length === 0) return;

    const container = document.getElementById("historyList");
    document.getElementById("historySection").style.display = "block";

    container.innerHTML = history.map(item => `
        <div class="card">
            <h3>${item.title}</h3>
            <div class="meta">
                Category: ${item.category || "Unknown"}<br>
                Rating: ⭐ ${item.rating}<br>
                ASIN: ${item.asin}
            </div>
        </div>
    `).join("");
}

function renderRecommendations(recs) {
    if (!recs || recs.length === 0) return;

    const container = document.getElementById("recommendationsList");
    document.getElementById("recommendationsSection").style.display = "block";

    container.innerHTML = recs.map(item => `
        <div class="card">
            <h3>${item.title}</h3>
            <div class="meta">
                Category: ${item.category || "Unknown"}<br>
                Score: ${item.score.toFixed(2)}<br>
                ASIN: ${item.asin}
            </div>
            <span class="badge">${item.model}</span>
        </div>
    `).join("");
}