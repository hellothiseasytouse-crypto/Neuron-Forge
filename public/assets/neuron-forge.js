(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const shell = $("#app-shell");
  const sidebar = $("#sidebar");
  const messages = $("#messages");
  const composer = $("#composer");
  const input = $("#message-input");
  const sendButton = $("#send-button");
  const worksGrid = $("#works-grid");
  const worksSearch = $("#works-search");
  let catalog = [];
  let conversationHistory = [];
  let sending = false;

  const welcomeMarkup = `
    <div class="welcome-state" id="welcome-state">
      <div class="welcome-orbit"><span class="orbit-dot"></span><span class="orbit-core">N</span><span class="orbit-star">✦</span></div>
      <h2>What are you thinking about?</h2>
      <p>Ask a question, explore an idea, or start a conversation about Shakespeare.</p>
      <div class="prompt-grid">
        <button class="prompt-card" data-prompt="Find passages about ambition in Macbeth" type="button"><span class="prompt-icon violet">♜</span><span><strong>Ambition in Macbeth</strong><small>Power, choice, and consequence</small></span><span class="prompt-arrow">↗</span></button>
        <button class="prompt-card" data-prompt="Show me passages about love in Romeo and Juliet" type="button"><span class="prompt-icon rose">♡</span><span><strong>Love in Romeo and Juliet</strong><small>Famous scenes and lines</small></span><span class="prompt-arrow">↗</span></button>
        <button class="prompt-card" data-prompt="Find a sonnet about time and beauty" type="button"><span class="prompt-icon amber">❋</span><span><strong>Time and beauty</strong><small>Explore the sonnets</small></span><span class="prompt-arrow">↗</span></button>
        <button class="prompt-card" data-prompt="Find passages about magic in The Tempest" type="button"><span class="prompt-icon teal">✧</span><span><strong>Magic in The Tempest</strong><small>Islands, spirits, and wonder</small></span><span class="prompt-arrow">↗</span></button>
      </div>
    </div>`;

  function switchView(view) {
    const names = { chat: "Chat", works: "Shakespeare library", about: "About Neuron Forge" };
    for (const key of Object.keys(names)) {
      const section = document.getElementById(key + "-view");
      const active = key === view;
      section.hidden = !active;
      section.classList.toggle("active", active);
    }
    document.querySelectorAll("[data-view]").forEach((button) => {
      button.classList.toggle("active", button.dataset.view === view);
    });
    $("#view-heading").textContent = names[view] || "Chat";
    shell.classList.remove("sidebar-open");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function addMessage(text, kind) {
    $("#welcome-state")?.remove();
    const node = document.createElement("article");
    node.className = "message " + kind;
    const label = document.createElement("span");
    label.className = "message-label";
    label.textContent = kind === "user" ? "YOU" : "NEURON FORGE";
    const body = document.createElement("span");
    body.className = "message-body";
    body.textContent = text;
    node.append(label, body);
    messages.appendChild(node);
    messages.scrollTop = messages.scrollHeight;
    return { node, body };
  }

  function clearConversation() {
    messages.innerHTML = welcomeMarkup;
    conversationHistory = [];
    bindPromptCards();
    input.value = "";
    updateCount();
    input.focus();
  }

  function bindPromptCards() {
    document.querySelectorAll("[data-prompt]").forEach((button) => {
      button.addEventListener("click", () => {
        switchView("chat");
        submitPrompt(button.dataset.prompt);
      }, { once: true });
    });
  }

  async function submitPrompt(value) {
    const prompt = String(value || "").trim();
    if (!prompt || sending) return;
    sending = true;
    addMessage(prompt, "user");
    conversationHistory.push({ role: "user", content: prompt });
    conversationHistory = conversationHistory.slice(-10);
    input.value = "";
    updateCount();
    sendButton.disabled = true;
    const pending = addMessage("Thinking through your question…", "assistant");
    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, messages: conversationHistory }),
        cache: "no-store"
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "The request could not be completed.");
      pending.body.textContent = data.text || "I couldn't form an answer to that question.";
      if (data.text) {
        conversationHistory.push({ role: "assistant", content: data.text });
        conversationHistory = conversationHistory.slice(-10);
      }
    } catch (error) {
      conversationHistory.pop();
      pending.body.textContent = "I couldn't reach the AI model: " + error.message;
      pending.node.classList.add("error");
    } finally {
      sending = false;
      sendButton.disabled = false;
      messages.scrollTop = messages.scrollHeight;
      input.focus();
    }
  }

  function updateCount() {
    $("#composer-count").textContent = input.value.length.toLocaleString() + " / 2000";
  }

  function formatNumber(value) {
    const n = Number(value || 0);
    return n.toLocaleString();
  }

  function renderSidebar(works) {
    const host = $("#sidebar-collection");
    host.innerHTML = "";
    works.slice(0, 6).forEach((work) => {
      const button = document.createElement("button");
      button.className = "collection-link";
      button.type = "button";
      const dot = document.createElement("span");
      dot.className = "book-dot";
      const name = document.createElement("span");
      name.textContent = work.title;
      button.append(dot, name);
      button.addEventListener("click", () => {
        switchView("chat");
        submitPrompt("Find passages from " + work.title);
      });
      host.appendChild(button);
    });
  }

  function renderWorks(works, search = "") {
    const query = search.trim().toLowerCase();
    const filtered = works.filter((work) => {
      return !query || work.title.toLowerCase().includes(query) || work.category.toLowerCase().includes(query);
    });
    worksGrid.innerHTML = "";
    $("#library-count").textContent = filtered.length + (filtered.length === 1 ? " work" : " works");
    if (!filtered.length) {
      const empty = document.createElement("div");
      empty.className = "works-loading";
      empty.textContent = "No titles match that search.";
      worksGrid.appendChild(empty);
      return;
    }
    filtered.forEach((work) => {
      const card = document.createElement("button");
      card.className = "work-card";
      card.type = "button";
      const top = document.createElement("div");
      top.className = "work-card-top";
      const category = document.createElement("span");
      category.className = "work-category";
      category.textContent = work.category;
      const arrow = document.createElement("span");
      arrow.className = "work-arrow";
      arrow.textContent = "↗";
      top.append(category, arrow);
      const title = document.createElement("strong");
      title.textContent = work.title;
      const detail = document.createElement("small");
      detail.textContent = formatNumber(work.words) + " words · stored in this project";
      card.append(top, title, detail);
      card.addEventListener("click", () => {
        switchView("chat");
        submitPrompt("Find passages from " + work.title);
      });
      worksGrid.appendChild(card);
    });
  }

  async function loadLibrary() {
    try {
      const response = await fetch("/api/generate", { cache: "no-store" });
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.error || "Library unavailable");
      catalog = Array.isArray(data.works_catalog) ? data.works_catalog : [];
      $("#top-status").textContent = data.assistant_ready ? "Local chat ready" : "Local chat unavailable";
      $("#sidebar-status").textContent = data.assistant_ready ? "Local chat ready" : "Local chat unavailable";
      $("#sidebar-status-detail").textContent = data.assistant_ready
        ? "Offline chat · FAI source loaded"
        : "Local engine unavailable";
      $("#status-pip").classList.toggle("ready", Boolean(data.assistant_ready));
      $("#stat-works").textContent = formatNumber(catalog.length);
      $("#stat-words").textContent = formatNumber(data.training_words);
      $("#stat-passages").textContent = formatNumber(data.passages);
      renderSidebar(catalog);
      renderWorks(catalog);
    } catch (error) {
      $("#top-status").textContent = "Assistant unavailable";
      $("#sidebar-status").textContent = "Offline";
      $("#sidebar-status-detail").textContent = error.message;
    }
  }

  document.querySelectorAll(".nav-item[data-view], .status-info[data-view]").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
  $("#collapse-sidebar").addEventListener("click", () => shell.classList.toggle("sidebar-collapsed"));
  $("#open-sidebar").addEventListener("click", () => shell.classList.add("sidebar-open"));
  $("#mobile-scrim").addEventListener("click", () => shell.classList.remove("sidebar-open"));
  $("#new-chat").addEventListener("click", () => {
    switchView("chat");
    clearConversation();
  });
  $("#clear-chat").addEventListener("click", clearConversation);
  composer.addEventListener("submit", (event) => {
    event.preventDefault();
    submitPrompt(input.value);
  });
  input.addEventListener("input", updateCount);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      composer.requestSubmit();
    }
  });
  worksSearch.addEventListener("input", () => renderWorks(catalog, worksSearch.value));
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      switchView("chat");
      clearConversation();
    }
    if (event.key === "Escape") shell.classList.remove("sidebar-open");
  });

  bindPromptCards();
  loadLibrary();
})();
