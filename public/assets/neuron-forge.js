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

  const STORAGE_KEY = "neuronforge.workspace.v2";
  const MAX_SAVED_CHATS = 12;
  let workspace = loadWorkspace();

  function loadWorkspace() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
      if (saved && saved.version === 2 && Array.isArray(saved.chats)) {
        return {
          version: 2,
          chats: saved.chats.filter((chat) => chat && typeof chat.id === "string" && Array.isArray(chat.messages)),
          activeChatId: typeof saved.activeChatId === "string" ? saved.activeChatId : null
        };
      }
    } catch (_) {
      // Corrupt or disabled local storage should never prevent the chat UI from starting.
    }
    return { version: 2, chats: [], activeChatId: null };
  }

  function persistWorkspace() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(workspace));
      return true;
    } catch (_) {
      const status = $("#sidebar-status-detail");
      if (status) status.textContent = "Browser storage is full or unavailable";
      return false;
    }
  }

  function makeChatId() {
    return (window.crypto && typeof window.crypto.randomUUID === "function")
      ? window.crypto.randomUUID()
      : "chat-" + Date.now() + "-" + Math.random().toString(36).slice(2, 10);
  }

  function activeChat() {
    return workspace.chats.find((chat) => chat.id === workspace.activeChatId) || null;
  }

  function createChat() {
    const chat = { id: makeChatId(), title: "New conversation", messages: [], updatedAt: Date.now() };
    workspace.chats.unshift(chat);
    workspace.chats = workspace.chats.slice(0, MAX_SAVED_CHATS);
    workspace.activeChatId = chat.id;
    conversationHistory = [];
    messages.innerHTML = welcomeMarkup;
    bindPromptCards();
    persistWorkspace();
    renderSavedChats();
    return chat;
  }

  function persistCurrentChat() {
    let chat = activeChat();
    if (!chat) chat = createChat();
    chat.messages = conversationHistory.slice(-40).map((item) => ({
      role: item.role,
      content: String(item.content || "").slice(0, 1800)
    }));
    const firstUser = chat.messages.find((item) => item.role === "user");
    if (firstUser) {
      chat.title = firstUser.content.replace(/\s+/g, " ").trim().slice(0, 52) || "Conversation";
    }
    chat.updatedAt = Date.now();
    workspace.chats.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
    workspace.chats = workspace.chats.slice(0, MAX_SAVED_CHATS);
    persistWorkspace();
    renderSavedChats();
  }

  function renderSavedChats() {
    const host = $("#chat-collection");
    if (!host) return;
    host.innerHTML = "";
    const chats = workspace.chats.filter((chat) => chat.messages && chat.messages.length).slice(0, 8);
    if (!chats.length) {
      const empty = document.createElement("div");
      empty.className = "collection-empty";
      empty.textContent = "Your conversations will appear here";
      host.appendChild(empty);
      return;
    }
    chats.forEach((chat) => {
      const button = document.createElement("button");
      button.className = "collection-link" + (chat.id === workspace.activeChatId ? " active" : "");
      button.type = "button";
      const dot = document.createElement("span");
      dot.className = "book-dot";
      const title = document.createElement("span");
      title.textContent = chat.title || "Conversation";
      button.append(dot, title);
      button.title = chat.title || "Conversation";
      button.addEventListener("click", () => openSavedChat(chat.id));
      host.appendChild(button);
    });
  }

  function openSavedChat(chatId) {
    const chat = workspace.chats.find((item) => item.id === chatId);
    if (!chat) return;
    workspace.activeChatId = chatId;
    conversationHistory = chat.messages.filter((item) => item && ["user", "assistant"].includes(item.role) && typeof item.content === "string").map((item) => ({ role: item.role, content: item.content }));
    messages.innerHTML = "";
    if (!conversationHistory.length) {
      messages.innerHTML = welcomeMarkup;
      bindPromptCards();
    } else {
      conversationHistory.forEach((item) => addMessage(item.content, item.role));
    }
    persistWorkspace();
    renderSavedChats();
    switchView("chat");
  }

  function buildMemoryContext() {
    const secretPattern = /\b(password|passcode|api[ -]?key|secret|access token|refresh token|credit card|cvv|one[- ]time password|otp)\b/i;
    const records = [];
    workspace.chats
      .filter((chat) => chat.id !== workspace.activeChatId)
      .sort((a, b) => (a.updatedAt || 0) - (b.updatedAt || 0))
      .forEach((chat) => {
        (chat.messages || []).forEach((item) => {
          if (!item || !["user", "assistant"].includes(item.role) || typeof item.content !== "string") return;
          const content = item.content.trim();
          if (!content || secretPattern.test(content)) return;
          records.push({ role: item.role, content: content.slice(0, 700) });
        });
      });
    return records.slice(-20);
  }

  function restoreWorkspace() {
    const chat = activeChat() || createChat();
    conversationHistory = (chat.messages || []).filter((item) => item && ["user", "assistant"].includes(item.role) && typeof item.content === "string").map((item) => ({ role: item.role, content: item.content }));
    if (conversationHistory.length) {
      messages.innerHTML = "";
      conversationHistory.forEach((item) => addMessage(item.content, item.role));
    }
    renderSavedChats();
  }

  const welcomeMarkup = `
    <div class="welcome-state" id="welcome-state">
      <div class="welcome-orbit"><span class="orbit-dot"></span><span class="orbit-core">N</span><span class="orbit-star">✦</span></div>
      <h2>What are you thinking about?</h2>
      <p>Ask a question, explore an idea, or start a conversation about Shakespeare.</p>
      <div class="prompt-grid">
        <button class="prompt-card" data-prompt="Help me break down a difficult idea" type="button"><span class="prompt-icon violet">♜</span><span><strong>Understand an idea</strong><small>Explain it clearly, step by step</small></span><span class="prompt-arrow">↗</span></button>
        <button class="prompt-card" data-prompt="Help me plan a project from scratch" type="button"><span class="prompt-icon rose">♡</span><span><strong>Plan a project</strong><small>Turn a rough idea into a clear plan</small></span><span class="prompt-arrow">↗</span></button>
        <button class="prompt-card" data-prompt="Help me solve a problem step by step" type="button"><span class="prompt-icon amber">❋</span><span><strong>Work through a problem</strong><small>Break it into manageable steps</small></span><span class="prompt-arrow">↗</span></button>
        <button class="prompt-card" data-prompt="Help me improve a piece of writing" type="button"><span class="prompt-icon teal">✧</span><span><strong>Improve my writing</strong><small>Make it clearer and more effective</small></span><span class="prompt-arrow">↗</span></button>
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
    return { node, body, label };
  }

  function clearConversation() {
    messages.innerHTML = welcomeMarkup;
    conversationHistory = [];
    const chat = activeChat();
    if (chat) {
      chat.messages = [];
      chat.title = "New conversation";
      chat.updatedAt = Date.now();
    }
    persistWorkspace();
    renderSavedChats();
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
    persistCurrentChat();
    input.value = "";
    updateCount();
    sendButton.disabled = true;
    const pending = addMessage("Thinking through your question…", "assistant");
    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          messages: conversationHistory.slice(-18).map((item) => ({ role: item.role, content: item.content.slice(0, 1500) })),
          memories: buildMemoryContext()
        }),
        cache: "no-store"
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "The request could not be completed.");
      pending.body.textContent = data.text || "I couldn't form an answer to that question.";
      if (data.model_label) pending.label.textContent = data.model_label;
      if (data.neural_experimental) pending.node.classList.add("experimental-neural");
      if (data.text) {
        conversationHistory.push({ role: "assistant", content: data.text });
        persistCurrentChat();
      }
    } catch (error) {
      conversationHistory.pop();
      persistCurrentChat();
      pending.body.textContent = "I couldn't reach the local assistant: " + error.message;
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
      const assistantReady = Boolean(data.assistant_ready);
      $("#top-status").textContent = assistantReady ? "Neuron Forge · Local AI" : "Local assistant unavailable";
      $("#sidebar-status").textContent = assistantReady ? "Local assistant ready" : "Local assistant unavailable";
      $("#sidebar-status-detail").textContent = assistantReady
        ? "browser chat context · arithmetic · reference search"
        : "The local conversation service could not be reached";
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
    createChat();
    input.value = "";
    updateCount();
    input.focus();
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

  restoreWorkspace();
  bindPromptCards();
  loadLibrary();
})();
