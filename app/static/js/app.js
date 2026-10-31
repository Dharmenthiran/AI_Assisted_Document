/**
 * Document AI Assistant — Frontend Client Application
 * Handles document upload, status polling, chat interactions, and source rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const uploadFeedback = document.getElementById("upload-feedback");
  const documentsList = document.getElementById("documents-list");
  const emptyDocsState = document.getElementById("empty-docs-state");
  const docCountSpan = document.getElementById("doc-count");
  const refreshDocsBtn = document.getElementById("refresh-docs-btn");
  const docFilterSelect = document.getElementById("doc-filter-select");
  const modelSelect = document.getElementById("model-select");
  const clearChatBtn = document.getElementById("clear-chat-btn");
  const chatMessages = document.getElementById("chat-messages");
  const welcomeCard = document.getElementById("welcome-card");
  const chatForm = document.getElementById("chat-form");
  const questionInput = document.getElementById("question-input");
  const sendBtn = document.getElementById("send-btn");
  const systemStatusPill = document.getElementById("system-status-pill");
  const systemStatusText = document.getElementById("system-status-text");
  const llmWarningBar = document.getElementById("llm-warning-bar");
  const toastContainer = document.getElementById("toast-container");
  const specLlm = document.getElementById("spec-llm");
  const modelSpeedWarning = document.getElementById("model-speed-warning");

  let isGenerating = false;
  let activeDocuments = [];
  let availableModelsLoaded = false;

  // --------------------------------------------------------------------------
  // Utility Functions
  // --------------------------------------------------------------------------

  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(20px)";
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  function formatBytes(bytes) {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function checkModelSpeed(modelName) {
    if (!modelSpeedWarning) return;
    const m = (modelName || "").toLowerCase();
    if (m.includes("7b") || m.includes("8b") || m.includes("14b") || m.includes("70b")) {
      modelSpeedWarning.innerHTML = `⚠️ <strong>7B model on CPU:</strong> May take 3-5+ mins. <button type="button" id="switch-to-fast-btn" style="background:none;border:none;color:#60a5fa;cursor:pointer;text-decoration:underline;padding:0;font-size:inherit;">Switch to Fast Model (qwen3:0.6b)</button>`;
      modelSpeedWarning.classList.remove("hidden");
      const switchBtn = document.getElementById("switch-to-fast-btn");
      if (switchBtn) {
        switchBtn.onclick = (e) => {
          e.preventDefault();
          selectFastestModel();
        };
      }
    } else {
      modelSpeedWarning.classList.add("hidden");
      modelSpeedWarning.innerHTML = "";
    }
  }

  function selectFastestModel() {
    if (!modelSelect) return;
    const fastOpt = Array.from(modelSelect.options).find(o => o.value.includes("0.6b") || o.value.includes("tiny") || o.value.includes("3b"));
    if (fastOpt) {
      modelSelect.value = fastOpt.value;
      localStorage.setItem("document_rag_selected_model", fastOpt.value);
      if (specLlm) specLlm.textContent = fastOpt.value;
      checkModelSpeed(fastOpt.value);
      showToast(`Switched to fast CPU model: ${fastOpt.value}`, "success");
    }
  }

  // --------------------------------------------------------------------------
  // System Health & Model Discovery
  // --------------------------------------------------------------------------

  async function checkSystemHealth() {
    try {
      const res = await fetch("/health");
      const data = await res.json();
      const ollama = data.dependencies?.ollama;

      if (ollama?.available_models && ollama.available_models.length > 0) {
        updateModelDropdown(ollama.available_models, ollama.configured_model);
      }

      if (data.status === "healthy" && ollama?.online && ollama?.model_present) {
        systemStatusPill.className = "status-pill status-online";
        systemStatusText.textContent = "System Ready (LLM Online)";
        llmWarningBar.classList.add("hidden");
      } else if (ollama && !ollama.online) {
        systemStatusPill.className = "status-pill status-offline";
        systemStatusText.textContent = "Ollama Offline";
        llmWarningBar.innerHTML = `⚠️ <strong>Ollama service is unreachable</strong> at ${ollama.configured_model || 'localhost'}. Ensure <code>ollama serve</code> is running.`;
        llmWarningBar.classList.remove("hidden");
      } else if (ollama && !ollama.model_present) {
        systemStatusPill.className = "status-pill status-offline";
        systemStatusText.textContent = "Model Missing";
        llmWarningBar.innerHTML = `⚠️ Configured model <strong>${ollama.configured_model}</strong> is not installed in Ollama. Run: <code>ollama pull ${ollama.configured_model}</code>`;
        llmWarningBar.classList.remove("hidden");
      } else {
        systemStatusPill.className = "status-pill status-checking";
        systemStatusText.textContent = "Degraded";
      }
    } catch (err) {
      systemStatusPill.className = "status-pill status-offline";
      systemStatusText.textContent = "Backend Unreachable";
    }
  }

  // Restore saved model preference from localStorage
  const storedModel = localStorage.getItem("document_rag_selected_model");
  if (storedModel && modelSelect) {
    const hasOpt = Array.from(modelSelect.options).some(o => o.value === storedModel);
    if (hasOpt) {
      modelSelect.value = storedModel;
      if (specLlm) specLlm.textContent = storedModel;
    }
  }
  checkModelSpeed(modelSelect ? modelSelect.value : "");

  function updateModelDropdown(models, defaultModel) {
    if (!modelSelect || !models || models.length === 0) return;
    const currentVal = localStorage.getItem("document_rag_selected_model") || modelSelect.value || defaultModel;
    
    // Check if models actually changed to avoid resetting user selection
    const existingOptions = Array.from(modelSelect.options).map(o => o.value);
    const hasChanged = models.length !== existingOptions.length || models.some(m => !existingOptions.includes(m));
    
    if (hasChanged || !availableModelsLoaded) {
      modelSelect.innerHTML = "";
      models.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m;
        let label = m;
        if (m === defaultModel) label += " (Default)";
        if (m.includes("0.6b")) label += " ⚡⚡ Fast CPU (~40s)";
        else if (m.includes("tiny")) label += " ⚡ Fast CPU";
        else if (m.includes("1.5b")) label += " ⚡ Reasoning (~2m)";
        else if (m.includes("3b") || m.includes("3.2")) label += " Balanced (~1.8m)";
        else if (m.includes("7b") || m.includes("8b")) label += " 🐢 Heavy CPU (3-5m)";
        opt.textContent = label;
        if (m === currentVal) {
          opt.selected = true;
        }
        modelSelect.appendChild(opt);
      });
      availableModelsLoaded = true;
      if (specLlm && modelSelect.value) {
        specLlm.textContent = modelSelect.value;
      }
      checkModelSpeed(modelSelect.value);
    }
  }

  if (modelSelect) {
    modelSelect.addEventListener("change", () => {
      localStorage.setItem("document_rag_selected_model", modelSelect.value);
      if (specLlm) specLlm.textContent = modelSelect.value;
      checkModelSpeed(modelSelect.value);
      showToast(`Active model switched to: ${modelSelect.value}`, "info");
    });
  }

  // --------------------------------------------------------------------------
  // Document Management (Upload, List, Delete)
  // --------------------------------------------------------------------------

  async function fetchDocuments() {
    try {
      const res = await fetch("/api/documents");
      if (!res.ok) throw new Error("Failed to load documents");
      const data = await res.json();
      activeDocuments = data.documents || [];
      renderDocumentList(activeDocuments);
    } catch (err) {
      console.error("Error loading documents:", err);
    }
  }

  function renderDocumentList(docs) {
    docCountSpan.textContent = docs.length;

    // Update filter dropdown
    const currentVal = docFilterSelect.value;
    docFilterSelect.innerHTML = '<option value="">All Uploaded Documents</option>';
    docs.forEach((doc) => {
      const opt = document.createElement("option");
      opt.value = doc.document_id;
      opt.textContent = doc.filename;
      if (doc.document_id === currentVal) opt.selected = true;
      docFilterSelect.appendChild(opt);
    });

    if (docs.length === 0) {
      documentsList.innerHTML = "";
      documentsList.appendChild(emptyDocsState);
      return;
    }

    documentsList.innerHTML = "";
    docs.forEach((doc) => {
      const card = document.createElement("div");
      card.className = "doc-item-card";
      card.innerHTML = `
        <div class="doc-meta-col">
          <span class="doc-filename" title="${doc.filename}">${escapeHtml(doc.filename)}</span>
          <div class="doc-badges">
            <span class="doc-badge-pill">${doc.page_count} page(s)</span>
            <span class="doc-badge-pill">${doc.chunk_count} chunks</span>
            <span class="doc-badge-pill">${formatBytes(doc.file_size_bytes)}</span>
          </div>
        </div>
        <button class="icon-btn delete-doc-btn" data-id="${doc.document_id}" data-filename="${escapeHtml(doc.filename)}" title="Delete document and remove vector records">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
          </svg>
        </button>
      `;
      documentsList.appendChild(card);
    });

    // Attach delete listeners
    document.querySelectorAll(".delete-doc-btn").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        const docId = btn.getAttribute("data-id");
        const filename = btn.getAttribute("data-filename");
        if (confirm(`Are you sure you want to delete '${filename}' and purge its indexed vector records?`)) {
          await deleteDocument(docId, filename);
        }
      });
    });
  }

  async function deleteDocument(docId, filename) {
    try {
      const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Deletion failed");
      }
      showToast(`Deleted '${filename}' and purged vector records.`, "success");
      await fetchDocuments();
      await checkSystemHealth();
    } catch (err) {
      showToast(`Error deleting document: ${err.message}`, "error");
    }
  }

  async function handleFileUpload(file) {
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadFeedback("Only PDF files (.pdf) are supported.", "error");
      return;
    }

    if (file.size > 15 * 1024 * 1024) {
      setUploadFeedback("File size exceeds maximum allowed 15MB.", "error");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setUploadFeedback("Ingesting PDF: extracting text, chunking & generating embeddings...", "loading");

    try {
      const res = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Upload and indexing failed.");
      }

      setUploadFeedback(`Successfully indexed '${data.filename}' (${data.chunk_count} chunks created).`, "success");
      showToast(`Document indexed successfully!`, "success");
      fileInput.value = "";
      await fetchDocuments();
      await checkSystemHealth();

      setTimeout(() => {
        uploadFeedback.classList.add("hidden");
      }, 5000);
    } catch (err) {
      setUploadFeedback(`Upload failed: ${err.message}`, "error");
      showToast(`Error: ${err.message}`, "error");
    }
  }

  function setUploadFeedback(msg, type) {
    uploadFeedback.className = `feedback-box ${type}`;
    uploadFeedback.textContent = msg;
    uploadFeedback.classList.remove("hidden");
  }

  // Dropzone events
  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-active");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-active");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  refreshDocsBtn.addEventListener("click", fetchDocuments);

  // --------------------------------------------------------------------------
  // Chat Interaction
  // --------------------------------------------------------------------------

  // Auto-resize input
  questionInput.addEventListener("input", () => {
    questionInput.style.height = "auto";
    questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + "px";
  });

  // Enter to send, Shift+Enter for newline
  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isGenerating) {
        chatForm.dispatchEvent(new Event("submit"));
      }
    }
  });

  clearChatBtn.addEventListener("click", () => {
    chatMessages.innerHTML = "";
    chatMessages.appendChild(welcomeCard);
    welcomeCard.classList.remove("hidden");
  });

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question || isGenerating) return;

    if (activeDocuments.length === 0) {
      showToast("Please upload and index a PDF document first.", "error");
      return;
    }

    // Hide welcome card once chat starts
    welcomeCard.classList.add("hidden");

    // Append user message
    appendUserMessage(question);
    questionInput.value = "";
    questionInput.style.height = "auto";

    // Set loading state
    isGenerating = true;
    sendBtn.disabled = true;
    const typingIndicator = appendTypingIndicator();
    scrollToBottom();

    try {
      const selectedDocId = docFilterSelect.value || null;
      const selectedModel = modelSelect ? modelSelect.value : null;

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question,
          document_id: selectedDocId,
          model: selectedModel,
        }),
      });

      const data = await res.json();
      typingIndicator.remove();

      if (!res.ok) {
        appendErrorMessage(data.detail || "Failed to generate answer from RAG pipeline.");
      } else {
        appendAssistantMessage(data.answer, data.sources, data.context_used, data.warning, data.model);
      }
    } catch (err) {
      typingIndicator.remove();
      appendErrorMessage(`Network or server error: ${err.message}`);
    } finally {
      isGenerating = false;
      sendBtn.disabled = false;
      questionInput.focus();
      scrollToBottom();
    }
  });

  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user-row";
    row.innerHTML = `
      <span class="message-sender">You</span>
      <div class="bubble user-bubble">${escapeHtml(text)}</div>
    `;
    chatMessages.appendChild(row);
    scrollToBottom();
  }

  function appendTypingIndicator() {
    const row = document.createElement("div");
    row.className = "message-row assistant-row";
    row.innerHTML = `
      <span class="message-sender">Assistant</span>
      <div class="typing-indicator">
        <div class="typing-dots">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
        <span>Retrieving context & generating answer...</span>
      </div>
    `;
    chatMessages.appendChild(row);
    return row;
  }

  function appendAssistantMessage(answer, sources, contextUsed, warning, modelUsed) {
    const row = document.createElement("div");
    row.className = "message-row assistant-row";

    let sourcesHtml = "";
    if (sources && sources.length > 0) {
      const sourceItems = sources
        .map(
          (s, idx) => `
          <div class="source-card">
            <div class="source-header">
              <div class="source-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                </svg>
                <span>${escapeHtml(s.filename)}</span>
                <span class="source-page-badge">Page ${s.page}</span>
              </div>
              ${s.score !== null ? `<span class="source-score" title="Cosine Similarity Score">Sim: ${(s.score * 100).toFixed(1)}%</span>` : ""}
            </div>
            <div class="source-snippet">${escapeHtml(s.snippet)}</div>
          </div>
        `
        )
        .join("");

      sourcesHtml = `
        <div class="sources-container">
          <button class="sources-toggle" type="button">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
            </svg>
            <span>Verified Sources (${sources.length} citations)</span>
          </button>
          <div class="sources-list">${sourceItems}</div>
        </div>
      `;
    }

    let warningHtml = warning
      ? `<div style="margin-top: 8px; font-size: 0.75rem; color: var(--accent-warning);">ℹ️ ${escapeHtml(warning)}</div>`
      : "";

    const modelBadge = modelUsed
      ? `<span class="font-mono text-xs" style="color: var(--accent-cyan); margin-left: 6px;">[${escapeHtml(modelUsed)}]</span>`
      : "";

    row.innerHTML = `
      <span class="message-sender">Assistant ${modelBadge}</span>
      <div class="bubble assistant-bubble">${escapeHtml(answer)}</div>
      ${warningHtml}
      ${sourcesHtml}
    `;

    chatMessages.appendChild(row);

    // Accordion toggle
    const toggleBtn = row.querySelector(".sources-toggle");
    if (toggleBtn) {
      const list = row.querySelector(".sources-list");
      toggleBtn.addEventListener("click", () => {
        const isHidden = list.style.display === "none";
        list.style.display = isHidden ? "flex" : "none";
      });
    }

    scrollToBottom();
  }

  function appendErrorMessage(errorMsg) {
    const row = document.createElement("div");
    row.className = "message-row assistant-row";
    row.innerHTML = `
      <span class="message-sender">Assistant</span>
      <div class="bubble assistant-bubble" style="border-color: rgba(239,68,68,0.4); color: #fca5a5;">
        ⚠️ <strong>Error:</strong> ${escapeHtml(errorMsg)}
      </div>
    `;
    chatMessages.appendChild(row);
    scrollToBottom();
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Initial startup calls
  checkSystemHealth();
  fetchDocuments();
  // Periodic health check every 15 seconds
  setInterval(checkSystemHealth, 15000);
});
