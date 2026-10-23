(function () {
  const { api, t, toast, showStatus, hide, setButtonBusy,
          escapeHtml, qs } = window.UG;

  let ws = null;
  let connected = false;
  let streaming = false;
  let models = [];
  let loras = [];
  let history = [];
  let currentBubble = null;
  let currentText = '';
  let currentUser = '';
  let reconnectTimer = null;

  function setConn(on) {
    connected = !!on;
    const el = qs('#chat-conn');
    el.textContent = on ? t('web.chat.connected') : t('web.chat.disconnected');
    el.className = 'badge ' + (on ? 'badge-success' : 'badge-ghost');
  }

  function setStreaming(on) {
    streaming = !!on;
    qs('#chat-send').disabled = on;
    qs('#chat-stop').classList.toggle('hidden', !on);
    qs('#chat-typing').classList.toggle('hidden', !on);
  }

  function scrollBottom() {
    const log = qs('#chat-log');
    log.scrollTop = log.scrollHeight;
  }

  function markdown(text) {
    let html = escapeHtml(text || '');
    try {
      const marked = window.marked;
      if (marked) {
        html = typeof marked.parse === 'function' ? marked.parse(text || '') : marked(text || '');
        if (window.DOMPurify) html = window.DOMPurify.sanitize(html);
      }
    } catch (e) { html = escapeHtml(text || ''); }
    return html;
  }

  function addBubble(role, text) {
    const empty = qs('#chat-empty');
    if (empty) empty.remove();
    const wrap = document.createElement('div');
    wrap.className = 'chat ' + (role === 'user' ? 'chat-end' : 'chat-start');
    const head = document.createElement('div');
    head.className = 'chat-header text-sm muted';
    head.textContent = role === 'user' ? t('web.chat.you') : t('web.chat.assistant');
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble ' + (role === 'user' ? 'chat-bubble-primary' : '');
    if (role === 'assistant') bubble.innerHTML = markdown(text);
    else bubble.textContent = text;
    wrap.appendChild(head);
    wrap.appendChild(bubble);
    qs('#chat-log').appendChild(wrap);
    scrollBottom();
    return bubble;
  }

  function renderBubble(bubble, text) {
    if (bubble) bubble.innerHTML = markdown(text);
    scrollBottom();
  }

  function resetChat() {
    history = [];
    currentBubble = null;
    currentText = '';
    const log = qs('#chat-log');
    log.innerHTML = '<div id="chat-empty" class="chat-empty muted">' +
      escapeHtml(t('web.chat.empty')) + '</div>';
  }

  async function loadModels() {
    try {
      const res = await api('/api/models');
      models = res.data || [];
      const sel = qs('#chat-base');
      const cur = sel.value;
      sel.innerHTML = models.map((m) =>
        '<option value="' + escapeHtml(m.display_name) + '">' +
        escapeHtml(m.display_name) + '</option>').join('');
      if (cur && models.some((m) => m.display_name === cur)) sel.value = cur;
    } catch (e) { toast(e.message, 'error'); }
  }

  function renderLoraInfo() {
    const name = qs('#chat-lora').value;
    const info = loras.find((x) => x.name === name);
    const box = qs('#chat-lora-info');
    if (!info) { box.textContent = ''; return; }
    const lines = [];
    if (info.base_model_display) lines.push(t('web.chat.trained_base') + ': ' + info.base_model_display);
    if (info.seq_len) lines.push('seq_len: ' + info.seq_len);
    if (info.datasets && info.datasets.length) lines.push(t('web.chat.trained_data') + ': ' + info.datasets.join(', '));
    if (info.prompt_mode) lines.push(t('web.chat.prompt_mode') + ': ' + info.prompt_mode);
    if (info.status) lines.push(t('web.common.status') + ': ' + info.status);
    if (!info.has_manifest) lines.push(t('web.chat.no_manifest'));
    box.textContent = lines.join('\n');
    if (info.base_model_display && models.some((m) => m.display_name === info.base_model_display)) {
      const sel = qs('#chat-base');
      const known = Array.from(sel.options).some((o) => o.value === sel.value);
      // 只在用户还没选（或原来的选中项已失效）时套用训练时用的基座，
      // 不要覆盖用户手动选的第 0 个模型
      if (!sel.value || !known) sel.value = info.base_model_display;
    }
  }

  async function loadLoras() {
    try {
      const res = await api('/api/inference/loras');
      loras = res.data || [];
      const sel = qs('#chat-lora');
      const cur = sel.value;
      sel.innerHTML = loras.map((x) =>
        '<option value="' + escapeHtml(x.name) + '">' + escapeHtml(x.name) +
        '</option>').join('');
      if (cur && loras.some((x) => x.name === cur)) sel.value = cur;
      renderLoraInfo();
    } catch (e) { toast(e.message, 'error'); }
  }

  async function loadStatus() {
    try {
      const res = await api('/api/inference/status');
      const d = res.data || {};
      if (d.loaded) showStatus(qs('#chat-load-status'), d.message || t('inf.loaded'), 'success');
      else showStatus(qs('#chat-load-status'), d.message || t('inf.not_loaded'), 'info');
    } catch (e) { toast(e.message, 'error'); }
  }

  async function loadModel() {
    const base = qs('#chat-base').value;
    const lora = qs('#chat-lora').value;
    if (!base || !lora) { toast(t('inf.need_both'), 'warn'); return; }
    const btn = qs('#chat-load');
    setButtonBusy(btn, true, t('web.chat.loading'));
    showStatus(qs('#chat-load-status'), t('web.chat.loading'), 'info');
    try {
      const res = await api('/api/inference/load', {
        method: 'POST', body: { base_model_name: base, lora_name: lora },
      });
      showStatus(qs('#chat-load-status'), res.data.message || t('inf.loaded'), 'success');
      toast(res.data.message || t('inf.loaded'), 'success');
    } catch (e) {
      showStatus(qs('#chat-load-status'), e.message, 'error');
    } finally {
      setButtonBusy(btn, false);
    }
  }

  async function unloadModel() {
    const btn = qs('#chat-unload');
    setButtonBusy(btn, true, t('web.loading'));
    try {
      const res = await api('/api/inference/unload', { method: 'POST' });
      showStatus(qs('#chat-load-status'), res.data.message || t('inf.unloaded'), 'info');
    } catch (e) { toast(e.message, 'error'); }
    finally { setButtonBusy(btn, false); }
  }

  function send(data) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      toast(t('web.chat.disconnected'), 'warn');
      connectWS();
      return false;
    }
    ws.send(JSON.stringify(data));
    return true;
  }

  function startChat() {
    if (streaming) { toast(t('inf.busy'), 'warn'); return; }
    const input = qs('#chat-input');
    const message = input.value.trim();
    if (!message) return;
    if (!connected) { toast(t('web.chat.disconnected'), 'warn'); connectWS(); return; }
    currentUser = message;
    currentText = '';
    addBubble('user', message);
    currentBubble = addBubble('assistant', '');
    currentBubble.classList.add('streaming');
    setStreaming(true);
    input.value = '';
    const ok = send({
      type: 'chat',
      message,
      history,
      system_prompt: qs('#chat-system').value,
      max_new_tokens: Number(qs('#chat-max-tokens').value || 256),
      temperature: Number(qs('#chat-temp').value || 0.7),
      top_p: Number(qs('#chat-top-p').value || 0.95),
      top_k: Number(qs('#chat-top-k').value || 40),
    });
    if (!ok) {
      currentBubble.classList.remove('streaming');
      setStreaming(false);
    }
  }

  function stopChat() {
    if (!streaming) return;
    send({ type: 'cancel' });
    showStatus(qs('#chat-load-status'), t('web.job_stopping'), 'warning');
  }

  function finishChat() {
    if (currentBubble) currentBubble.classList.remove('streaming');
    if (currentUser) {
      history.push({ role: 'user', content: currentUser });
      if (currentText.trim()) {
        history.push({ role: 'assistant', content: currentText });
      }
    }
    currentUser = '';
    currentText = '';
    currentBubble = null;
    setStreaming(false);
  }

  function handleMessage(msg) {
    const type = msg.type;
    if (type === 'ready') {
      setConn(true);
      if (msg.data) {
        const loaded = !!msg.data.loaded;
        if (loaded) {
          showStatus(qs('#chat-load-status'),
            t('inf.status_loaded', { base: msg.data.base, lora: msg.data.lora }),
            'success');
        }
      }
      return;
    }
    if (type === 'start') {
      return;
    }
    if (type === 'delta') {
      currentText += msg.text || '';
      renderBubble(currentBubble, currentText);
      return;
    }
    if (type === 'error') {
      if (currentBubble) {
        // 一个字都没生成（比如 inf.busy / 参数错误）就把空气泡收掉，
        // 免得界面上留一个空的“助手”气泡
        const wrap = currentBubble.parentElement;
        if (!currentText.trim() && wrap) {
          wrap.remove();
        } else {
          currentBubble.classList.remove('streaming');
          currentBubble.innerHTML = '<span class="text-error">' +
            escapeHtml(msg.message || t('web.error')) + '</span>';
        }
      }
      toast(msg.message || t('web.error'), 'error');
      setStreaming(false);
      currentBubble = null;
      currentText = '';
      currentUser = '';
      return;
    }
    if (type === 'done') {
      finishChat();
      if (msg.cancelled) toast(t('web.job_cancelled'), 'warning');
      return;
    }
    if (type === 'cancelling') {
      showStatus(qs('#chat-load-status'), t('web.job_stopping'), 'warning');
      return;
    }
  }

  function connectWS() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(proto + '//' + location.host + '/api/inference/ws');
    ws.onopen = () => {
      setConn(true);
      showStatus(qs('#chat-load-status'), t('web.chat.connected'), 'success');
    };
    ws.onmessage = (ev) => {
      try { handleMessage(JSON.parse(ev.data)); } catch (e) {}
    };
    ws.onclose = () => {
      setConn(false);
      setStreaming(false);
      reconnectTimer = setTimeout(connectWS, 2500);
    };
    ws.onerror = () => { try { ws.close(); } catch (e) {} };
  }

  qs('#chat-load').addEventListener('click', loadModel);
  qs('#chat-unload').addEventListener('click', unloadModel);
  qs('#chat-refresh').addEventListener('click', () => {
    loadModels(); loadLoras(); loadStatus();
  });
  qs('#chat-lora').addEventListener('change', renderLoraInfo);
  qs('#chat-send').addEventListener('click', startChat);
  qs('#chat-stop').addEventListener('click', stopChat);
  qs('#chat-clear').addEventListener('click', resetChat);
  qs('#chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      startChat();
    }
  });

  setConn(false);
  setStreaming(false);
  loadModels();
  loadLoras();
  loadStatus();
  connectWS();
})();
