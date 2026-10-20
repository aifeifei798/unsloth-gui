(function () {
  const I18N = window.I18N || {};

  function qs(sel, root) {
    return (root || document).querySelector(sel);
  }

  function qsa(sel, root) {
    return Array.from((root || document).querySelectorAll(sel));
  }

  function t(key, params) {
    let s = I18N[key] != null ? I18N[key] : key;
    if (params) {
      s = String(s).replace(/\{(\w+)\}/g, (m, k) =>
        params[k] != null ? params[k] : m);
    }
    return s;
  }

  async function api(path, opts) {
    const options = Object.assign({}, opts || {});
    if (options.body && !(options.body instanceof FormData) &&
        typeof options.body !== 'string') {
      options.headers = Object.assign(
        { 'Content-Type': 'application/json' }, options.headers || {});
      options.body = JSON.stringify(options.body);
    }
    const res = await fetch(path, options);
    const text = await res.text();
    let body = null;
    try { body = text ? JSON.parse(text) : null; } catch (e) {
      body = { ok: false, error: { message: text || ('HTTP ' + res.status) } };
    }
    if (!res.ok || (body && body.ok === false)) {
      const msg = (body && body.error && body.error.message)
        || (body && body.detail) || ('HTTP ' + res.status);
      const err = new Error(msg);
      err.status = res.status;
      err.body = body;
      throw err;
    }
    return body;
  }

  function toast(message, kind, timeout) {
    const box = document.getElementById('toasts');
    if (!box) return;
    const el = document.createElement('div');
    const k = kind || 'info';
    el.className = 'alert alert-' + (k === 'warn' ? 'warning' : k);
    el.textContent = message;
    box.appendChild(el);
    setTimeout(() => { try { el.remove(); } catch (e) {} }, timeout || 5200);
  }

  function showStatus(el, message, kind) {
    if (!el) return;
    el.className = 'alert alert-' + (kind || 'info');
    el.textContent = message;
    el.classList.remove('hidden');
  }

  function hide(el) { if (el) el.classList.add('hidden'); }

  function confirmDialog(message, title) {
    const dlg = document.getElementById('confirm-dialog');
    if (!dlg || typeof dlg.showModal !== 'function') {
      return Promise.resolve(window.confirm(message));
    }
    const titleEl = document.getElementById('confirm-title');
    const msgEl = document.getElementById('confirm-message');
    if (titleEl) titleEl.textContent = title || t('web.confirm');
    if (msgEl) msgEl.textContent = message || '';
    return new Promise((resolve) => {
      const yes = dlg.querySelector('[data-confirm="yes"]');
      const no = dlg.querySelector('[data-confirm="no"]');
      let settled = false;
      function finish(value) {
        if (settled) return;
        settled = true;
        yes.removeEventListener('click', onYes);
        no.removeEventListener('click', onNo);
        dlg.removeEventListener('cancel', onCancel);
        try { dlg.close(); } catch (e) {}
        resolve(value);
      }
      function onYes() { finish(true); }
      function onNo() { finish(false); }
      function onCancel(e) { e.preventDefault(); finish(false); }
      yes.addEventListener('click', onYes);
      no.addEventListener('click', onNo);
      dlg.addEventListener('cancel', onCancel);
      dlg.showModal();
    });
  }

  function setButtonBusy(btn, busy, busyText) {
    if (!btn) return;
    if (busy) {
      if (!btn.dataset.originalHtml) btn.dataset.originalHtml = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = '<span class="loading loading-spinner loading-xs"></span> '
        + (busyText || t('web.loading'));
    } else {
      btn.disabled = false;
      if (btn.dataset.originalHtml) {
        btn.innerHTML = btn.dataset.originalHtml;
        delete btn.dataset.originalHtml;
      }
    }
  }

  function watchJob(jobId, handlers) {
    const h = handlers || {};
    const es = new EventSource('/api/jobs/' + encodeURIComponent(jobId) + '/events');
    ['state', 'progress', 'log', 'metrics', 'done', 'error',
     'cancelled', 'end'].forEach((name) => {
      es.addEventListener(name, (ev) => {
        let data = {};
        try { data = ev.data ? JSON.parse(ev.data) : {}; } catch (e) {}
        if (typeof h[name] === 'function') h[name](data);
        if (name === 'end') es.close();
      });
    });
    es.onerror = function () {
      if (typeof h.connectionError === 'function') h.connectionError();
    };
    return es;
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function formatBytes(n) {
    const v = Number(n || 0);
    if (v < 1024) return v + ' B';
    if (v < 1024 * 1024) return (v / 1024).toFixed(1) + ' KB';
    if (v < 1024 * 1024 * 1024) return (v / 1024 / 1024).toFixed(1) + ' MB';
    return (v / 1024 / 1024 / 1024).toFixed(2) + ' GB';
  }

  function emptyRow(tbody, colspan, text) {
    if (!tbody) return;
    tbody.innerHTML = '<tr><td colspan="' + colspan +
      '" class="text-center muted py-4">' + escapeHtml(text) + '</td></tr>';
  }

  // 通用 tab 切换：按钮 data-tab="名字"，面板 data-panel="名字"
  function initTabs(barSel) {
    const bar = qs(barSel);
    if (!bar) return function () {};
    const buttons = qsa('[data-tab]', bar);
    const panels = qsa('[data-panel]');
    function select(name) {
      buttons.forEach((b) => b.classList.toggle('tab-active', b.dataset.tab === name));
      panels.forEach((p) => p.classList.toggle('hidden', p.dataset.panel !== name));
    }
    buttons.forEach((b) => b.addEventListener('click', () => select(b.dataset.tab)));
    const active = buttons.find((b) => b.classList.contains('tab-active')) || buttons[0];
    if (active) select(active.dataset.tab);
    return select;
  }

  function attachCommon() {
    const themeBtn = document.getElementById('theme-toggle');
    function applyTheme(theme) {
      document.documentElement.dataset.theme = theme;
      try { localStorage.setItem('ugui-theme', theme); } catch (e) {}
      if (themeBtn) themeBtn.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
    let currentTheme = 'light';
    try { currentTheme = localStorage.getItem('ugui-theme') || 'light'; } catch (e) {}
    applyTheme(currentTheme);
    if (themeBtn) {
      themeBtn.addEventListener('click', () => {
        applyTheme(document.documentElement.dataset.theme === 'dark'
          ? 'light' : 'dark');
      });
    }

    const lang = document.getElementById('lang-select');
    if (lang) {
      lang.addEventListener('change', async () => {
        try {
          await api('/api/lang/' + encodeURIComponent(lang.value));
          window.location.reload();
        } catch (e) { toast(e.message, 'error'); }
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attachCommon);
  } else {
    attachCommon();
  }

  window.UG = {
    t, api, toast, showStatus, hide, confirm: confirmDialog,
    setButtonBusy, watchJob, escapeHtml, formatBytes, emptyRow, initTabs,
    qs, qsa,
  };
})();
