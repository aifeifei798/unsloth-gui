(function () {
  const { api, t, toast, showStatus, hide, setButtonBusy, watchJob,
          escapeHtml, qs } = window.UG;

  let models = [];
  let datasets = [];
  let jobId = null;
  let watcher = null;
  let running = false;

  function modeChanged() {
    const epoch = qs('#train-mode').value === 'epochs';
    qs('#train-epochs-wrap').classList.toggle('hidden', !epoch);
    qs('#train-steps-wrap').classList.toggle('hidden', epoch);
  }

  function setRunning(on) {
    running = !!on;
    qs('#train-start').disabled = running;
    qs('#train-stop').disabled = !running;
    qs('#train-running-badge').classList.toggle('hidden', !running);
  }

  function appendLog(text) {
    const el = qs('#train-log');
    el.textContent += (el.textContent ? '\n' : '') + text;
    el.scrollTop = el.scrollHeight;
  }

  function selectedDatasets() {
    return Array.from(qs('#train-datasets')
      .querySelectorAll('input[type=checkbox]:checked')).map((x) => x.value);
  }

  function applyRecommendations() {
    const chosen = new Set(selectedDatasets());
    const merged = {};
    datasets.forEach((d) => {
      if (chosen.has(d.name) && d.recommended_params) {
        Object.assign(merged, d.recommended_params);
      }
    });
    if (!Object.keys(merged).length) {
      hide(qs('#train-reco'));
      return;
    }
    if (merged.training_mode) {
      const m = String(merged.training_mode).toLowerCase();
      qs('#train-mode').value = (m.includes('epoch') || m.includes('轮次'))
        ? 'epochs' : 'steps';
      modeChanged();
    }
    if (merged.num_epochs != null) qs('#train-epochs').value = merged.num_epochs;
    if (merged.lora_r != null) qs('#train-lora-r').value = merged.lora_r;
    if (merged.lora_alpha != null) qs('#train-lora-alpha').value = merged.lora_alpha;
    if (merged.learning_rate != null) qs('#train-lr').value = merged.learning_rate;
    const el = qs('#train-reco');
    el.textContent = t('pd.reco', { updates: JSON.stringify(merged) })
      .replace(/^\s+/, '');
    el.classList.remove('hidden');
  }

  async function loadModels() {
    try {
      const res = await api('/api/models');
      models = res.data || [];
      const sel = qs('#train-model');
      const cur = sel.value;
      sel.innerHTML = models.map((m) =>
        '<option value="' + escapeHtml(m.display_name) + '">' +
        escapeHtml(m.display_name) + ' · ' + escapeHtml(m.source_label) +
        '</option>').join('');
      if (cur && models.some((m) => m.display_name === cur)) sel.value = cur;
      if (sel.value) {
        const m = models.find((x) => x.display_name === sel.value);
        if (m && Number(qs('#train-seq').value) === 2048) {
          qs('#train-seq').value = m.max_seq_length;
        }
      }
    } catch (e) { toast(e.message, 'error'); }
  }

  async function loadDatasets() {
    try {
      const res = await api('/api/datasets?scope=trainable');
      datasets = res.data || [];
      const box = qs('#train-datasets');
      const old = new Set(selectedDatasets());
      if (!datasets.length) {
        box.innerHTML = '<span class="empty">' + escapeHtml(t('web.train.no_datasets')) + '</span>';
        const warn = qs('#train-warnings');
        warn.innerHTML = '<div class="alert alert-warning mb-2">' +
          escapeHtml(t('web.train.no_datasets')) +
          ' <a class="link" href="/prepare">' + escapeHtml(t('web.page.prepare')) + '</a></div>';
        hide(qs('#train-reco'));
        return;
      }
      qs('#train-warnings').innerHTML = (res.warnings || []).map((w) =>
        '<div class="alert alert-warning mb-2">' + escapeHtml(w) + '</div>').join('');
      box.innerHTML = datasets.map((d) =>
        '<label><input type="checkbox" class="checkbox checkbox-xs checkbox-primary" value="' +
        escapeHtml(d.name) + '"' + (old.has(d.name) ? ' checked' : '') + '> ' +
        escapeHtml(d.name) + '</label>').join('');
      box.querySelectorAll('input[type=checkbox]').forEach((c) =>
        c.addEventListener('change', applyRecommendations));
      applyRecommendations();
    } catch (e) {
      toast(e.message, 'error');
    }
  }

  function body() {
    return {
      experiment_name: qs('#train-exp').value.trim(),
      resume_training: qs('#train-resume').checked,
      truncate_dataset: qs('#train-truncate').checked,
      max_samples: Number(qs('#train-max-samples').value || 200),
      training_mode: qs('#train-mode').value,
      num_epochs: Number(qs('#train-epochs').value || 1),
      max_steps: Number(qs('#train-steps').value || 100),
      save_steps: Number(qs('#train-save-steps').value || 50),
      selected_model_name: qs('#train-model').value,
      selected_dataset_names: selectedDatasets(),
      lora_r: Number(qs('#train-lora-r').value || 8),
      lora_alpha: Number(qs('#train-lora-alpha').value || 16),
      batch_size: Number(qs('#train-batch').value || 1),
      grad_accum: Number(qs('#train-grad').value || 8),
      lr: Number(qs('#train-lr').value || 0.0002),
      max_seq_length: Number(qs('#train-seq').value || 2048),
      use_chat_template: qs('#train-chat-tpl').checked,
    };
  }

  function resetMetrics() {
    qs('#metric-step').textContent = '—';
    qs('#metric-loss').textContent = '—';
    qs('#metric-lr').textContent = '—';
    qs('#metric-epoch').textContent = '—';
  }

  function cleanup() {
    if (watcher) { try { watcher.close(); } catch (e) {} watcher = null; }
    setRunning(false);
  }

  function subscribe(id) {
    if (watcher) { try { watcher.close(); } catch (e) {} }
    watcher = watchJob(id, {
      state(data) {
        if (data.state === 'running' || data.state === 'queued') setRunning(true);
        if (data.state === 'done' || data.state === 'error' ||
            data.state === 'cancelled') setRunning(false);
      },
      progress(data) {
        qs('#train-progress').value =
          Math.max(0, Math.min(100, Number(data.frac || 0) * 100));
        qs('#train-progress-desc').textContent = data.desc || '';
      },
      log(data) { if (data.text) appendLog(data.text); },
      metrics(data) {
        if (data.step != null) qs('#metric-step').textContent = data.step;
        if (data.loss != null) qs('#metric-loss').textContent = Number(data.loss).toFixed(4);
        if (data.learning_rate != null) qs('#metric-lr').textContent = Number(data.learning_rate).toExponential(2);
        if (data.epoch != null) qs('#metric-epoch').textContent = Number(data.epoch).toFixed(2);
      },
      done(data) {
        const result = data.result || {};
        const ok = result.ok !== false;
        showStatus(qs('#train-status'), result.last || t('web.job_done'),
          ok ? 'success' : 'error');
        toast(ok ? t('web.job_done') : t('web.job_failed'), ok ? 'success' : 'error');
        qs('#train-progress').value = 100;
      },
      error(data) {
        showStatus(qs('#train-status'), (data.message || t('web.job_failed')) +
          (data.trace ? '\n' + data.trace : ''), 'error');
        toast(data.message || t('web.job_failed'), 'error');
      },
      cancelled(data) {
        showStatus(qs('#train-status'), data.message || t('web.job_cancelled'), 'warning');
      },
      end() { cleanup(); },
      connectionError() {
        showStatus(qs('#train-status'), t('web.sse_reconnect'), 'warning');
      },
    });
  }

  async function start() {
    const b = body();
    if (!b.experiment_name) { toast(t('tr.no_exp_name'), 'warn'); return; }
    if (!b.selected_dataset_names.length) { toast(t('tr.no_datasets'), 'warn'); return; }
    if (!b.selected_model_name) { toast(t('tr.no_model', { name: '?' }), 'warn'); return; }
    qs('#train-log').textContent = '';
    resetMetrics();
    qs('#train-progress').value = 0;
    hide(qs('#train-status'));
    const btn = qs('#train-start');
    setButtonBusy(btn, true, t('web.starting'));
    try {
      const res = await api('/api/train/start', { method: 'POST', body: b });
      jobId = res.data.job_id;
      setRunning(true);
      subscribe(jobId);
    } catch (e) {
      showStatus(qs('#train-status'), e.message, 'error');
      toast(e.message, 'error');
    } finally {
      setButtonBusy(btn, false);
    }
  }

  async function stop() {
    const btn = qs('#train-stop');
    setButtonBusy(btn, true, t('web.stopping'));
    try {
      const res = await api('/api/train/stop', { method: 'POST' });
      showStatus(qs('#train-status'), res.message || t('web.job_stopping'), 'warning');
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      setButtonBusy(btn, false);
    }
  }

  async function refreshStatus() {
    try {
      const res = await api('/api/train/status');
      const d = res.data || {};
      const job = d.job;
      if (job && (job.state === 'running' || job.state === 'queued')) {
        jobId = job.id;
        setRunning(true);
        subscribe(job.id);
      } else {
        setRunning(!!d.running);
      }
    } catch (e) {
      toast(e.message, 'error');
    }
  }

  async function tensorboard() {
    const btn = qs('#train-tb');
    setButtonBusy(btn, true, t('web.starting'));
    try {
      const res = await api('/api/train/tensorboard');
      const d = res.data || {};
      qs('#train-tb-status').textContent = d.message || '';
      if (d.started) {
        const url = location.protocol + '//' + location.hostname + ':' + d.port;
        const frame = qs('#train-tb-frame');
        frame.src = url;
        frame.classList.remove('hidden');
        window.open(url, '_blank', 'noopener');
      }
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      setButtonBusy(btn, false);
    }
  }

  qs('#train-mode').addEventListener('change', modeChanged);
  qs('#train-start').addEventListener('click', start);
  qs('#train-stop').addEventListener('click', stop);
  qs('#train-tb').addEventListener('click', tensorboard);
  qs('#train-refresh').addEventListener('click', () => {
    loadModels();
    loadDatasets();
    refreshStatus();
  });
  qs('#train-model').addEventListener('change', () => {
    const m = models.find((x) => x.display_name === qs('#train-model').value);
    if (m) qs('#train-seq').value = m.max_seq_length;
  });

  const now = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  qs('#train-exp').value = 'exp_' + now.getFullYear() + pad(now.getMonth() + 1) +
    pad(now.getDate()) + '_' + pad(now.getHours()) + pad(now.getMinutes());
  modeChanged();
  setRunning(false);
  loadModels();
  loadDatasets();
  refreshStatus();
})();
