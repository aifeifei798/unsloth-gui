(function () {
  const { api, t, toast, showStatus, hide, setButtonBusy, watchJob,
          formatBytes, escapeHtml, qs, qsa } = window.UG;

  let uploadId = null;
  let token = null;
  let columns = [];
  let jobId = null;
  let watcher = null;

  function selectTab(name) {
    qsa('#prep-tabs [data-tab]').forEach((btn) => {
      btn.classList.toggle('tab-active', btn.dataset.tab === name);
    });
    qs('#prep-panel-source').classList.toggle('hidden', name !== 'source');
    qs('#prep-panel-mapping').classList.toggle('hidden', name !== 'mapping');
  }

  function renderMulti(el, cols, selected) {
    if (!el) return;
    if (!cols || !cols.length) {
      el.innerHTML = '<span class="empty">' + escapeHtml(t('web.prepare.inspect_first')) + '</span>';
      return;
    }
    const chosen = new Set(selected || []);
    el.innerHTML = cols.map((c) =>
      '<label><input type="checkbox" class="checkbox checkbox-xs checkbox-primary" value="' +
      escapeHtml(c) + '"' + (chosen.has(c) ? ' checked' : '') + '> ' +
      escapeHtml(c) + '</label>').join('');
  }

  function getChecked(el) {
    if (!el) return [];
    return Array.from(el.querySelectorAll('input[type=checkbox]:checked'))
      .map((x) => x.value);
  }

  function renderColumns(cols) {
    columns = cols || [];
    renderMulti(qs('#prep-ins'), columns, columns.length ? [columns[0]] : []);
    renderMulti(qs('#prep-input'), columns, []);
    renderMulti(qs('#prep-think'), columns, []);
    renderMulti(qs('#prep-out'), columns, columns.length ? [columns[columns.length - 1]] : []);
  }

  function setMappingEnabled(on) {
    qs('#prep-preview').disabled = !on;
    qs('#prep-generate').disabled = !on;
  }

  function sourceChanged() {
    const kind = qs('#prep-source').value;
    qs('#prep-pane-upload').classList.toggle('hidden', kind !== 'upload');
    qs('#prep-pane-hf').classList.toggle('hidden', kind !== 'hf');
    qs('#prep-pane-existing').classList.toggle('hidden', kind !== 'existing');
    token = null;
    setMappingEnabled(false);
  }

  async function loadExisting() {
    try {
      const res = await api('/api/datasets?scope=all');
      const sel = qs('#prep-existing');
      const cur = sel.value;
      sel.innerHTML = (res.data || []).map((e) =>
        '<option value="' + escapeHtml(e.name) + '">' + escapeHtml(e.name) +
        (e.trainable ? '' : ' · ' + escapeHtml(t('web.datasets.not_trainable'))) +
        '</option>').join('');
      if (cur && Array.from(sel.options).some((o) => o.value === cur)) sel.value = cur;
    } catch (e) {
      toast(e.message, 'error');
    }
  }

  async function doUpload() {
    const input = qs('#prep-file');
    if (!input.files || !input.files.length) return;
    const fd = new FormData();
    fd.append('file', input.files[0]);
    qs('#prep-upload-status').textContent = t('web.loading');
    token = null;
    setMappingEnabled(false);
    try {
      const res = await api('/api/prepare/upload', { method: 'POST', body: fd });
      uploadId = res.data.upload_id;
      qs('#prep-upload-status').textContent =
        res.data.name + ' · ' + formatBytes(res.data.size);
      toast(t('web.prepare.uploaded'), 'success');
    } catch (e) {
      uploadId = null;
      qs('#prep-upload-status').textContent = '';
      toast(e.message, 'error');
    }
  }

  async function inspect() {
    const kind = qs('#prep-source').value;
    const body = { kind };
    if (kind === 'upload') {
      if (!uploadId) { toast(t('p.err.no_file'), 'warn'); return; }
      body.upload_id = uploadId;
    } else if (kind === 'hf') {
      body.hf_id = qs('#prep-hf-id').value.trim();
      body.split = qs('#prep-hf-split').value.trim() || 'train';
    } else {
      body.existing_name = qs('#prep-existing').value;
    }
    const btn = qs('#prep-inspect');
    setButtonBusy(btn, true, t('web.loading'));
    hide(qs('#prep-inspect-status'));
    try {
      const res = await api('/api/prepare/inspect', { method: 'POST', body });
      token = res.data.token;
      qs('#prep-inspect-out').textContent = res.data.text;
      renderColumns(res.data.columns);
      setMappingEnabled(true);
      showStatus(qs('#prep-inspect-status'),
        t('web.prepare.columns', { n: (res.data.columns || []).length }), 'success');
      if (!qs('#prep-name').value) {
        qs('#prep-name').value = 'dataset_' + new Date().toISOString().slice(0, 10).replace(/-/g, '');
      }
      selectTab('mapping');
      toast(t('web.prepare.to_mapping'), 'success');
    } catch (e) {
      token = null;
      setMappingEnabled(false);
      showStatus(qs('#prep-inspect-status'), e.message, 'error');
    } finally {
      setButtonBusy(btn, false);
    }
  }

  function mappingBody(includeName) {
    const body = {
      token,
      instruction_cols: getChecked(qs('#prep-ins')),
      input_cols: getChecked(qs('#prep-input')),
      think_cols: getChecked(qs('#prep-think')),
      output_cols: getChecked(qs('#prep-out')),
      fixed_instruction: qs('#prep-fixed').value,
    };
    if (includeName) body.output_name = qs('#prep-name').value.trim();
    return body;
  }

  async function preview() {
    if (!token) { toast(t('dp.err.no_state'), 'warn'); return; }
    const btn = qs('#prep-preview');
    setButtonBusy(btn, true, t('web.loading'));
    try {
      const res = await api('/api/prepare/preview', {
        method: 'POST', body: mappingBody(false),
      });
      qs('#prep-preview-out').textContent = res.data.preview;
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      setButtonBusy(btn, false);
    }
  }

  function appendLog(text, level) {
    const log = qs('#prep-log');
    log.textContent += (log.textContent ? '\n' : '') + text;
    log.scrollTop = log.scrollHeight;
  }

  function cleanupJob() {
    if (watcher) { try { watcher.close(); } catch (e) {} watcher = null; }
    jobId = null;
    qs('#prep-cancel').classList.add('hidden');
    qs('#prep-generate').disabled = !token;
  }

  async function generate() {
    if (!token) { toast(t('dp.err.no_state'), 'warn'); return; }
    const body = mappingBody(true);
    if (!body.output_name) { toast(t('web.need_output_name'), 'warn'); return; }
    if (!body.output_cols.length) { toast(t('dp.err.no_out'), 'warn'); return; }
    qs('#prep-progress').value = 0;
    qs('#prep-log').textContent = '';
    hide(qs('#prep-status'));
    try {
      const res = await api('/api/prepare/generate', { method: 'POST', body });
      jobId = res.data.job_id;
      qs('#prep-generate').disabled = true;
      qs('#prep-cancel').classList.remove('hidden');
      showStatus(qs('#prep-status'), t('web.job_running'), 'info');
      watcher = watchJob(jobId, {
        state(data) {
          if (data.state === 'running') showStatus(qs('#prep-status'), t('web.job_running'), 'info');
        },
        progress(data) {
          const pct = Math.max(0, Math.min(100, Number(data.frac || 0) * 100));
          qs('#prep-progress').value = pct;
          if (data.desc) showStatus(qs('#prep-status'), data.desc, 'info');
        },
        log(data) { if (data.text) appendLog(data.text); },
        done(data) {
          const result = data.result || {};
          qs('#prep-progress').value = 100;
          showStatus(qs('#prep-status'), result.status || t('web.job_done'), 'success');
          if (result.preview) qs('#prep-preview-out').textContent = result.preview;
          toast(result.status || t('web.job_done'), 'success');
          loadExisting();
        },
        error(data) {
          showStatus(qs('#prep-status'), (data.message || t('web.job_failed')) +
            (data.trace ? '\n' + data.trace : ''), 'error');
          toast(data.message || t('web.job_failed'), 'error');
        },
        cancelled(data) {
          showStatus(qs('#prep-status'), data.message || t('web.job_cancelled'), 'warning');
        },
        end() { cleanupJob(); },
        connectionError() {
          showStatus(qs('#prep-status'), t('web.sse_reconnect'), 'warning');
        },
      });
    } catch (e) {
      showStatus(qs('#prep-status'), e.message, 'error');
      cleanupJob();
    }
  }

  async function cancel() {
    if (!jobId) return;
    try {
      await api('/api/jobs/' + encodeURIComponent(jobId) + '/cancel', { method: 'POST' });
      showStatus(qs('#prep-status'), t('web.job_stopping'), 'warning');
    } catch (e) { toast(e.message, 'error'); }
  }

  qs('#prep-source').addEventListener('change', sourceChanged);
  qs('#prep-file').addEventListener('change', doUpload);
  qs('#prep-inspect').addEventListener('click', inspect);
  qs('#prep-preview').addEventListener('click', preview);
  qs('#prep-generate').addEventListener('click', generate);
  qs('#prep-cancel').addEventListener('click', cancel);
  qsa('#prep-tabs [data-tab]').forEach((btn) => {
    btn.addEventListener('click', () => selectTab(btn.dataset.tab));
  });
  sourceChanged();
  selectTab('source');
  loadExisting();
})();
