(function () {
  const { api, t, toast, showStatus, hide, confirm, setButtonBusy,
          emptyRow, escapeHtml, qs } = window.UG;
  const tbody = qs('#datasets-tbody');
  let entries = [];
  let selected = null;

  function renderWarnings(list) {
    const box = qs('#datasets-warnings');
    if (!box) return;
    box.innerHTML = (list || []).map((w) =>
      '<div class="alert alert-warning mb-2">⚠️ ' + escapeHtml(w) + '</div>').join('');
  }

  function render() {
    if (!entries.length) {
      emptyRow(tbody, 6, t('web.empty'));
      return;
    }
    tbody.innerHTML = entries.map((e) => {
      const badge = e.trainable
        ? '<span class="badge badge-success badge-sm">✅</span>'
        : '<span class="badge badge-ghost badge-sm" title="' +
          escapeHtml(e.reason || '') + '">❌</span>';
      return '<tr class="models-row" data-name="' + escapeHtml(e.name) + '">'
        + '<td><strong>' + escapeHtml(e.name) + '</strong></td>'
        + '<td>' + escapeHtml(e.rows) + '</td>'
        + '<td>' + escapeHtml(e.source) + '</td>'
        + '<td class="small">' + escapeHtml(e.created || '—') + '</td>'
        + '<td>' + escapeHtml(e.size || '—') + '</td>'
        + '<td>' + badge + '</td>'
        + '</tr>';
    }).join('');
    Array.from(tbody.querySelectorAll('tr')).forEach((tr) => {
      tr.addEventListener('click', () => select(tr.dataset.name));
    });
    Array.from(tbody.querySelectorAll('tr')).forEach((tr) => {
      tr.classList.toggle('selected', tr.dataset.name === selected);
    });
  }

  async function load() {
    try {
      const res = await api('/api/datasets');
      entries = res.data || [];
      renderWarnings(res.warnings);
      if (selected && !entries.some((e) => e.name === selected)) {
        selected = null;
        qs('#dataset-detail').textContent = t('d.detail_empty');
        qs('#dataset-title').textContent = t('d.detail');
      }
      render();
      if (selected) await select(selected, true);
    } catch (e) {
      entries = [];
      emptyRow(tbody, 6, t('web.empty'));
      toast(e.message, 'error');
    }
  }

  async function select(name, quiet) {
    selected = name;
    Array.from(tbody.querySelectorAll('tr')).forEach((tr) => {
      tr.classList.toggle('selected', tr.dataset.name === name);
    });
    qs('#dataset-title').textContent = name;
    qs('#dataset-sample-out').classList.add('hidden');
    try {
      const res = await api('/api/datasets/detail?name=' + encodeURIComponent(name));
      qs('#dataset-detail').textContent = res.data.detail;
    } catch (e) {
      if (!quiet) toast(e.message, 'error');
      qs('#dataset-detail').textContent = e.message;
    }
  }

  async function sample() {
    if (!selected) { toast(t('d.err.noselect_sample'), 'warn'); return; }
    const btn = qs('#dataset-sample');
    setButtonBusy(btn, true, t('web.loading'));
    try {
      const res = await api('/api/datasets/sample?name=' + encodeURIComponent(selected));
      const out = qs('#dataset-sample-out');
      out.textContent = res.data.sample;
      out.classList.remove('hidden');
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      setButtonBusy(btn, false);
    }
  }

  async function rename() {
    if (!selected) { toast(t('d.err.noselect'), 'warn'); return; }
    const newName = qs('#dataset-new-name').value.trim();
    if (!newName) { toast(t('web.need_new_name'), 'warn'); return; }
    const btn = qs('#dataset-rename');
    setButtonBusy(btn, true, t('web.saving'));
    try {
      const res = await api('/api/datasets/rename', {
        method: 'POST', body: { old: selected, new: newName },
      });
      toast(res.message || t('d.renamed', { new: newName }), 'success');
      selected = res.data.new;
      qs('#dataset-new-name').value = '';
      await load();
    } catch (e) {
      showStatus(qs('#dataset-status'), e.message, 'error');
    } finally {
      setButtonBusy(btn, false);
    }
  }

  async function remove() {
    if (!selected) { toast(t('d.err.noselect'), 'warn'); return; }
    const ok = await confirm(t('web.confirm_delete', { name: selected }), t('d.del'));
    if (!ok) return;
    const btn = qs('#dataset-delete');
    setButtonBusy(btn, true, t('web.deleting'));
    try {
      const res = await api('/api/datasets?name=' + encodeURIComponent(selected),
                            { method: 'DELETE' });
      toast(res.message || t('dm.deleted', { name: selected, parts: '', kept: '' }), 'success');
      selected = null;
      qs('#dataset-detail').textContent = t('d.detail_empty');
      qs('#dataset-title').textContent = t('d.detail');
      qs('#dataset-sample-out').classList.add('hidden');
      await load();
    } catch (e) {
      showStatus(qs('#dataset-status'), e.message, 'error');
    } finally {
      setButtonBusy(btn, false);
    }
  }

  qs('#datasets-refresh').addEventListener('click', load);
  qs('#dataset-sample').addEventListener('click', sample);
  qs('#dataset-rename').addEventListener('click', rename);
  qs('#dataset-delete').addEventListener('click', remove);
  load();
})();
