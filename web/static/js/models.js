(function () {
  const { api, t, toast, showStatus, hide, confirm, setButtonBusy, emptyRow,
          escapeHtml, qs } = window.UG;
  const tbody = qs('#models-tbody');
  let models = [];
  let editing = null;

  function fillForm(m) {
    editing = m ? m.display_name : null;
    qs('#model-form-title').textContent = m ? t('web.models.edit') : t('web.models.new');
    qs('#model-display').value = m ? m.display_name : '';
    qs('#model-source').value = m ? (m.source || 'auto') : 'auto';
    qs('#model-id').value = m ? m.model_id : '';
    qs('#model-4bit').checked = m ? !!m.load_in_4bit : true;
    qs('#model-fast').checked = m ? !!m.fast_download : false;
    qs('#model-dtype').value = m && m.dtype ? m.dtype : '';
    qs('#model-seq').value = m ? m.max_seq_length : 2048;
    qs('#model-chat').value = m ? (m.chat_template || '') : '';
    qs('#model-status').classList.add('hidden');
    Array.from(tbody.querySelectorAll('tr')).forEach((tr) => {
      tr.classList.toggle('selected', !!m && tr.dataset.name === m.display_name);
    });
  }

  function render() {
    if (!models.length) {
      emptyRow(tbody, 7, t('web.empty'));
      return;
    }
    tbody.innerHTML = models.map((m) => {
      const missing = m.missing_local
        ? ' <span class="badge badge-error badge-xs">' + escapeHtml(t('web.models.local_missing')) + '</span>'
        : '';
      return '<tr class="models-row" data-name="' + escapeHtml(m.display_name) + '">'
        + '<td><strong>' + escapeHtml(m.display_name) + '</strong></td>'
        + '<td>' + escapeHtml(m.source_label || m.source) + missing + '</td>'
        + '<td class="small">' + escapeHtml(m.model_id) + '</td>'
        + '<td>' + (m.load_in_4bit ? '✅' : '—') + '</td>'
        + '<td>' + (m.fast_download ? '✅' : '—') + '</td>'
        + '<td>' + escapeHtml(m.dtype || t('common.auto')) + '</td>'
        + '<td>' + escapeHtml(m.max_seq_length) + '</td>'
        + '</tr>';
    }).join('');
    Array.from(tbody.querySelectorAll('tr')).forEach((tr) => {
      tr.addEventListener('click', () => {
        const m = models.find((x) => x.display_name === tr.dataset.name);
        if (m) fillForm(m);
      });
    });
  }

  async function load() {
    try {
      const res = await api('/api/models');
      models = res.data || [];
      render();
      if (editing && !models.some((m) => m.display_name === editing)) {
        fillForm(null);
      }
    } catch (e) {
      models = [];
      emptyRow(tbody, 7, t('web.empty'));
      toast(e.message, 'error');
    }
  }

  async function save() {
    const btn = qs('#model-save');
    const body = {
      display_name: qs('#model-display').value.trim(),
      model_id: qs('#model-id').value.trim(),
      source: qs('#model-source').value,
      load_in_4bit: qs('#model-4bit').checked,
      fast_download: qs('#model-fast').checked,
      dtype: qs('#model-dtype').value || null,
      max_seq_length: Number(qs('#model-seq').value || 0),
      chat_template: qs('#model-chat').value.trim() || null,
    };
    setButtonBusy(btn, true, t('web.saving'));
    try {
      const res = editing
        ? await api('/api/models', { method: 'PUT',
            body: Object.assign({ original_name: editing }, body) })
        : await api('/api/models', { method: 'POST', body });
      showStatus(qs('#model-status'), res.message || t('web.saved'), 'success');
      editing = body.display_name;
      await load();
      const m = models.find((x) => x.display_name === editing);
      if (m) fillForm(m);
      toast(res.message || t('web.saved'), 'success');
    } catch (e) {
      showStatus(qs('#model-status'), e.message, 'error');
    } finally {
      setButtonBusy(btn, false);
    }
  }

  async function remove() {
    const name = editing || qs('#model-display').value.trim();
    if (!name) {
      toast(t('web.select_row'), 'warn');
      return;
    }
    const ok = await confirm(t('web.confirm_delete', { name }), t('m.del'));
    if (!ok) return;
    const btn = qs('#model-delete');
    setButtonBusy(btn, true, t('web.deleting'));
    try {
      const res = await api('/api/models?name=' + encodeURIComponent(name),
                            { method: 'DELETE' });
      toast(res.message || t('m.deleted', { name }), 'success');
      fillForm(null);
      await load();
    } catch (e) {
      showStatus(qs('#model-status'), e.message, 'error');
    } finally {
      setButtonBusy(btn, false);
    }
  }

  qs('#models-refresh').addEventListener('click', load);
  qs('#model-new').addEventListener('click', () => fillForm(null));
  qs('#model-save').addEventListener('click', save);
  qs('#model-delete').addEventListener('click', remove);
  load();
})();
