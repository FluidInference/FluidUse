(function () {
  if (window.__cua) { return; }
  let nextToken = 1;
  const tokenOf = (el) => {
    if (!el.dataset.cuaToken) { el.dataset.cuaToken = 'el-' + (nextToken++); }
    return el.dataset.cuaToken;
  };
  const byToken = (token) => document.querySelector('[data-cua-token="' + token + '"]');
  const clean = (text) => (text || '')
    .replace(/\s+/g, ' ')
    .replace(/\s*\*+\s*$/, '')
    .replace(/\s*\((required|optional)\)\s*$/i, (m) => /optional/i.test(m) ? ' (optional)' : '')
    .replace(/\s*required$/i, '')
    .replace(/\s*:\s*$/, '')
    .trim();
  const textOf = (node) => {
    const copy = node.cloneNode(true);
    copy.querySelectorAll('input, select, textarea, button, script, style').forEach((n) => n.remove());
    return clean(copy.innerText !== undefined ? copy.innerText : copy.textContent);
  };
  const visible = (el) => {
    if (el.disabled || el.readOnly) { return false; }
    const rects = el.getClientRects();
    if (!rects.length) { return false; }
    const style = getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none' || parseFloat(style.opacity) === 0) { return false; }
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };
  const labelFor = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria && clean(aria)) { return clean(aria); }
    const labelledBy = el.getAttribute('aria-labelledby');
    if (labelledBy) {
      const text = labelledBy.split(/\s+/).map((id) => document.getElementById(id)).filter(Boolean).map(textOf).join(' ');
      if (clean(text)) { return clean(text); }
    }
    if (el.id) {
      const label = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (label && textOf(label)) { return textOf(label); }
    }
    const parentLabel = el.closest('label');
    if (parentLabel && textOf(parentLabel)) { return textOf(parentLabel); }
    if (el.tagName === 'BUTTON' || el.getAttribute('role') === 'button' || el.tagName === 'A') {
      return clean(el.innerText || el.value || el.title || el.getAttribute('aria-label'));
    }
    if (el.tagName === 'INPUT' && (el.type === 'submit' || el.type === 'button')) {
      return clean(el.value || el.title);
    }
    let node = el;
    for (let depth = 0; depth < 4 && node; depth++) {
      let prev = node.previousElementSibling;
      while (prev) {
        if (!prev.matches('input, select, textarea, button, script, style')) {
          const text = textOf(prev);
          if (text && text.length < 120) { return text; }
        }
        prev = prev.previousElementSibling;
      }
      node = node.parentElement;
    }
    return clean(el.placeholder || el.name || el.title);
  };
  const roleFor = (el) => {
    const tag = el.tagName;
    if (tag === 'SELECT') { return 'ComboBox'; }
    if (tag === 'TEXTAREA' || el.isContentEditable) { return 'Edit'; }
    if (tag === 'BUTTON' || el.getAttribute('role') === 'button') { return 'Button'; }
    if (tag === 'A' && el.getAttribute('role') === 'button') { return 'Button'; }
    if (tag === 'INPUT') {
      const type = (el.type || 'text').toLowerCase();
      if (type === 'checkbox') { return 'CheckBox'; }
      if (type === 'file') { return 'FileUpload'; }
      if (type === 'submit' || type === 'button' || type === 'reset') { return 'Button'; }
      if (['hidden', 'radio', 'range', 'color', 'image'].includes(type)) { return null; }
      return 'Edit';
    }
    return null;
  };
  const valueOf = (el) => {
    if (el.tagName === 'SELECT') {
      const option = el.selectedOptions && el.selectedOptions[0];
      return option && option.value !== '' ? clean(option.text) : '';
    }
    if (el.isContentEditable) { return clean(el.innerText); }
    return el.value || '';
  };
  const setNativeValue = (el, value) => {
    const prototype = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const descriptor = Object.getOwnPropertyDescriptor(prototype, 'value');
    if (descriptor && descriptor.set) { descriptor.set.call(el, value); } else { el.value = value; }
  };
  window.__cua = {
    snapshot() {
      const selector = 'input, textarea, select, button, [role="button"], [contenteditable="true"]';
      const elements = [];
      document.querySelectorAll(selector).forEach((el) => {
        const role = roleFor(el);
        if (!role) { return; }
        let anchor = el;
        if (role === 'FileUpload' && !visible(el)) {
          anchor = el.closest('label') || (el.id && document.querySelector('label[for="' + CSS.escape(el.id) + '"]'));
          if (!anchor || !visible(anchor)) { return; }
        } else if (!visible(el)) { return; }
        const rect = anchor.getBoundingClientRect();
        elements.push({
          token: tokenOf(el),
          role,
          label: labelFor(el),
          value: role === 'FileUpload' ? Array.from(el.files || []).map((f) => f.name).join(', ') : valueOf(el),
          placeholder: clean(el.placeholder || ''),
          checked: role === 'CheckBox' ? !!el.checked : null,
          frame: [rect.x + window.scrollX, rect.y + window.scrollY, rect.width, rect.height],
        });
      });
      return JSON.stringify({ title: document.title, url: location.href, elements });
    },
    focus(token) {
      const el = byToken(token);
      if (!el) { return false; }
      el.scrollIntoView({ block: 'center', behavior: 'smooth' });
      el.focus();
      return true;
    },
    highlight(token, on) {
      let el = byToken(token);
      if (!el) { return false; }
      if (el.type === 'file' && !visible(el)) { el = el.closest('label') || el; }
      if (on) {
        el.dataset.cuaOutline = el.style.outline || '';
        el.style.outline = '3px solid #ff6a00';
        el.style.outlineOffset = '2px';
      } else {
        el.style.outline = el.dataset.cuaOutline || '';
        el.style.outlineOffset = '';
      }
      return true;
    },
    setValue(token, value, commit) {
      const el = byToken(token);
      if (!el) { return false; }
      if (el.tagName === 'SELECT') {
        const wanted = value.trim().toLowerCase();
        const option = Array.from(el.options).find((o) => o.text.trim().toLowerCase() === wanted || o.value.trim().toLowerCase() === wanted);
        if (!option) { return false; }
        el.value = option.value;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
      }
      if (el.isContentEditable) {
        el.innerText = value;
        el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: value }));
        return true;
      }
      setNativeValue(el, value);
      el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: value }));
      if (commit) {
        el.dispatchEvent(new Event('change', { bubbles: true }));
        el.dispatchEvent(new FocusEvent('blur', { bubbles: true }));
        return el.value === value;
      }
      return true;
    },
    click(token) {
      const el = byToken(token);
      if (!el) { return false; }
      el.scrollIntoView({ block: 'center', behavior: 'smooth' });
      el.click();
      return true;
    },
    attach(token, base64, name, mime) {
      const el = byToken(token);
      if (!el || el.type !== 'file') { return false; }
      const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
      const transfer = new DataTransfer();
      transfer.items.add(new File([bytes], name, { type: mime }));
      el.files = transfer.files;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
      return el.files.length === 1;
    },
    checked(token) {
      const el = byToken(token);
      return el ? !!el.checked : null;
    },
  };
})();
