// Language switching: html[data-lang] toggles .lang-en / .lang-zh visibility
(function () {
  const LANG_KEY = 'me2605-lang';
  let lang = localStorage.getItem(LANG_KEY) || 'en';

  function apply() {
    document.documentElement.setAttribute('data-lang', lang);
    const btn = document.querySelector('.lang-toggle');
    if (btn) btn.textContent = lang === 'en' ? '中文' : 'English';
  }

  function toggle() {
    lang = lang === 'en' ? 'zh' : 'en';
    localStorage.setItem(LANG_KEY, lang);
    apply();
  }

  document.addEventListener('DOMContentLoaded', function () {
    apply();
    const btn = document.querySelector('.lang-toggle');
    if (btn) btn.addEventListener('click', toggle);
  });
})();
