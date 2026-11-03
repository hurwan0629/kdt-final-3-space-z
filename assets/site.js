(function () {
  var root = document.documentElement;

  function currentTheme() {
    var set = root.getAttribute('data-theme');
    if (set) return set;
    return window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  var lightbox = document.querySelector('.lightbox');
  function openLightbox(src, caption) {
    if (!lightbox) return;
    lightbox.querySelector('img').src = src;
    lightbox.querySelector('figcaption').textContent = caption || '';
    lightbox.hidden = false;
  }
  function closeLightbox() { if (lightbox) lightbox.hidden = true; }

  document.addEventListener('click', function (event) {
    var toggle = event.target.closest('[data-theme-toggle]');
    if (toggle) {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('szr-theme', next); } catch (e) {}
      return;
    }
    if (event.target.closest('[data-menu-toggle]')) {
      document.querySelector('.sidebar').classList.toggle('open');
      return;
    }
    var shot = event.target.closest('.shot-btn');
    if (shot) {
      var figure = shot.closest('figure');
      openLightbox(shot.getAttribute('data-full'), figure && figure.querySelector('figcaption') ? figure.querySelector('figcaption').textContent : '');
      return;
    }
    if (lightbox && !lightbox.hidden && (event.target === lightbox || event.target.closest('[data-close]'))) closeLightbox();
  });
  document.addEventListener('keydown', function (event) { if (event.key === 'Escape') closeLightbox(); });

  var pages = Array.prototype.slice.call(document.querySelectorAll('section.page'));
  if (!pages.length) return;
  var crumb = document.querySelector('[data-crumb]');
  var baseTitle = document.title;
  var scroller = document.querySelector('.flow-scroll');

  function show() {
    var id = decodeURIComponent(location.hash.slice(1));
    var page = pages.filter(function (p) { return p.getAttribute('data-page') === id; })[0];
    if (!page) { page = pages[0]; id = page.getAttribute('data-page'); }
    pages.forEach(function (p) { p.classList.toggle('active', p === page); });
    document.querySelectorAll('[data-target]').forEach(function (el) {
      el.classList.toggle('current', el.getAttribute('data-target') === id);
    });
    var title = page.getAttribute('data-title') || id;
    if (crumb) crumb.textContent = title;
    document.title = title + ' · ' + baseTitle;
    var sidebar = document.querySelector('.sidebar');
    if (sidebar) sidebar.classList.remove('open');
    window.scrollTo(0, 0);
    var node = scroller && scroller.querySelector('.fn[data-target="' + id + '"]');
    if (node) {
      var x = parseFloat(node.getAttribute('data-x')) || 0;
      scroller.scrollTo({ left: Math.max(0, x - scroller.clientWidth / 2 + 110), behavior: 'smooth' });
    }
    var item = sidebar && sidebar.querySelector('.current');
    if (item) {
      var top = item.offsetTop - sidebar.offsetTop;
      if (top < sidebar.scrollTop || top > sidebar.scrollTop + sidebar.clientHeight - 40) sidebar.scrollTop = top - 80;
    }
  }
  window.addEventListener('hashchange', show);
  show();
})();
