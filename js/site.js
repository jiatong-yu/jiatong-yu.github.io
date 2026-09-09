/*
 * Renders the site from markdown files in /content at page load.
 *
 * A page opts in by setting attributes on <body>:
 *   data-page="home"                       -> renders content/home.md
 *   data-page="post" data-post="<slug>"    -> renders content/blog/<slug>.md
 *   data-base="../"                        -> path prefix back to the site root
 */

(function () {
  const body = document.body;
  const BASE = body.dataset.base || '';
  const PAGE = body.dataset.page;

  // A post page defaults to the markdown file named after it, so adding a post
  // means copying the template to blogs/<slug>.html and writing <slug>.md.
  const SLUG =
    body.dataset.post ||
    decodeURIComponent(location.pathname.split('/').pop() || '').replace(/\.html$/, '');

  /* ---------------------------------------------------------------- utils */

  // Every markdown file a page reads is recorded so live reload can watch it.
  const loaded = new Set();

  // Listeners bound to the rendered markup are dropped when live reload
  // replaces it, so a dev session does not pile up handlers on stale nodes.
  let renderScope = new AbortController();

  async function fetchText(url) {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
    loaded.add(url);
    return res.text();
  }

  // Values are single-line. Anything after the first colon is the value, so
  // URLs and "Title: subtitle" both work without quoting.
  function splitFrontMatter(source) {
    const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(source);
    if (!match) return { meta: {}, body: source };

    const meta = {};
    for (const line of match[1].split(/\r?\n/)) {
      const pair = /^([A-Za-z0-9_-]+)\s*:\s*(.*)$/.exec(line.trim());
      if (pair) meta[pair[1]] = pair[2].trim();
    }
    return { meta, body: source.slice(match[0].length) };
  }

  /* ----------------------------------------------------------------- math */

  // Marked would mangle TeX (underscores become emphasis, backslashes get
  // eaten), so every math span is swapped for an opaque token before parsing
  // and swapped back into the generated HTML afterwards.
  // Environments are matched before \[...\], because a row break such as
  // "\\[4pt]" inside an environment otherwise looks like a display-math opener
  // and swallows everything up to the next \].
  const MATH_PATTERNS = [
    /\$\$[\s\S]+?\$\$/g,
    /\\begin\{(equation|align|gather|multline)\*?\}[\s\S]*?\\end\{\1\*?\}/g,
    /\\begin\{(aligned|align|cases|array|matrix|pmatrix|bmatrix|split)\*?\}[\s\S]*?\\end\{\1\*?\}/g,
    /\\\[[\s\S]+?\\\]/g,
    /\\\([\s\S]+?\\\)/g,
    /\$(?![\s$])(?:[^$\\\n]|\\.|\n(?!\s*\n))+?\$/g,
  ];

  function protectMath(text, store) {
    for (const pattern of MATH_PATTERNS) {
      text = text.replace(pattern, (math) => {
        store.push(math);
        return `@@MATH${store.length - 1}@@`;
      });
    }
    return text;
  }

  // Math goes back into an HTML string, so a "<" in a subscript such as
  // "y_{<t}" would open a tag and swallow the rest of the formula before
  // MathJax ever sees it. Quotes are escaped too, since a formula can sit in
  // an attribute: the title of a figure becomes its caption.
  function escapeHtml(text) {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function restoreMath(html, store) {
    return html.replace(/@@MATH(\d+)@@/g, (_, i) => escapeHtml(store[Number(i)]));
  }

  /* ------------------------------------------------------------ directives */

  // ::: theorem Optional title
  // markdown body
  // :::
  const DIRECTIVES = {
    theorem: { box: 'theorem-box', title: 'theorem-title' },
    lemma: { box: 'lemma-box', title: 'lemma-title' },
    item: { box: 'item', title: 'item-title' },
    note: { box: 'post-note', title: 'post-note-title' },
  };

  // Directives nest, so this consumes lines recursively: everything between a
  // "::: kind" line and its matching ":::" is rendered as its own document and
  // wrapped in the box markup.
  function renderBlocks(lines, cursor, math) {
    let html = '';
    let plain = [];

    const flushPlain = () => {
      if (plain.length) html += marked.parse(protectMath(plain.join('\n'), math));
      plain = [];
    };

    while (cursor.i < lines.length) {
      const line = lines[cursor.i];

      if (/^:::\s*$/.test(line)) {
        cursor.i += 1;
        flushPlain();
        return html;
      }

      const start = /^:::\s*([a-z]+)\s*(.*)$/.exec(line);
      if (start && DIRECTIVES[start[1]]) {
        flushPlain();
        cursor.i += 1;

        const style = DIRECTIVES[start[1]];
        const title = start[2].trim();
        const inner = renderBlocks(lines, cursor, math);
        const heading = title
          ? `<div class="${style.title}">${renderInline(title)}</div>`
          : '';
        html += `<div class="${style.box}">${heading}${inner}</div>`;
        continue;
      }

      plain.push(line);
      cursor.i += 1;
    }

    flushPlain();
    return html;
  }

  /* --------------------------------------------------------------- render */

  function markdownToHtml(source) {
    const math = [];
    const html = renderBlocks(source.split(/\r?\n/), { i: 0 }, math);
    return restoreMath(html, math);
  }

  function renderInline(source) {
    const math = [];
    return restoreMath(marked.parseInline(protectMath(source || '', math)), math);
  }

  // A paragraph holding nothing but an image becomes a captioned figure; the
  // caption comes from the markdown image title: ![alt](src 'caption').
  function upgradeFigures(root) {
    root.querySelectorAll('p > img:only-child').forEach((img) => {
      const paragraph = img.parentElement;
      const figure = document.createElement('div');
      figure.className = 'figure-container';

      img.classList.add('figure-image');
      const caption = img.title;
      img.removeAttribute('title');
      figure.appendChild(img);

      if (caption) {
        const figcaption = document.createElement('figcaption');
        figcaption.className = 'figure-caption';
        figcaption.textContent = caption;
        figure.appendChild(figcaption);
      }
      paragraph.replaceWith(figure);
    });
  }

  /* -------------------------------------------------------------- contents */

  function slugify(text) {
    return (
      text
        .toLowerCase()
        .replace(/[^\w\s-]/g, '')
        .trim()
        .replace(/\s+/g, '-') || 'section'
    );
  }

  // "\tableofcontents" on a line of its own becomes a linked outline of the
  // post's "##" and "###" headings, numbered the way the sections read.
  // Headings are given ids here rather than by marked, so a link keeps
  // working when a heading is renamed only in case or punctuation.
  function renderContents(root) {
    const placeholder = [...root.querySelectorAll('p')].find(
      (p) => p.textContent.trim() === '\\tableofcontents'
    );
    if (!placeholder) return;

    const headings = [...root.querySelectorAll('h2, h3')];
    const used = new Set();
    const counts = [0, 0];
    const items = [];

    for (const heading of headings) {
      const level = heading.tagName === 'H2' ? 0 : 1;
      if (level === 0) counts[1] = 0;
      counts[level] += 1;
      if (level === 1 && counts[0] === 0) continue;

      let id = slugify(heading.textContent);
      for (let n = 2; used.has(id); n += 1) id = `${slugify(heading.textContent)}-${n}`;
      used.add(id);
      heading.id = id;

      const number = level === 0 ? `${counts[0]}` : `${counts[0]}.${counts[1]}`;
      items.push(
        `<li class="toc-item toc-level-${level + 2}">
           <a href="#${id}"><span class="toc-number">${number}</span>${escapeHtml(heading.textContent)}</a>
         </li>`
      );
    }

    if (!items.length) {
      placeholder.remove();
      return;
    }

    const nav = document.createElement('nav');
    nav.className = 'toc';
    nav.setAttribute('aria-label', 'Table of contents');
    nav.innerHTML = `<div class="toc-title">Contents</div><ul class="toc-list">${items.join('')}</ul>`;
    placeholder.replaceWith(nav);
  }

  // Awaited by the renderers: a re-render restores the scroll position once
  // the math has been laid out, and typeset changes the height of the page.
  function typesetMath(root) {
    if (!window.MathJax || !window.MathJax.typesetPromise) return Promise.resolve();
    if (window.MathJax.typesetClear) window.MathJax.typesetClear();
    return window.MathJax.typesetPromise([root]).catch(() => {});
  }

  /* ------------------------------------------------------------------ nav */

  function renderNav(meta, sections) {
    const links = sections
      .map(
        (s) =>
          `<li class="nav-item"><a class="nav-link js-scroll-trigger" href="#${s.id}">${s.title}</a></li>`
      )
      .join('');

    const social = [
      meta.email &&
        `<a class="social-icon" href="mailto:${meta.email}" aria-label="Email"><i class="fas fa-envelope"></i></a>`,
      meta.scholar &&
        `<a class="social-icon" href="${meta.scholar}" aria-label="Google Scholar"><i class="fab fa-google"></i></a>`,
      meta.github &&
        `<a class="social-icon" href="${meta.github}" aria-label="GitHub"><i class="fab fa-github"></i></a>`,
    ]
      .filter(Boolean)
      .join('');

    document.getElementById('topNav').innerHTML = `
      <div class="topnav-inner">
        <a class="navbar-brand js-scroll-trigger" href="${BASE}index.html#page-top">
          <span class="topnav-name">${meta.name}</span>
        </a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation"><span class="navbar-toggler-icon"></span></button>
        <div class="collapse navbar-collapse" id="navbarResponsive">
          <ul class="navbar-nav">${links}</ul>
          <div class="topnav-social">${social}</div>
        </div>
      </div>`;
  }

  /* ------------------------------------------------------------- homepage */

  // Top-level "## Heading" splits the file into page sections. An explicit id
  // can be given as "## Blogs {#blog}" to keep an existing anchor stable.
  function splitSections(source) {
    const sections = [];
    let current = null;

    for (const line of source.split(/\r?\n/)) {
      const heading = /^##\s+(.+?)\s*(?:\{#([\w-]+)\})?\s*$/.exec(line);
      if (heading) {
        current = {
          title: heading[1],
          id: heading[2] || heading[1].toLowerCase().replace(/[^\w]+/g, '-'),
          lines: [],
        };
        sections.push(current);
      } else if (current) {
        current.lines.push(line);
      }
    }
    return sections.map((s) => ({ ...s, text: s.lines.join('\n') }));
  }

  // Publications and teaching are lists of short records rather than prose, so
  // each entry is written as "### Title" followed by one line per field.
  function splitEntries(text) {
    const entries = [];
    let current = null;

    for (const line of text.split(/\r?\n/)) {
      const heading = /^###\s+(.+?)\s*$/.exec(line);
      if (heading) {
        current = { title: heading[1], fields: [] };
        entries.push(current);
      } else if (current && line.trim()) {
        current.fields.push(line.trim());
      }
    }
    return entries;
  }

  // Markdown links in entry metadata become the site's bracketed link buttons.
  function renderLinkButtons(source) {
    return renderInline(source).replace(
      /<a href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/g,
      (_, href, text) => `[<a class="link-btn" href="${href}">${text}</a>]`
    );
  }

  // The title carries the link to the paper, so an entry is written as
  // "### [Title](url)" followed by the authors and then the venue.
  function renderPublications(text) {
    const items = splitEntries(text).map(({ title, fields }) => {
      const [authors = '', venue = ''] = fields;

      return `
        <li>
          <div class="pub-head">
            <span class="pub-title">${renderInline(title)}</span>
            ${venue ? `<span class="pub-venue">${renderInline(venue)}</span>` : ''}
          </div>
          ${authors ? `<div class="pub-authors">${renderInline(authors)}</div>` : ''}
        </li>`;
    });

    return `<ol class="pub-list">${items.join('\n')}</ol>`;
  }

  // An author list is kept to a single line: the names that do not fit are
  // hidden behind an ellipsis that reveals the full list when clicked.
  function clampAuthorLists(root) {
    root.querySelectorAll('.pub-authors').forEach((element) => {
      const full = element.innerHTML.trim();
      // Author names never contain a comma, so the rendered markup splits
      // cleanly even when a name is wrapped in a link.
      const names = full.split(/,\s*/);
      if (names.length < 3) return;

      let expanded = false;

      const expand = () => {
        expanded = true;
        element.classList.remove('is-clamped');
        element.innerHTML = full;
      };

      const fit = () => {
        if (expanded) return;

        element.classList.add('is-clamped');
        element.innerHTML = full;
        if (element.scrollWidth <= element.clientWidth) {
          element.classList.remove('is-clamped');
          return;
        }

        // Longest run of names that still leaves room for the toggle.
        for (let shown = names.length - 1; ; shown -= 1) {
          element.innerHTML = `${names.slice(0, shown).join(', ')}, <button type="button" class="pub-authors-more" aria-label="Show all authors">&hellip;</button>`;
          if (element.scrollWidth <= element.clientWidth || shown === 1) break;
        }
        element.querySelector('.pub-authors-more').addEventListener('click', expand);
      };

      fit();
      window.addEventListener('resize', fit, { signal: renderScope.signal });
    });
  }

  function renderTeaching(text) {
    return splitEntries(text)
      .map(
        ({ title, fields }) => `
        <div class="item">
          <div class="item-title">${renderInline(title)}</div>
          <div class="item-meta">${renderLinkButtons(fields.join(' '))}</div>
        </div>`
      )
      .join('\n');
  }

  // The blog section lists post slugs; titles and summaries are read from each
  // post's front matter so they only ever live in one place.
  async function renderBlogSection(text) {
    const slugs = text
      .split(/\r?\n/)
      .map((line) => /^[-*]\s+(\S+)\s*$/.exec(line.trim()))
      .filter(Boolean)
      .map((match) => match[1]);

    const entries = await Promise.all(
      slugs.map(async (slug) => {
        try {
          const { meta } = splitFrontMatter(
            await fetchText(`${BASE}content/blog/${slug}.md`)
          );
          return `
            <article class="blog-entry">
              <h4 class="entry-title"><a href="${BASE}blogs/${slug}.html">${renderInline(meta.title || slug)}</a></h4>
              <div class="blog-tldr">${renderInline(meta.tldr || '')}</div>
            </article>`;
        } catch (err) {
          console.warn(`Skipping blog post "${slug}":`, err.message);
          return '';
        }
      })
    );
    return entries.join('\n');
  }

  async function renderHome() {
    const { meta, body: source } = splitFrontMatter(
      await fetchText(`${BASE}content/home.md`)
    );
    const sections = splitSections(source);

    document.title = meta.title || meta.name || document.title;
    renderNav(meta, sections);

    const renderers = {
      publications: renderPublications,
      teaching: renderTeaching,
      blog: renderBlogSection,
    };

    const html = await Promise.all(
      sections.map(async (section) => {
        const render = renderers[section.id] || markdownToHtml;
        const inner = await render(section.text);
        const header =
          section.id === 'about'
            ? `
              <div class="section-header about-header">
                <img class="about-portrait rounded-circle" src="${BASE}assets/img/profile_round.png" alt="${meta.name}" />
                <div class="about-heading">
                  <h3 class="section-title">${section.title}</h3>
                  <div class="section-rule"></div>
                </div>
              </div>`
            : `
              <div class="section-header">
                <h3 class="section-title">${section.title}</h3>
                <div class="section-rule"></div>
              </div>`;

        return `
          <section class="page-section" id="${section.id}">
            <div class="page-section-content">
              ${header}
              ${inner}
            </div>
          </section>`;
      })
    );

    const main = document.getElementById('content');
    main.innerHTML = `<br>${html.join('\n')}<br>`;
    clampAuthorLists(main);
    upgradeFigures(main);
    await typesetMath(main);
  }

  /* ------------------------------------------------------------ blog post */

  async function renderPost() {
    const slug = SLUG;
    const { meta, body: source } = splitFrontMatter(
      await fetchText(`${BASE}content/blog/${slug}.md`)
    );

    document.title = `${meta.title || slug} | ${meta.site_name || 'Jiatong Yu'}`;

    const home = splitFrontMatter(await fetchText(`${BASE}content/home.md`));
    renderNav(home.meta, []);

    const byline = [meta.author, meta.date].filter(Boolean).join(', ');
    const main = document.getElementById('content');
    main.innerHTML = `
      <br>
      <section class="page-section blog-entry" id="blog">
        <div class="page-section-content">
          <div class="section-header">
            <h1>${renderInline(meta.title || slug)}</h1>
            <div class="section-rule"></div>
            ${byline ? `<div class="author">by ${renderInline(byline)}</div>` : ''}
            <hr>
          </div>
          ${markdownToHtml(source)}
        </div>
      </section>`;

    renderContents(main);
    upgradeFigures(main);
    await typesetMath(main);
  }

  /* ---------------------------------------------------------- live reload */

  // Poll the markdown this page is built from and re-render when it changes,
  // so saving a file in the editor updates the browser. Local development only.
  function watchContent() {
    const local = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
    if (!local) return;

    // Fall back to the page's own sources when the first render failed, so a
    // fixed file still triggers a reload.
    if (!loaded.size) {
      loaded.add(`${BASE}content/home.md`);
      if (PAGE === 'post') loaded.add(`${BASE}content/blog/${SLUG}.md`);
    }

    const urls = [...loaded];
    const stamps = new Map();
    const signature = async (url) => {
      const res = await fetch(url, { method: 'HEAD', cache: 'no-store' });
      return `${res.headers.get('last-modified')}|${res.headers.get('content-length')}`;
    };

    let busy = false;

    setInterval(async () => {
      if (busy) return;

      for (const url of urls) {
        try {
          const now = await signature(url);
          const changed = stamps.has(url) && stamps.get(url) !== now;
          stamps.set(url, now);

          if (changed) {
            busy = true;
            await rerender();
            busy = false;
            return;
          }
        } catch (err) {
          /* server restarting or offline; try again next tick */
        }
      }
    }, 1000);
  }

  // Rebuilding the markup in place rather than reloading the page keeps the
  // reader where they were, which matters while writing a long post. The
  // scroll offset is reasserted after the render because replacing the content
  // briefly shortens the document and the browser clamps the position.
  async function rerender() {
    const { scrollX, scrollY } = window;

    try {
      await renderPage();
    } catch (err) {
      showError(err);
      return;
    }

    // The site scrolls smoothly, which would animate the restore into view
    // instead of the reader simply staying put.
    window.scrollTo({ left: scrollX, top: scrollY, behavior: 'instant' });
  }

  /* ----------------------------------------------------------------- boot */

  function showError(err) {
    console.error(err);
    document.getElementById('content').innerHTML = `
      <section class="page-section"><div class="page-section-content">
        <h3 class="section-title">Content failed to load</h3>
        <pre>${err.message}</pre>
        <p>Markdown is loaded over HTTP, so open the site through a local server
        (<code>python3 -m http.server 8000</code>) rather than as a file.</p>
      </div></section>`;
  }

  // Everything a render owns is rebuilt here, so live reload can call it again
  // against markup that has just been thrown away and replaced.
  async function renderPage() {
    renderScope.abort();
    renderScope = new AbortController();

    if (PAGE === 'home') await renderHome();
    else if (PAGE === 'post') await renderPost();

    if (document.getElementById('topNav')) {
      bootstrap.ScrollSpy.getInstance(document.body)?.dispose();
      new bootstrap.ScrollSpy(document.body, {
        target: '#topNav',
        rootMargin: '0px 0px -40%',
      });
    }

    const toggler = document.querySelector('.navbar-toggler');
    document.querySelectorAll('#navbarResponsive .nav-link').forEach((link) => {
      link.addEventListener('click', () => {
        if (toggler && window.getComputedStyle(toggler).display !== 'none') {
          toggler.click();
        }
      });
    });
  }

  window.addEventListener('DOMContentLoaded', async () => {
    marked.setOptions({ mangle: false, headerIds: false });

    try {
      await renderPage();
    } catch (err) {
      showError(err);
    }

    watchContent();
  });
})();
