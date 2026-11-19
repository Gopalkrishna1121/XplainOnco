/* XplainOnco site. No framework, no network access: KaTeX and the fonts are vendored, and every number shown in the
   demos comes from data/demo.js, which scripts/export_site_data.py generates from a SYNTHETIC cohort. */
(function () {
  "use strict";
  var D = window.XO_DEMO;
  var NS = "http://www.w3.org/2000/svg";
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function svg(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function html(tag, attrs, parent, text) {
    var e = document.createElement(tag);
    for (var k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function tex(el, src, display) {
    if (window.katex) window.katex.render(src, el, { throwOnError: false, displayMode: !!display });
    else el.textContent = src;
  }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function fmt(v, d) { return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d == null ? 2 : d); }

  /* ---------- repository link: works on github.io project pages, otherwise points at the "Run it" section ---------- */
  (function repoLinks() {
    var host = location.hostname, path = location.pathname.split("/").filter(Boolean);
    var url = null;
    if (/\.github\.io$/.test(host) && path.length) url = "https://github.com/" + host.split(".")[0] + "/" + path[0];
    $$("a[data-repo]").forEach(function (a) { if (url) { a.href = url; a.target = "_blank"; a.rel = "noopener"; } });
  })();

  /* ---------- wrap tables so they scroll sideways on narrow screens ---------- */
  $$("main table, article table, section table").forEach(function (t) {
    if (t.closest(".tabledata") || t.parentNode.classList.contains("tscroll")) return;
    var w = document.createElement("div"); w.className = "tscroll"; w.tabIndex = 0; w.setAttribute("role", "region"); w.setAttribute("aria-label", "Table, scrolls sideways");
    t.parentNode.insertBefore(w, t); w.appendChild(t);
  });

  /* ---------- rail scroll-spy ---------- */
  (function spy() {
    var links = $$("#railList a");
    var map = {};
    links.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var ids = Object.keys(map).filter(function (i) { return document.getElementById(i); });
    if (!("IntersectionObserver" in window)) return;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { links.forEach(function (l) { l.classList.remove("on"); }); map[en.target.id].classList.add("on"); }
      });
    }, { rootMargin: "-25% 0px -65% 0px" });
    ids.forEach(function (i) { io.observe(document.getElementById(i)); });
    if (map.top) map.top.classList.add("on");
  })();

  /* ---------- copy buttons ---------- */
  $$(".codeblock .copy").forEach(function (b) {
    b.addEventListener("click", function () {
      var t = b.parentNode.querySelector("pre").innerText;
      (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(function () {
        b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1400);
      }, function () { b.textContent = "Select and copy"; });
    });
  });

  /* =====================================================================================
     Architecture plate
     ===================================================================================== */
  var STAGES = {
    patient: {
      title: "Patient",
      body: "Every TCGA-BRCA patient contributes log-CPM expression, a flag for a non-silent somatic mutation and a GISTIC copy-number level from −2 to +2 for each retained gene, about 1,500 per fold, plus clinical covariates such as age. Patients missing any data type are removed. Normalisation, batch correction and gene selection all run inside each training fold, because selecting genes on the full dataset leaks label information.",
      eq: ["x_v\\in\\mathbb{R}^{3}\\ \\text{(expression, mutation, copy number)}", "X\\in\\mathbb{R}^{n\\times 3},\\qquad z\\in\\mathbb{R}^{3n+n'}"],
      where: "src/xplainonco/data/preprocess.py"
    },
    graph: {
      title: "Graph attention branch",
      body: "Genes are the nodes of one STRING graph shared by all patients; each patient supplies its own node features. A three-layer graph attention network mixes each gene with its neighbours. Attention coefficients depend on the patient. One learnable gate per STRING edge is shared across patients and records which edges carry subtype information at all; an L1 penalty keeps the gates sparse, and they give edge-level Integrated Gradients its path.",
      eq: ["h_v^{(l+1)}=\\mathrm{ELU}\\Big(\\sum_{u\\in N(v)\\cup\\{v\\}} m_{uv}\\,\\alpha^{(l)}_{vu}\\,W^{(l)}h_u\\Big)", "m_{uv}=\\sigma(\\omega_{uv}),\\quad m_{vv}=1"],
      where: "src/xplainonco/graph_branch.py"
    },
    trees: {
      title: "Node-conditioned soft decision trees",
      body: "A NODE ensemble of oblivious trees: every node at a given depth applies the same soft mixture of columns, chosen by 1.5-entmax, and compares it with a learned threshold. Splits are smooth, so thresholds and feature choices are learned by gradient descent. Trees are kept because on gene tables they still beat plain deep networks.",
      eq: ["f_{tk}(z)=\\sum_j a_{tkj}\\,z_j,\\qquad a_{tk}=\\mathrm{entmax}_{1.5}(\\xi_{tk})", "\\pi_{tk}(z)=\\sigma_{1.5}\\!\\Big(\\frac{f_{tk}(z)-\\tilde b_{tk}}{\\tilde\\tau_{tk}}\\Big)"],
      where: "src/xplainonco/tree_branch.py, src/xplainonco/entmax.py"
    },
    conditioning: {
      title: "Conditioning link",
      body: "This is the mechanism the paper proposes. The threshold and temperature of each split are shifted and scaled by the graph embeddings of exactly the genes that split reads, weighted by the same entmax coefficients that select them. The output layers start at zero, so training begins as an ordinary NODE ensemble. Try it in the next section.",
      eq: ["\\tilde b_{tk}=b_{tk}+\\sum_j a_{tkj}\\,\\gamma\\big(h_{g(j)}\\big)+\\gamma_0(h_G)", "\\tilde\\tau_{tk}=\\tau_{tk}\\exp\\Big(\\sum_j a_{tkj}\\,\\kappa\\big(h_{g(j)}\\big)\\Big)"],
      where: "ConditionedObliviousLayer in src/xplainonco/tree_branch.py"
    },
    fusion: {
      title: "Bidirectional cross-attention",
      body: "Gene tokens from the graph branch and tree tokens from the tree branch read each other with four attention heads in each direction. Both are mean-pooled, concatenated and passed to a linear softmax classifier. Conditioning changes how the trees partition patients; cross-attention decides how much each gene's context and each tree's decision contribute. Both attention maps are stored for the attribution layer.",
      eq: ["A^{GT}=\\mathrm{softmax}\\!\\Big(\\tfrac{HW_Q(LW_K)^{\\top}}{\\sqrt d}\\Big),\\qquad A^{TG}=\\mathrm{softmax}\\!\\Big(\\tfrac{LW_Q'(HW_K')^{\\top}}{\\sqrt d}\\Big)"],
      where: "src/xplainonco/fusion.py"
    },
    subtype: {
      title: "PAM50 subtype",
      body: "The classifier outputs a probability for each of the five subtypes. One loss trains everything: class-balanced cross-entropy (class weighting replaces SMOTE, because interpolated patients are not biologically plausible and would contaminate every attribution), an entropy term that stops cross-attention spreading evenly over all trees, and an L1 penalty on the edge gates.",
      eq: ["\\mathcal L=\\mathcal L_{CB}+\\lambda_1\\frac1n\\sum_v\\mathcal H\\big(A^{GT}_{v,:}\\big)+\\lambda_2\\sum_{(u,v)\\in E}m_{uv}"],
      where: "src/xplainonco/losses.py, src/xplainonco/train.py"
    },
    attribution: {
      title: "Cross-Modal Attribution Layer",
      body: "Runs after training and adds no trainable part. Joint Integrated Gradients scores every gene in one unit, the logit of the predicted class, and splits exactly into a graph-view part and a table-view part. Edge IG, gradient-weighted cross-attention and an HSIC Lasso surrogate add edge scores and non-additive effects. Prizes on genes and costs on edges then feed a degree-corrected prize-collecting Steiner tree.",
      eq: ["\\phi_i=(x_i-x_i')\\int_0^1\\frac{\\partial F_c\\big(x'+s(x-x')\\big)}{\\partial x_i}\\,ds,\\qquad \\phi_v=\\phi_v^{G}+\\phi_v^{T}", "p_v=\\max\\big(0,\\ (1-\\zeta)\\hat\\phi_v+\\zeta\\hat\\beta_v-\\mu\\hat d_v\\big)"],
      where: "src/xplainonco/attribution/"
    },
    subnetwork: {
      title: "Oncogenic Pathway Subnetwork",
      body: "The connected subgraph of STRING that maximises collected prize minus edge cost. Node size shows the prize, node colour shows the patient's expression z-score for that gene, a marker shows a mutation or copy-number change, and edge width shows relevance. The sign of each gene's score is kept as an annotation. The result is tested for over-representation against Hallmark and Reactome gene sets, with the retained genes as background.",
      eq: ["G_s=(V_s,E_s)=\\arg\\max\\ \\sum_{v\\in V_s}p_v-\\sum_{(u,v)\\in E_s}w_{uv}"],
      where: "src/xplainonco/attribution/subnetwork.py"
    }
  };
  (function arch() {
    var detail = $("#archDetail"), list = $("#archList");
    if (!detail) return;
    var order = ["patient", "graph", "trees", "conditioning", "fusion", "subtype", "attribution", "subnetwork"];
    order.forEach(function (k) {
      var b = html("button", { type: "button", class: "chip", "data-stage": k, "aria-pressed": "false" }, list, STAGES[k].title);
      b.addEventListener("click", function () { show(k); });
    });
    function show(k) {
      var s = STAGES[k];
      $$(".arch .stage").forEach(function (g) { g.classList.toggle("on", g.getAttribute("data-stage") === k); });
      $$("#archList .chip").forEach(function (c) { c.setAttribute("aria-pressed", c.getAttribute("data-stage") === k ? "true" : "false"); });
      detail.innerHTML = "";
      var left = html("div", {}, detail), right = html("div", {}, detail);
      html("h4", {}, left, s.title); html("p", {}, left, s.body);
      html("p", { class: "where" }, left, "In the code: " + s.where);
      s.eq.forEach(function (q) { var d = html("div", {}, right); tex(d, q, true); });
    }
    $$(".arch .stage").forEach(function (g) {
      var k = g.getAttribute("data-stage");
      g.addEventListener("click", function () { show(k); });
      g.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); show(k); } });
    });
    show("conditioning");
  })();

  /* =====================================================================================
     Node-conditioned split plate (Eqs. 3-5, illustrative gains)
     ===================================================================================== */
  function entmoid(x) {
    var ax = Math.abs(x), root = Math.sqrt(Math.max(8 - ax * ax, 0));
    var yneg = ax < 2 ? Math.pow(root - ax, 2) / 16 : 0;
    return x >= 0 ? 1 - yneg : yneg;
  }
  (function split() {
    var S = $("#splitSvg"); if (!S) return;
    var W = 560, H = 330, L = 52, R = 14, T = 16, B = 52, pw = W - L - R, ph = H - T - B;
    var XMIN = -3, XMAX = 3;
    var X = function (f) { return L + (f - XMIN) / (XMAX - XMIN) * pw; };
    var Y = function (p) { return T + (1 - p) * ph; };
    var colA = "#3e7cb1", colB = "#d4476b", colBase = "#8d8ba6";
    var els = {
      f: $("#sF"), ha: $("#sHA"), hb: $("#sHB"), g: $("#sG"), k: $("#sK"), off: $("#cOff"),
      of: $("#oF"), oha: $("#oHA"), ohb: $("#oHB"), og: $("#oG"), ok: $("#oK")
    };
    [-3, -2, -1, 0, 1, 2, 3].forEach(function (v) {
      svg("line", { x1: X(v), x2: X(v), y1: T, y2: T + ph, class: "grid" }, S);
      svg("text", { x: X(v), y: T + ph + 18, "text-anchor": "middle" }, S).textContent = v;
    });
    [0, 0.5, 1].forEach(function (p) {
      svg("line", { x1: L, x2: L + pw, y1: Y(p), y2: Y(p), class: "grid" }, S);
      svg("text", { x: L - 8, y: Y(p) + 4, "text-anchor": "end" }, S).textContent = p;
    });
    svg("line", { x1: L, x2: L + pw, y1: T + ph, y2: T + ph, class: "axis" }, S);
    svg("line", { x1: L, x2: L, y1: T, y2: T + ph, class: "axis" }, S);
    svg("text", { x: L + pw / 2, y: H - 8, "text-anchor": "middle" }, S).textContent = "weighted gene expression f";
    var yl = svg("text", { x: 14, y: T + ph / 2, "text-anchor": "middle", transform: "rotate(-90 14 " + (T + ph / 2) + ")" }, S);
    yl.textContent = "probability of going right";
    var base = svg("path", { fill: "none", stroke: colBase, "stroke-width": 2, "stroke-dasharray": "6 4" }, S);
    var pa = svg("path", { fill: "none", stroke: colA, "stroke-width": 3 }, S);
    var pb = svg("path", { fill: "none", stroke: colB, "stroke-width": 3 }, S);
    var vline = svg("line", { y1: T, y2: T + ph, stroke: "#1b1840", "stroke-width": 1.2, "stroke-dasharray": "1.5 3" }, S);
    var da = svg("circle", { r: 6, fill: colA, stroke: "#fff", "stroke-width": 2 }, S);
    var db = svg("circle", { r: 6, fill: colB, stroke: "#fff", "stroke-width": 2 }, S);
    var la = svg("text", { "font-weight": 700, fill: colA, style: "fill:" + colA }, S); la.textContent = "A";
    var lb = svg("text", { "font-weight": 700, style: "fill:" + colB }, S); lb.textContent = "B";
    var lg = $("#splitLegend");
    lg.innerHTML = '<span><svg width="30" height="10" class="sw"><line x1="0" x2="30" y1="5" y2="5" stroke="' + colBase + '" stroke-width="2.4" stroke-dasharray="6 4"/></svg>plain NODE split (constant threshold)</span>' +
      '<span><svg width="30" height="10" class="sw"><line x1="0" x2="30" y1="5" y2="5" stroke="' + colA + '" stroke-width="3"/></svg>patient A</span>' +
      '<span><svg width="30" height="10" class="sw"><line x1="0" x2="30" y1="5" y2="5" stroke="' + colB + '" stroke-width="3"/></svg>patient B</span>';
    var eq = $("#eqSplit");
    tex(eq, "\\tilde b = b + a\\,\\gamma(h),\\qquad \\tilde\\tau=\\tau\\,e^{\\,a\\,\\kappa(h)},\\qquad \\pi=\\sigma_{1.5}\\!\\left(\\frac{f-\\tilde b}{\\tilde\\tau}\\right)", true);
    var routes = $("#routes");

    function curve(bt, tau) {
      var d = "";
      for (var i = 0; i <= 240; i++) {
        var f = XMIN + (XMAX - XMIN) * i / 240;
        d += (i ? "L" : "M") + X(f).toFixed(1) + " " + Y(entmoid((f - bt) / tau)).toFixed(1);
      }
      return d;
    }
    function update() {
      var f = +els.f.value, ha = +els.ha.value, hb = +els.hb.value, g = +els.g.value, k = +els.k.value, off = els.off.checked;
      els.of.textContent = f.toFixed(2); els.oha.textContent = fmt(ha, 2); els.ohb.textContent = fmt(hb, 2);
      els.og.textContent = g.toFixed(2); els.ok.textContent = fmt(k, 2);
      [els.g, els.k].forEach(function (e) { e.disabled = off; });
      var P = function (h) { return off ? { b: 0, t: 1 } : { b: g * h, t: Math.exp(k * h) }; };
      var A = P(ha), Bp = P(hb);
      base.setAttribute("d", curve(0, 1));
      pa.setAttribute("d", curve(A.b, A.t)); pb.setAttribute("d", curve(Bp.b, Bp.t));
      var pA = entmoid((f - A.b) / A.t), pB = entmoid((f - Bp.b) / Bp.t);
      vline.setAttribute("x1", X(f)); vline.setAttribute("x2", X(f));
      da.setAttribute("cx", X(f)); da.setAttribute("cy", Y(pA)); db.setAttribute("cx", X(f)); db.setAttribute("cy", Y(pB));
      la.setAttribute("x", X(f) + 10); la.setAttribute("y", Y(pA) + (pA < pB ? 16 : -6));
      lb.setAttribute("x", X(f) + 10); lb.setAttribute("y", Y(pB) + (pB < pA ? 16 : -6));
      routes.innerHTML = "";
      [["Patient A", A, pA, colA, ha], ["Patient B", Bp, pB, colB, hb]].forEach(function (r) {
        var d = html("div", { class: "route", style: "--col:" + r[3] }, routes);
        var who = html("div", { class: "who" }, d); html("span", {}, who, r[0]); html("span", {}, who, "right " + Math.round(r[2] * 100) + "%");
        var bar = html("div", { class: "bar", role: "img", "aria-label": r[0] + " goes right with probability " + Math.round(r[2] * 100) + " percent" }, d);
        html("i", { class: "l", style: "width:" + (100 * (1 - r[2])) + "%" }, bar); html("i", { class: "r", style: "width:" + (100 * r[2]) + "%" }, bar);
        html("small", {}, d, "threshold " + fmt(r[1].b, 2) + ", temperature " + r[1].t.toFixed(2));
      });
      if (Math.abs(pA - pB) > 0.12) {
        html("p", { class: "caption", style: "margin:.2rem 0 0" }, routes, "Same weighted expression, different routing: " + (pA > pB ? "A" : "B") + " is sent right far more often.");
      }
    }
    $$("input", S.closest(".plate")).forEach(function (i) { i.addEventListener("input", update); });
    update();
  })();

  if (!D) {
    $$("#heroViz, #viewSvg, .charts").forEach(function (n) { if (n) n.setAttribute("hidden", ""); });
    return;
  }

  /* =====================================================================================
     Network view (hero and viewer share it)
     ===================================================================================== */
  var N = D.genes.length, W = 640, H = 600, PAD = 36;
  var EXPR_LOW = [62, 124, 177], EXPR_HIGH = [212, 71, 107], NEUTRAL = [244, 243, 248];
  function exprColour(z) {
    var t = clamp(z / 2.2, -1, 1), a = NEUTRAL, b = t < 0 ? EXPR_LOW : EXPR_HIGH, u = Math.abs(t);
    return "rgb(" + [0, 1, 2].map(function (i) { return Math.round(a[i] + (b[i] - a[i]) * u); }).join(",") + ")";
  }
  var edgeKey = {};
  D.edges.forEach(function (e, i) { edgeKey[e[0] + "," + e[1]] = i; });
  var zKey = function (z, m) { var f = function (v) { return Number.isInteger(v) ? v.toFixed(1) : String(v); }; return f(z) + "|" + f(m); };

  function prizes(st, zeta, mu) {
    var sa = 0, sb = 0, maxd = 1, i;
    for (i = 0; i < N; i++) { sa += st.phi_abs[i]; sb += st.beta[i]; if (D.genes[i].deg > maxd) maxd = D.genes[i].deg; }
    var z = sb > 0 ? zeta : 0, base = [], bmax = 0;
    for (i = 0; i < N; i++) { var v = (1 - z) * st.phi_abs[i] / Math.max(sa, 1e-12) + z * (sb > 0 ? st.beta[i] / sb : 0); base.push(v); if (v > bmax) bmax = v; }
    var p = base.map(function (v, j) { return Math.max(0, v - mu * bmax * D.genes[j].deg / maxd); });
    var pm = Math.max.apply(null, p.concat([1e-12]));
    return p.map(function (v) { return v / pm; });
  }

  function NetworkView(svgEl, opts) {
    this.svg = svgEl; this.opts = opts || {};
    this.state = { s: 0, mode: "subtype", zi: 1, mi: 2, truth: false, sel: -1 };
    var self = this;
    var gE = svg("g", {}, svgEl), gS = svg("g", {}, svgEl), gH = svg("g", {}, svgEl), gN = svg("g", {}, svgEl), gR = svg("g", {}, svgEl), gM = svg("g", {}, svgEl), gL = svg("g", {}, svgEl);
    this.px = D.genes.map(function (g) { return PAD + g.x * (W - 2 * PAD); });
    this.py = D.genes.map(function (g) { return PAD + g.y * (H - 2 * PAD); });
    D.edges.forEach(function (e) {
      svg("line", { x1: self.px[e[0]], y1: self.py[e[0]], x2: self.px[e[1]], y2: self.py[e[1]], class: "bg-edge" }, gE);
    });
    this.subEdges = D.edges.map(function (e) {
      return svg("line", { x1: self.px[e[0]], y1: self.py[e[0]], x2: self.px[e[1]], y2: self.py[e[1]], class: "sub-edge", pathLength: 1 }, gS);
    });
    this.hubs = D.genes.map(function (g, i) { return g.hub ? svg("circle", { cx: self.px[i], cy: self.py[i], r: 11, class: "hubmark" }, gH) : null; });
    this.nodes = D.genes.map(function (g, i) {
      var c = svg("circle", { cx: self.px[i], cy: self.py[i], r: 2.4, class: "node bg", tabindex: opts && opts.interactive ? 0 : -1 }, gN);
      var t = svg("title", {}, c); t.textContent = g.id;
      if (opts && opts.interactive) {
        c.addEventListener("click", function () { self.select(i); });
        c.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); self.select(i); } });
      }
      return c;
    });
    this.rings = D.genes.map(function (g, i) { return svg("circle", { cx: self.px[i], cy: self.py[i], r: 9, class: "ring" }, gR); });
    this.muts = D.genes.map(function (g, i) { return svg("rect", { width: 5.5, height: 5.5, class: "mut" }, gM); });
    this.labels = D.genes.map(function (g, i) { var t = svg("text", { x: self.px[i], y: self.py[i] - 12, "text-anchor": "middle" }, gL); t.textContent = g.id; return t; });
    this.onselect = null;
  }
  NetworkView.prototype.select = function (i) { this.state.sel = this.state.sel === i ? -1 : i; this.render(); if (this.onselect) this.onselect(this.state.sel); };
  NetworkView.prototype.current = function () {
    var st = D.subtypes[this.state.s], S = this.state, r = {};
    r.st = st;
    if (S.mode === "patient") {
      var p = st.patient;
      r.nodes = p.nodes; r.edges = p.edges; r.conn = p.connector; r.prize = p.prizes; r.expr = p.expr_z;
      r.mut = p.mut.map(function (v) { return v > 0; });
      r.stats = null;
    } else {
      var zeta = D.meta.zetas[S.zi], mu = D.meta.mus[S.mi];
      var g = st.grid[zKey(zeta, mu)];
      r.nodes = g.nodes; r.edges = g.edges; r.conn = g.connector; r.prize = prizes(st, zeta, mu); r.expr = st.expr_z;
      r.mut = st.mut_freq.map(function (v) { return v >= 0.25; });
      r.stats = g; r.zeta = zeta; r.mu = mu;
    }
    return r;
  };
  NetworkView.prototype.render = function () {
    var S = this.state, c = this.current(), self = this;
    var inSub = {}, connSet = {};
    c.nodes.forEach(function (n, k) { inSub[n] = true; if (c.conn[k]) connSet[n] = true; });
    var edgeOn = {};
    c.edges.forEach(function (e) { var k = e[0] < e[1] ? e[0] + "," + e[1] : e[1] + "," + e[0]; edgeOn[edgeKey[k]] = true; });
    var rr = c.st.edge_r;
    this.subEdges.forEach(function (l, i) {
      var on = !!edgeOn[i];
      l.classList.toggle("in", on);
      l.setAttribute("stroke-width", on ? (1.3 + 3.6 * (rr[i] || 0)).toFixed(2) : 1);
    });
    var ranked = c.nodes.filter(function (n) { return !connSet[n]; }).sort(function (a, b) { return c.prize[b] - c.prize[a]; });
    var topSet = {}, placed = [];
    if (this.opts.labels !== false) {  // greedy label placement: skip a label that would sit on one already placed
      for (var q = 0; q < ranked.length && placed.length < 9; q++) {
        var n0 = ranked[q], lx = this.px[n0], ly = this.py[n0];
        if (placed.every(function (o) { return Math.abs(o[0] - lx) > 34 || Math.abs(o[1] - ly) > 16; })) { placed.push([lx, ly]); topSet[n0] = true; }
      }
    }
    var truthMod = S.truth;
    this.nodes.forEach(function (el, i) {
      var inn = !!inSub[i], conn = !!connSet[i];
      el.setAttribute("class", "node " + (inn ? (conn ? "conn" : "in") : "bg") + (S.sel === i ? " sel" : ""));
      if (inn) {
        el.setAttribute("r", conn ? 5 : (5 + 15 * Math.sqrt(clamp(c.prize[i], 0, 1))).toFixed(1));
        el.style.fill = conn ? "#fff" : exprColour(c.expr[i]);
      } else { el.setAttribute("r", S.sel === i ? 5 : 2.4); el.style.fill = ""; }
      var rad = parseFloat(el.getAttribute("r"));
      var ring = self.rings[i];
      ring.setAttribute("r", rad + 4);
      ring.classList.toggle("on", truthMod && D.genes[i].module === S.s);
      if (self.hubs[i]) { self.hubs[i].setAttribute("r", rad + 5); self.hubs[i].classList.toggle("on", truthMod); }
      var m = self.muts[i], showm = inn && !conn && c.mut[i];
      m.setAttribute("x", self.px[i] + rad * 0.62); m.setAttribute("y", self.py[i] - rad * 0.62 - 4.5); m.classList.toggle("on", !!showm);
      var lab = self.labels[i];
      lab.setAttribute("y", self.py[i] - rad - 5);
      lab.classList.toggle("on", !!(topSet[i] || S.sel === i));
      el.firstChild.textContent = D.genes[i].id + (inn ? (conn ? " (connector)" : ", prize " + c.prize[i].toFixed(2)) : "");
    });
    this.last = c;
  };
  NetworkView.prototype.set = function (patch) { for (var k in patch) this.state[k] = patch[k]; this.render(); };

  /* ---------- hero ---------- */
  var hero;
  (function () {
    var s = $("#heroSvg"); if (!s) return;
    hero = new NetworkView(s, { labels: true });
    var chips = $("#heroChips"), host = $("#heroViz");
    var startSub = 0;
    D.subtypes.forEach(function (st, i) {
      var b = html("button", { type: "button", class: "chip", "aria-pressed": i === startSub ? "true" : "false" }, chips, st.name);
      b.addEventListener("click", function () {
        $$(".chip", chips).forEach(function (c, j) { c.setAttribute("aria-pressed", j === i ? "true" : "false"); });
        host.classList.remove("draw");
        hero.set({ s: i });
        if (viewer) viewer.set({ s: i });
      });
    });
    hero.set({ s: startSub });
    if (reduce) host.classList.remove("draw"); else setTimeout(function () { host.classList.remove("draw"); }, 2200);
  })();

  /* ---------- viewer ---------- */
  var viewer;
  (function () {
    var s = $("#viewSvg"); if (!s) return;
    viewer = new NetworkView(s, { interactive: true });
    var panel = $("#vPanel"), segS = $("#vSubtype");
    var zEl = $("#vZeta"), mEl = $("#vMu"), oz = $("#oZeta"), om = $("#oMu");
    D.subtypes.forEach(function (st, i) {
      var b = html("button", { type: "button", "aria-pressed": i === 0 ? "true" : "false" }, segS, st.name);
      b.addEventListener("click", function () {
        $$("button", segS).forEach(function (x, j) { x.setAttribute("aria-pressed", j === i ? "true" : "false"); });
        viewer.state.sel = -1; viewer.set({ s: i }); refresh();
        if (hero) { $$("#heroChips .chip").forEach(function (c, j) { c.setAttribute("aria-pressed", j === i ? "true" : "false"); }); hero.set({ s: i }); }
      });
    });
    $$("#vMode button").forEach(function (b) {
      b.addEventListener("click", function () {
        $$("#vMode button").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        viewer.set({ mode: b.getAttribute("data-m") }); refresh();
      });
    });
    function sliders() {
      var zi = +zEl.value, mi = +mEl.value;
      oz.textContent = D.meta.zetas[zi].toFixed(2); om.textContent = D.meta.mus[mi].toFixed(1);
      var patient = viewer.state.mode === "patient";
      zEl.disabled = mEl.disabled = patient;
      viewer.state.zi = zi; viewer.state.mi = mi;
    }
    [zEl, mEl].forEach(function (e) { e.addEventListener("input", function () { sliders(); viewer.render(); refresh(); }); });
    $("#vTruth").addEventListener("change", function (e) {
      viewer.set({ truth: e.target.checked });
      $("#truthKey").hidden = !e.target.checked; refresh();
    });
    viewer.onselect = function () { refresh(); };

    function row(dl, k, v) { html("dt", {}, dl, k); html("dd", {}, dl, v); }
    function refresh() {
      sliders(); viewer.render();
      var c = viewer.last, st = c.st, S = viewer.state;
      panel.innerHTML = "";
      var n = c.nodes.length, conn = c.conn.reduce(function (a, b) { return a + Number(b); }, 0);
      var sum = html("p", { class: "sum" }, panel);
      var parts = "<strong>" + n + " genes</strong>, " + conn + (conn === 1 ? " connector." : " connectors.");
      if (S.truth) {
        var inMod = c.nodes.filter(function (g) { return D.genes[g].module === S.s; }).length;
        var hubs = c.nodes.filter(function (g) { return D.genes[g].hub; }).length;
        var lab = c.nodes.filter(function (g) { return D.genes[g].label; }).length;
        parts += " <strong>" + inMod + " of " + st.planted_module.length + "</strong> planted driver genes, <strong>" + hubs + "</strong> hub" + (hubs === 1 ? "" : "s") + ", " + lab + " label-defining.";
      }
      sum.innerHTML = parts;
      if (S.sel >= 0) {
        var i = S.sel, g = D.genes[i], pz = c.prize[i];
        html("h4", {}, panel, g.id);
        var dl = html("dl", {}, panel);
        row(dl, "Interaction partners", g.deg);
        row(dl, "Prize", pz.toFixed(2));
        var pat = S.mode === "patient";
        row(dl, pat ? "Expression z-score" : "Mean expression z-score", fmt(c.expr[i], 2));
        var gG = pat ? st.patient.phi_G[i] : st.phi_G[i], gT = pat ? st.patient.phi_T[i] : st.phi_T[i];
        row(dl, "Graph-view IG", fmt(gG, 3)); row(dl, "Table-view IG", fmt(gT, 3));
        var tot = Math.abs(gG) + Math.abs(gT) || 1;
        var bar = html("div", { class: "split-bar", role: "img", "aria-label": "Share of attribution from the graph view and the table view" }, panel);
        html("i", { style: "width:" + (100 * Math.abs(gG) / tot) + "%;background:#6b5ba8" }, bar); html("i", { style: "width:" + (100 * Math.abs(gT) / tot) + "%;background:#c7762b" }, bar);
        html("small", { class: "caption" }, panel, "Purple: graph view. Amber: table view. Both add up to one score in logit units.");
        if (!pat) row(html("dl", { style: "margin-top:.5rem" }, panel), "Surrogate weight β", st.beta[i].toFixed(4));
        if (S.truth) {
          var tags = [];
          if (g.module === S.s) tags.push("planted driver of " + st.name);
          else if (g.module >= 0) tags.push("planted driver of " + D.subtypes[g.module].name);
          if (g.hub) tags.push("hub"); if (g.label) tags.push("label-defining gene");
          html("p", { class: "caption", style: "margin:.6rem 0 0" }, panel, "Truth: " + (tags.length ? tags.join(", ") : "no planted role") + ".");
        }
        var back = html("button", { type: "button", class: "btn ghost small", style: "margin-top:.7rem" }, panel, "Back to top genes");
        back.addEventListener("click", function () { viewer.set({ sel: -1 }); refresh(); });
      } else {
        html("h4", {}, panel, "Top genes by prize");
        var ul = html("ul", { class: "topn" }, panel);
        c.nodes.map(function (g, k) { return { g: g, k: k }; }).filter(function (o) { return !c.conn[o.k]; })
          .sort(function (a, b) { return c.prize[b.g] - c.prize[a.g]; }).slice(0, 8).forEach(function (o) {
            var li = html("li", {}, ul); html("b", { style: "font-weight:600" }, li, D.genes[o.g].id);
            var bb = html("span", { class: "bar" }, li); html("i", { style: "width:" + (100 * c.prize[o.g]) + "%" }, bb);
            html("small", {}, li, c.prize[o.g].toFixed(2));
            li.addEventListener("click", function () { viewer.select(o.g); });
          });
        html("p", { class: "empty", style: "margin:.7rem 0 0" }, panel, "Select a gene on the map for its scores. Connectors are not ranked.");
      }
    }
    viewer.refresh = refresh;
    viewer.set({ zi: 1, mi: 2 }); refresh();
    var truthKey = $("#truthKey");
    truthKey.insertAdjacentHTML("afterend", '<span id="hubKey" hidden><svg width="18" height="18" class="sw"><circle cx="9" cy="9" r="7" fill="none" stroke="#4a4770" stroke-dasharray="1.5 2"/></svg>hub</span>');
    $("#vTruth").addEventListener("change", function (e) { $("#hubKey").hidden = !e.target.checked; });
  })();

  /* =====================================================================================
     AND-pair note
     ===================================================================================== */
  (function () {
    var host = $("#andNote"); if (!host) return;
    var st = D.subtypes.filter(function (s) { return s.and_pair_ranks; })[0];
    if (!st) return;
    var r = st.and_pair_ranks, n = D.meta.n_genes;
    host.innerHTML = "";
    html("h3", {}, host, "A check against a known interaction");
    html("p", {}, host, "In the synthetic cohort, " + st.name + " is built to need two genes high at the same time, while the largest class has only one of them. Over " + st.n_patients + " correctly classified test patients, Integrated Gradients ranks the pair " + r.ig[0] + " and " + r.ig[1] + " of " + n + " genes. The HSIC Lasso surrogate, fitted on the first 4 of those patients, ranks it " + r.surrogate[0] + " and " + r.surrogate[1] + ".");
    html("p", { class: "caption" }, host, "One run on synthetic data. The paper's claim is that a surrogate can recover effects an additive path integral under-credits; the planned ablation with ζ = 0 on TCGA-BRCA is what tests it.");
  })();

  /* =====================================================================================
     Charts
     ===================================================================================== */
  var SER = {
    XplainOnco: { color: "#d4476b", dash: "", w: 3, cls: "" },
    degree: { color: "#5a57b3", dash: "7 4", w: 2.4, cls: "dash" },
    random: { color: "#8d8ba6", dash: "1.5 4", w: 2.6, cls: "dot" }
  };
  function lineChart(host, o) {
    host.innerHTML = "";
    html("h4", {}, host, o.title); html("p", { class: "sub" }, host, o.sub);
    var Wc = 360, Hc = 240, L = 44, R = 12, T = 8, B = 38, pw = Wc - L - R, ph = Hc - T - B;
    var xs = o.series[0].x, xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
    var ymin = o.ymin, ymax = o.ymax;
    var X = function (v) { return L + (v - xmin) / ((xmax - xmin) || 1) * pw; };
    var Y = function (v) { return T + (1 - (v - ymin) / (ymax - ymin)) * ph; };
    var s = svg("svg", { viewBox: "0 0 " + Wc + " " + Hc, role: "img", "aria-label": o.title + ". " + o.sub }, host);
    var yt = o.yticks;
    yt.forEach(function (v) {
      svg("line", { x1: L, x2: L + pw, y1: Y(v), y2: Y(v), class: "gr" }, s);
      svg("text", { x: L - 7, y: Y(v) + 4, "text-anchor": "end" }, s).textContent = v.toFixed(o.ydec != null ? o.ydec : 2);
    });
    o.xticks.forEach(function (v) {
      svg("line", { x1: X(v), x2: X(v), y1: T + ph, y2: T + ph + 4, class: "ax" }, s);
      svg("text", { x: X(v), y: T + ph + 17, "text-anchor": "middle" }, s).textContent = o.xlabels ? o.xlabels(v) : v;
    });
    svg("line", { x1: L, x2: L + pw, y1: T + ph, y2: T + ph, class: "ax" }, s);
    svg("text", { x: L + pw / 2, y: Hc - 4, "text-anchor": "middle" }, s).textContent = o.xlabel;
    o.series.forEach(function (se) {
      var d = se.x.map(function (v, i) { return (i ? "L" : "M") + X(v).toFixed(1) + " " + Y(se.y[i]).toFixed(1); }).join("");
      svg("path", { d: d, class: "ln", stroke: se.color, "stroke-width": se.w, "stroke-dasharray": se.dash || "none" }, s);
      if (o.markers) se.x.forEach(function (v, i) { svg("circle", { cx: X(v), cy: Y(se.y[i]), r: 3.6, fill: se.color, stroke: "#fff", "stroke-width": 1.5 }, s); });
    });
    var cross = svg("line", { y1: T, y2: T + ph, stroke: "#1b1840", "stroke-width": 1, opacity: 0 }, s);
    var dots = o.series.map(function (se) { return svg("circle", { r: 5, fill: se.color, stroke: "#fff", "stroke-width": 2, opacity: 0 }, s); });
    var tip = html("div", { class: "tip" }, host);
    var hit = svg("rect", { x: L, y: T, width: pw, height: ph, fill: "transparent" }, s);
    function move(ev) {
      var r = s.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * Wc;
      var best = 0, bd = 1e9; xs.forEach(function (v, i) { var d = Math.abs(X(v) - px); if (d < bd) { bd = d; best = i; } });
      cross.setAttribute("x1", X(xs[best])); cross.setAttribute("x2", X(xs[best])); cross.setAttribute("opacity", 0.5);
      o.series.forEach(function (se, k) { dots[k].setAttribute("cx", X(xs[best])); dots[k].setAttribute("cy", Y(se.y[best])); dots[k].setAttribute("opacity", 1); });
      tip.innerHTML = "<b>" + (o.tipx ? o.tipx(xs[best], best) : xs[best]) + "</b><br>" + o.series.map(function (se) { return se.name + ": " + se.y[best].toFixed(2); }).join("<br>");
      tip.style.opacity = 1;
      var hr = host.getBoundingClientRect();
      var left = (X(xs[best]) / Wc) * r.width + (r.left - hr.left) + 12;
      tip.style.left = Math.min(left, hr.width - tip.offsetWidth - 4) + "px"; tip.style.top = (r.top - hr.top + 30) + "px";
    }
    function leave() { cross.setAttribute("opacity", 0); dots.forEach(function (d) { d.setAttribute("opacity", 0); }); tip.style.opacity = 0; }
    hit.addEventListener("mousemove", move); hit.addEventListener("mouseleave", leave); hit.addEventListener("touchstart", function (e) { move(e.touches[0]); }, { passive: true });
    var det = html("details", { class: "tabledata" }, host); html("summary", {}, det, "Show as table");
    var tb = html("table", {}, det), tr = html("tr", {}, html("thead", {}, tb));
    html("th", { scope: "col" }, tr, o.xlabel); o.series.forEach(function (se) { html("th", { scope: "col" }, tr, se.name); });
    var body = html("tbody", {}, tb);
    xs.forEach(function (v, i) { var r = html("tr", {}, body); html("td", {}, r, o.tipx ? o.tipx(v, i) : v); o.series.forEach(function (se) { html("td", {}, r, se.y[i].toFixed(3)); }); });
  }

  (function faith() {
    var segF = $("#fSubtype"); if (!segF) return;
    var legend = $("#chLegend");
    legend.innerHTML = ["XplainOnco ranking", "degree ranking (reference)", "random ranking (reference)"].map(function (t, i) {
      var k = ["XplainOnco", "degree", "random"][i];
      return '<span style="--col:' + SER[k].color + '"><i class="' + SER[k].cls + '"></i>' + t + "</span>";
    }).join("");
    D.subtypes.forEach(function (st, i) {
      var b = html("button", { type: "button", "aria-pressed": i === 0 ? "true" : "false" }, segF, st.name);
      b.addEventListener("click", function () { $$("button", segF).forEach(function (x, j) { x.setAttribute("aria-pressed", j === i ? "true" : "false"); }); draw(i); });
    });
    function draw(i) {
      var st = D.subtypes[i], f = st.faithfulness, keys = ["XplainOnco", "degree", "random"];
      var kk = f.XplainOnco.k, n = D.meta.n_genes;
      var mk = function (field) { return keys.map(function (k) { return { name: k === "XplainOnco" ? "XplainOnco" : k, color: SER[k].color, dash: SER[k].dash, w: SER[k].w, x: kk, y: f[k][field] }; }); };
      var xt = [kk[0], kk[Math.floor(kk.length / 3)], kk[Math.floor(2 * kk.length / 3)], kk[kk.length - 1]];
      lineChart($("#chDel"), { title: "Deletion", sub: "Predicted-class probability as top genes are removed. Lower is more faithful.", series: mk("deletion"), ymin: 0, ymax: 1, yticks: [0, 0.25, 0.5, 0.75, 1], xticks: xt, xlabel: "genes removed", tipx: function (v) { return v + " genes removed"; } });
      lineChart($("#chIns"), { title: "Insertion", sub: "Probability as top genes are restored to an all-baseline input. Higher is more faithful.", series: mk("insertion"), ymin: 0, ymax: 1, yticks: [0, 0.25, 0.5, 0.75, 1], xticks: xt, xlabel: "genes restored", tipx: function (v) { return v + " genes restored"; } });
      var a = function (k, w) { return f[k][w + "_auc"].toFixed(2); };
      $("#chNote").textContent = "Area under the curve for " + st.name + " over " + n + " genes. Deletion, lower is better: XplainOnco " + a("XplainOnco", "deletion") + ", degree " + a("degree", "deletion") + ", random " + a("random", "deletion") + ". Insertion, higher is better: XplainOnco " + a("XplainOnco", "insertion") + ", degree " + a("degree", "insertion") + ", random " + a("random", "insertion") + ". The ranking was built from the first " + Math.max(1, Math.floor(st.n_patients / 2)) + " correctly classified test patients and the curves are averaged over the other " + (st.n_patients - Math.max(1, Math.floor(st.n_patients / 2))) + ", so the ranking is not scored on the patients that produced it. One synthetic run with very few patients, not a benchmark.";
    }
    draw(0);
  })();

  (function sanity() {
    var host = $("#chSan"); if (!host) return;
    var sn = D.sanity || [], txt = $("#sanText");
    if (!sn.length) { host.textContent = "The sanity run did not complete for this export."; return; }
    var names = sn.map(function (s) { return s.randomised_through; });
    var xs = sn.map(function (s, i) { return i + 1; });
    lineChart(host, { title: "Agreement with the trained model's scores", sub: "Spearman correlation after each cumulative randomisation step, output first.", markers: true,
      series: [{ name: "Spearman", color: "#d4476b", dash: "", w: 2.6, x: xs, y: sn.map(function (s) { return s.spearman; }) }],
      ymin: -0.25, ymax: 1, yticks: [0, 0.25, 0.5, 0.75, 1], xticks: [1, Math.ceil(xs.length / 2), xs.length], xlabel: "layers randomised (output towards input)",
      tipx: function (v, i) { return "through " + names[i]; } });
    var last = sn[sn.length - 1].spearman, first = sn[0].spearman, low = Math.min.apply(null, sn.map(function (s) { return s.spearman; }));
    var lead = "Cascading randomisation after Adebayo et al.: weights are re-initialised starting at the classifier and moving towards the input, and the attribution is recomputed each time. Here the correlation with the original scores is <strong>" + first.toFixed(2) + "</strong> after the first step and <strong>" + last.toFixed(2) + "</strong> after all " + sn.length + " (lowest " + low.toFixed(2) + "). ";
    var verdict = last > 0.3
      ? "<strong>This run does not pass the check cleanly.</strong> A large share of the ranking survives a fully randomised model. One likely reason is that each gene's score is its input difference from the baseline times a gradient, and the input difference does not change when weights are randomised; this run has not separated that effect from a real fault, so treat the explanations here as not yet validated."
      : "The correlation falls as more of the model is randomised, which is what a score that depends on the trained weights should do.";
    txt.innerHTML = lead + verdict + " Hover a point for the layer.";
  })();

})();
