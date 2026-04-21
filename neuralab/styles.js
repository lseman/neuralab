export const STYLES = `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; min-height: 100%; }
:root { color-scheme: dark; }
button, input, select, textarea { font: inherit; }

.nl-root {
  --canvas-bg: #1a1a1a;
  --canvas-dot: rgba(255,255,255,0.06);
  --chrome: #141414;
  --chrome-2: #1e1e1e;
  --chrome-3: #252525;
  --chrome-4: #2d2d2d;
  --bd: rgba(255,255,255,0.08);
  --bd-hi: rgba(255,255,255,0.16);
  --bd-focus: rgba(99,179,237,0.7);
  --fg-0: #f5f5f5;
  --fg-1: #d4d4d4;
  --fg-2: #a3a3a3;
  --fg-3: #6b6b6b;
  --fg-4: #404040;
  /* Node port colors */
  --node-blue:   #3b82f6;
  --node-orange: #f97316;
  --node-green:  #22c55e;
  --node-violet: #a855f7;
  --node-rose:   #f43f5e;
  --node-teal:   #14b8a6;
  --node-amber:  #f59e0b;
  /* Semantic */
  --accent:    #3b82f6;
  --accent-hi: #60a5fa;
  --accent-dim:rgba(59,130,246,0.15);
  --green:     #22c55e;
  --green-dim: rgba(34,197,94,0.14);
  --amber:     #f59e0b;
  --rose:      #f43f5e;
  --violet:    #a855f7;
  --surface-glow: radial-gradient(circle at top left, rgba(20,184,166,0.18), transparent 34%), radial-gradient(circle at bottom right, rgba(59,130,246,0.22), transparent 28%);
  --shadow-node: 0 4px 24px rgba(0,0,0,0.5), 0 1px 3px rgba(0,0,0,0.4);
  --shadow-panel: 0 8px 32px rgba(0,0,0,0.6);

  min-height: 100vh;
  color: var(--fg-0);
  background: var(--canvas-bg);
  font: 13px/1.55 'Inter', system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
  overflow: hidden;
}

body { background: transparent; }

.nl-root .mono {
  font-family: 'JetBrains Mono', monospace;
  font-variant-numeric: tabular-nums;
}

/* ── Canvas dotgrid background ───────────────────────────────── */
.nl-canvas-bg {
  position: fixed;
  inset: 0;
  z-index: 0;
  background-image: radial-gradient(circle, var(--canvas-dot) 1px, transparent 1px);
  background-size: 24px 24px;
  pointer-events: none;
}

/* ── Top chrome bar ──────────────────────────────────────────── */
.nl-topbar {
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 50;
  height: 48px;
  display: flex;
  align-items: center;
  gap: 0;
  background: var(--chrome);
  border-bottom: 1px solid var(--bd);
}

.nl-topbar-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 16px;
  height: 100%;
  border-right: 1px solid var(--bd);
  min-width: 200px;
}

.nl-brand-icon {
  width: 28px; height: 28px;
  border-radius: 7px;
  background: var(--node-blue);
  display: grid; place-items: center;
  flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(59,130,246,0.4);
}

.nl-brand-text {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.nl-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--fg-0);
  letter-spacing: -0.01em;
  line-height: 1;
}

.nl-subtitle {
  font-size: 10px;
  color: var(--fg-3);
  letter-spacing: 0.02em;
  line-height: 1;
  margin-top: 2px;
}

/* ── Topbar center: run controls ─────────────────────────────── */
.nl-topbar-controls {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 0 16px;
  height: 100%;
  border-right: 1px solid var(--bd);
}

/* ── Topbar right: metrics ───────────────────────────────────── */
.nl-topbar-metrics {
  display: flex;
  align-items: stretch;
  height: 100%;
  margin-left: auto;
}

.nl-tmetric {
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 0 14px;
  border-left: 1px solid var(--bd);
  min-width: 90px;
}

.nl-tmetric .k {
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--fg-4);
  line-height: 1;
}

.nl-tmetric .v {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 500;
  color: var(--fg-0);
  line-height: 1;
  margin-top: 3px;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.nl-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  height: 32px;
  padding: 0 12px;
  border-radius: 6px;
  border: 1px solid var(--bd);
  background: var(--chrome-3);
  color: var(--fg-1);
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
  white-space: nowrap;
}

.nl-btn:hover {
  background: var(--chrome-4);
  border-color: var(--bd-hi);
  color: var(--fg-0);
}

.nl-btn.primary {
  background: var(--node-blue);
  border-color: var(--node-blue);
  color: #fff;
  font-weight: 600;
  box-shadow: 0 2px 8px rgba(59,130,246,0.3);
}

.nl-btn.primary:hover {
  background: var(--accent-hi);
  border-color: var(--accent-hi);
}

.nl-btn.icon {
  width: 32px;
  padding: 0;
}

.nl-btn.ghost {
  background: transparent;
  border-color: transparent;
}

.nl-btn.ghost:hover {
  background: var(--chrome-3);
  border-color: var(--bd);
}

.nl-btn.danger {
  color: var(--rose);
  border-color: transparent;
  background: transparent;
}

.nl-btn.danger:hover {
  background: rgba(244,63,94,0.12);
  border-color: rgba(244,63,94,0.3);
}

.nl-btn:disabled, .nl-btn:disabled:hover {
  opacity: 0.38;
  cursor: not-allowed;
  background: var(--chrome-2);
  border-color: var(--bd);
  color: var(--fg-3);
  box-shadow: none;
}

/* ── Left sidebar ────────────────────────────────────────────── */
.nl-sidebar {
  position: fixed;
  top: 48px; left: 0; bottom: 0;
  width: 288px;
  z-index: 40;
  background: var(--chrome);
  border-right: 1px solid var(--bd);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.nl-sidebar-scroll {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
}

.nl-sidebar-scroll::-webkit-scrollbar { width: 4px; }
.nl-sidebar-scroll::-webkit-scrollbar-thumb {
  background: var(--chrome-4);
  border-radius: 2px;
}

/* ── Right sidebar ───────────────────────────────────────────── */
.nl-sidebar-right {
  position: fixed;
  top: 48px; right: 0; bottom: 0;
  width: 280px;
  z-index: 40;
  background: var(--chrome);
  border-left: 1px solid var(--bd);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── Center canvas area ──────────────────────────────────────── */
.nl-canvas {
  position: fixed;
  top: 48px;
  left: 288px;
  right: 280px;
  bottom: 0;
  z-index: 10;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 14px;
}

.nl-hero {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  padding: 20px 22px;
  border: 1px solid var(--bd);
  border-radius: 20px;
  background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01)), var(--surface-glow), var(--chrome-2);
  box-shadow: var(--shadow-node);
}

.nl-hero-copy {
  max-width: 720px;
}

.nl-hero-kicker {
  font-size: 10px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--node-teal);
  margin-bottom: 10px;
  font-weight: 700;
}

.nl-hero-title {
  margin: 0;
  font-size: 31px;
  line-height: 1.06;
  letter-spacing: -0.04em;
  color: var(--fg-0);
  max-width: 14ch;
}

.nl-hero-text {
  margin: 12px 0 0;
  max-width: 56ch;
  font-size: 14px;
  color: var(--fg-2);
}

.nl-hero-actions {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  flex-shrink: 0;
}

.nl-overview-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 12px 0;
}

.nl-overview-card {
  padding: 14px;
  border-radius: 16px;
  border: 1px solid var(--bd);
  background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01)), var(--chrome-2);
  box-shadow: var(--shadow-node);
  min-height: 112px;
}

.nl-overview-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--fg-4);
}

.nl-overview-value {
  margin-top: 10px;
  font-size: 21px;
  line-height: 1.1;
  letter-spacing: -0.03em;
  color: var(--fg-0);
}

.nl-overview-meta {
  margin-top: 8px;
  font-size: 11px;
  line-height: 1.5;
  color: var(--fg-3);
}

.nl-preset-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr)) 1.2fr;
  gap: 10px;
}

.nl-preset-card,
.nl-snapshot-card {
  text-align: left;
  border: 1px solid var(--bd);
  border-radius: 14px;
  background: var(--chrome-3);
  color: var(--fg-0);
  cursor: pointer;
  transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
}

.nl-preset-card {
  padding: 14px;
}

.nl-preset-card:hover,
.nl-snapshot-card:hover {
  transform: translateY(-1px);
  border-color: var(--bd-hi);
  background: var(--chrome-4);
}

.nl-preset-title-row,
.nl-snapshot-top,
.nl-snapshot-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.nl-preset-title,
.nl-snapshot-name {
  font-size: 13px;
  font-weight: 600;
}

.nl-preset-copy,
.nl-snapshot-meta {
  margin-top: 8px;
  font-size: 11px;
  line-height: 1.5;
  color: var(--fg-3);
}

.nl-preset-meta {
  margin-top: 14px;
  font-size: 10px;
  color: var(--fg-4);
}

.nl-snapshot-column {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
}

.nl-snapshot-header {
  padding: 0 2px;
  color: var(--fg-2);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.nl-snapshot-card {
  padding: 12px;
}

.nl-snapshot-card.active {
  border-color: var(--node-blue);
  box-shadow: 0 0 0 1px rgba(59,130,246,0.25);
}

.nl-snapshot-step {
  font-size: 10px;
  color: var(--node-teal);
}

.nl-snapshot-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 10px;
}

.nl-snapshot-key {
  display: block;
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--fg-4);
}

.nl-snapshot-val {
  display: block;
  margin-top: 4px;
  font-size: 13px;
  color: var(--fg-1);
}

.nl-snapshot-actions {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}

.nl-snapshot-actions .nl-btn {
  flex: 1;
}

.nl-empty-state,
.nl-copy {
  padding: 10px 12px;
  border: 1px solid var(--bd);
  border-radius: 10px;
  background: rgba(255,255,255,0.02);
  font-size: 11px;
  line-height: 1.55;
  color: var(--fg-3);
}

/* ── Node cards ──────────────────────────────────────────────── */
.nl-node {
  background: var(--chrome-2);
  border: 1px solid var(--bd);
  border-radius: 10px;
  overflow: hidden;
  box-shadow: var(--shadow-node);
  transition: border-color 0.14s;
  position: relative;
}

.nl-node:hover {
  border-color: var(--bd-hi);
}

.nl-node-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px 9px;
  background: var(--chrome-3);
  border-bottom: 1px solid var(--bd);
}

.nl-node-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.nl-node-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--fg-0);
  letter-spacing: 0.01em;
  flex: 1;
}

.nl-node-type {
  font-size: 9px;
  font-family: 'JetBrains Mono', monospace;
  color: var(--fg-4);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.nl-node-body {
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

/* Port row at bottom of node */
.nl-node-ports {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 10px 8px;
  border-top: 1px solid var(--bd);
  background: rgba(0,0,0,0.15);
}

.nl-port {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 9px;
  font-family: 'JetBrains Mono', monospace;
  color: var(--fg-4);
  letter-spacing: 0.03em;
}

.nl-port-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  border: 1.5px solid currentColor;
}

.nl-port.input-port { color: var(--node-teal); }
.nl-port.output-port { color: var(--node-orange); flex-direction: row-reverse; }

/* ── Section accordion in sidebar ───────────────────────────── */
.nl-section {
  border-bottom: 1px solid var(--bd);
}

.nl-section-hd {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  cursor: default;
  user-select: none;
  background: var(--chrome);
}

.nl-section-hd-label {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--fg-3);
  flex: 1;
}

.nl-section-hd .badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  color: var(--fg-4);
  background: var(--chrome-3);
  padding: 2px 7px;
  border-radius: 3px;
}

.nl-section-body {
  padding: 12px 14px;
  background: var(--chrome);
}

/* ── Flow connector line between nodes ───────────────────────── */
.nl-connector {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 18px;
  position: relative;
}

.nl-connector::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 1px;
  height: 100%;
  background: linear-gradient(180deg, var(--node-orange) 0%, var(--node-blue) 100%);
  opacity: 0.4;
}

.nl-connector-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--chrome-4);
  border: 1px solid var(--bd-hi);
  z-index: 1;
}

/* ── Form controls ───────────────────────────────────────────── */
.nl-stack { display: flex; flex-direction: column; gap: 10px; }
.nl-row { display: flex; align-items: center; gap: 8px; }
.nl-row.between { justify-content: space-between; }

.nl-label {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--fg-3);
  white-space: nowrap;
}

.nl-value-display {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--fg-1);
  padding: 1px 6px;
  background: var(--chrome-3);
  border-radius: 4px;
  border: 1px solid var(--bd);
}

.nl-select, .nl-input {
  width: 100%;
  height: 30px;
  padding: 0 9px;
  border-radius: 5px;
  border: 1px solid var(--bd);
  background: var(--chrome-3);
  color: var(--fg-0);
  font-size: 12px;
  outline: none;
  transition: border-color 0.12s, box-shadow 0.12s;
}

.nl-select:focus, .nl-input:focus {
  border-color: var(--bd-focus);
  box-shadow: 0 0 0 2px rgba(99,179,237,0.15);
}

.nl-select.mono, .nl-input.mono {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
}

.nl-slider {
  width: 100%;
  appearance: none;
  background: transparent;
  cursor: pointer;
  height: 18px;
}

.nl-slider::-webkit-slider-runnable-track {
  height: 3px;
  border-radius: 999px;
  background: var(--chrome-4);
}

.nl-slider::-webkit-slider-thumb {
  appearance: none;
  width: 13px; height: 13px;
  border-radius: 50%;
  background: var(--node-blue);
  margin-top: -5px;
  border: 2px solid var(--chrome-2);
  box-shadow: 0 0 0 1px var(--node-blue);
  transition: transform 0.1s;
}

.nl-slider::-webkit-slider-thumb:hover { transform: scale(1.2); }

.nl-slider::-moz-range-track {
  height: 3px;
  border-radius: 999px;
  background: var(--chrome-4);
}

.nl-slider::-moz-range-thumb {
  width: 13px; height: 13px;
  border-radius: 50%;
  background: var(--node-blue);
  border: 2px solid var(--chrome-2);
}

.nl-check {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--fg-1);
  cursor: pointer;
  user-select: none;
}

.nl-check input {
  width: 14px; height: 14px;
  accent-color: var(--node-blue);
  cursor: pointer;
}

/* ── Dataset thumbnails ──────────────────────────────────────── */
.nl-dataset-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 5px;
}

.nl-dataset-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 8px 4px;
  border-radius: 7px;
  border: 1px solid var(--bd);
  background: var(--chrome-3);
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s;
}

.nl-dataset-btn:hover {
  background: var(--chrome-4);
  border-color: var(--bd-hi);
}

.nl-dataset-btn.active {
  background: var(--accent-dim);
  border-color: var(--node-blue);
}

.nl-dataset-name {
  font-size: 8px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--fg-3);
}

.nl-dataset-btn.active .nl-dataset-name { color: var(--accent-hi); }

/* ── Kbd ─────────────────────────────────────────────────────── */
.nl-kbd {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  padding: 2px 6px;
  border-radius: 4px;
  background: var(--chrome-3);
  border: 1px solid var(--bd-hi);
  color: var(--fg-2);
}

/* ── Chip ────────────────────────────────────────────────────── */
.nl-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 8px;
  border-radius: 4px;
  border: 1px solid var(--bd);
  background: var(--chrome-3);
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--fg-3);
}

.nl-chip.running {
  background: rgba(34,197,94,0.12);
  border-color: rgba(34,197,94,0.35);
  color: var(--green);
}

.nl-chip.blue {
  background: var(--accent-dim);
  border-color: rgba(59,130,246,0.35);
  color: var(--accent-hi);
}

/* ── Canvas panels ───────────────────────────────────────────── */
.nl-canvas-panel {
  background: var(--chrome-2);
  border: 1px solid var(--bd);
  border-radius: 10px;
  overflow: hidden;
  box-shadow: var(--shadow-node);
}

.nl-flow-wrapper {
  width: 100%;
  min-height: 360px;
  border: 1px solid var(--bd);
  border-radius: 16px;
  overflow: hidden;
  background: var(--chrome-2);
}

.nl-flow-node-label {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.nl-flow-node-meta {
  font-size: 10px;
  color: var(--fg-4);
  line-height: 1.4;
}

.nl-flow-explain {
  margin-top: 12px;
  font-size: 11px;
  color: var(--fg-3);
  padding: 0 12px;
}

.nl-network-diagram {
  width: 100%;
  min-height: 248px;
  border-radius: 20px;
  background: transparent;
  border: 0;
  padding: 8px 8px 10px;
  box-shadow: none;
}

.nl-network-stage {
  position: relative;
  margin: 0 auto;
}

.nl-network-toolbar {
  position: relative;
  height: 28px;
  margin-bottom: 6px;
}

.nl-network-layer-control {
  position: absolute;
  top: 0;
  display: flex;
  justify-content: center;
}

.nl-layer-spin {
  display: inline-flex;
  gap: 4px;
}

.nl-btn.mini {
  min-width: 26px;
  padding: 0 6px;
  height: 24px;
  font-size: 11px;
  border-radius: 999px;
  background: var(--chrome-3);
}

.nl-layer-spacer {
  width: 28px;
}

.nl-network-svg-wrap {
  width: 100%;
  overflow: auto;
  padding-bottom: 4px;
}

.nl-network-svg {
  display: block;
  max-width: none;
  margin: 0;
}

.nl-block-fallback {
  display: grid;
  gap: 8px;
  margin-top: 12px;
  padding: 0 12px;
}

.nl-block-chip {
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid var(--bd);
  background: var(--chrome-3);
}

.nl-block-chip-name {
  display: block;
  font-size: 12px;
  font-weight: 700;
  color: var(--fg-0);
}

.nl-block-chip-meta {
  display: block;
  margin-top: 4px;
  font-size: 10px;
  color: var(--fg-4);
}

.nl-canvas-panel-hd {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 13px;
  background: var(--chrome-3);
  border-bottom: 1px solid var(--bd);
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--fg-3);
}

.nl-canvas-panel-body {
  padding: 12px;
}

/* ── Gradient flow bars ──────────────────────────────────────── */
.nl-grad-bar-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.nl-grad-bar-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  color: var(--fg-4);
  width: 48px;
  flex-shrink: 0;
}

.nl-grad-bar-track {
  flex: 1;
  height: 4px;
  background: var(--chrome-4);
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 2px;
}

.nl-grad-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.25s ease;
}

.nl-grad-bar-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 8px;
  color: var(--fg-4);
  width: 50px;
  text-align: right;
  flex-shrink: 0;
}

/* ── Optimizer toggle pills ──────────────────────────────────── */
.nl-opt-pills {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.nl-opt-pill {
  flex: 1;
  min-width: 52px;
  padding: 5px 6px;
  border-radius: 5px;
  border: 1px solid var(--bd);
  background: var(--chrome-3);
  color: var(--fg-3);
  font-size: 10px;
  font-weight: 500;
  cursor: pointer;
  text-align: center;
  transition: all 0.12s;
  font-family: 'JetBrains Mono', monospace;
}

.nl-opt-pill:hover {
  background: var(--chrome-4);
  color: var(--fg-1);
}

.nl-opt-pill.active {
  background: var(--accent-dim);
  border-color: var(--node-blue);
  color: var(--accent-hi);
  font-weight: 600;
}

/* ── Status row ──────────────────────────────────────────────── */
.nl-statusbar {
  position: fixed;
  bottom: 0; left: 288px; right: 280px;
  z-index: 40;
  height: 26px;
  background: var(--chrome);
  border-top: 1px solid var(--bd);
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 0 14px;
  font-size: 10px;
  color: var(--fg-4);
}

.nl-statusbar .mono { font-family: 'JetBrains Mono', monospace; font-size: 10px; }

/* ── Legend dots ─────────────────────────────────────────────── */
.nl-legend {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 10px;
  color: var(--fg-2);
}

.nl-legend-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 4px;
}

/* ── Diag rows ───────────────────────────────────────────────── */
.nl-diag-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px solid var(--bd);
  font-size: 11px;
}

.nl-diag-row:last-child { border-bottom: none; }
.nl-diag-key { color: var(--fg-3); }
.nl-diag-val { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--fg-1); }

/* ── Modal ───────────────────────────────────────────────────── */
.nl-modal {
  position: fixed;
  inset: 0;
  z-index: 100;
  display: grid;
  place-items: center;
  padding: 20px;
  background: rgba(0,0,0,0.7);
  backdrop-filter: blur(6px);
}

.nl-modal-card {
  width: min(760px, 100%);
  max-height: 84vh;
  overflow: hidden;
  border-radius: 10px;
  border: 1px solid var(--bd-hi);
  background: var(--chrome);
  box-shadow: var(--shadow-panel);
  display: flex;
  flex-direction: column;
}

.nl-modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--bd);
  background: var(--chrome-3);
}

.nl-modal-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--fg-0);
}

.nl-modal-body {
  margin: 0;
  padding: 16px;
  overflow: auto;
  color: var(--fg-1);
  font: 11px/1.7 'JetBrains Mono', monospace;
  flex: 1;
  background: #0d0d0d;
}

.nl-share-link {
  margin: 0;
  padding: 14px;
  border-radius: 10px;
  border: 1px solid var(--bd);
  background: var(--chrome);
  color: var(--fg-1);
  font: 11px/1.65 'JetBrains Mono', monospace;
  white-space: pre-wrap;
  word-break: break-all;
}

/* ── Scrollbar shared ────────────────────────────────────────── */
.nl-scroll::-webkit-scrollbar { width: 4px; height: 4px; }
.nl-scroll::-webkit-scrollbar-thumb { background: var(--chrome-4); border-radius: 2px; }

/* ── Architecture block stack ────────────────────────────────── */
.nl-arch-stack {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 0;
  padding: 4px 0;
}

.nl-arch-block {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  background: var(--chrome-3);
  border: 1px solid var(--bd);
  border-radius: 8px;
  cursor: pointer;
  transition: border-color 0.12s, background 0.12s;
  user-select: none;
}

.nl-arch-block:hover {
  border-color: var(--bd-hi);
  background: var(--chrome-4);
}

.nl-arch-block.selected {
  border-color: var(--node-blue);
  background: rgba(59,130,246,0.08);
  box-shadow: 0 0 0 2px rgba(59,130,246,0.2);
}

.nl-arch-block--input { border-left: 2px solid var(--node-teal); }
.nl-arch-block--output { border-left: 2px solid var(--node-orange); }

.nl-arch-block-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.nl-arch-block-body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.nl-arch-block-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--fg-1);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.nl-arch-block-meta {
  font-size: 9px;
  font-family: 'JetBrains Mono', monospace;
  color: var(--fg-4);
}

.nl-arch-block-tag {
  font-size: 8px;
  font-family: 'JetBrains Mono', monospace;
  color: var(--fg-4);
  background: var(--chrome-4);
  border: 1px solid var(--bd);
  border-radius: 3px;
  padding: 1px 5px;
}

.nl-arch-block-actions {
  display: flex;
  gap: 2px;
  opacity: 0;
  transition: opacity 0.12s;
}

.nl-arch-block:hover .nl-arch-block-actions { opacity: 1; }
.nl-arch-block.selected .nl-arch-block-actions { opacity: 1; }

.nl-arch-action {
  background: none;
  border: 1px solid var(--bd);
  border-radius: 4px;
  color: var(--fg-3);
  font-size: 10px;
  line-height: 1;
  padding: 2px 5px;
  cursor: pointer;
  transition: background 0.1s, color 0.1s;
}

@media (max-width: 1440px) {
  .nl-overview-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .nl-preset-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .nl-snapshot-column {
    grid-column: 1 / -1;
  }
}

@media (max-width: 1180px) {
  .nl-sidebar,
  .nl-sidebar-right {
    position: static;
    width: auto;
    height: auto;
    border: 1px solid var(--bd);
    border-radius: 16px;
    margin: 0 14px 10px;
  }

  .nl-canvas {
    position: static;
    left: auto;
    right: auto;
    top: auto;
    bottom: auto;
    padding: 10px 14px 40px;
  }

  .nl-statusbar {
    left: 0;
    right: 0;
  }
}

@media (max-width: 900px) {
  .nl-topbar {
    position: static;
    height: auto;
    flex-wrap: wrap;
  }

  .nl-topbar-brand,
  .nl-topbar-controls,
  .nl-topbar-metrics {
    border-right: 0;
    border-left: 0;
    width: 100%;
  }

  .nl-topbar-metrics {
    flex-wrap: wrap;
  }

  .nl-tmetric {
    min-width: 33.33%;
  }

  .nl-canvas,
  .nl-sidebar,
  .nl-sidebar-right {
    margin-top: 0;
  }

  .nl-hero {
    flex-direction: column;
  }

  .nl-hero-title {
    font-size: 26px;
    max-width: none;
  }

  .nl-overview-grid,
  .nl-preset-grid {
    grid-template-columns: 1fr;
  }

  .nl-dataset-grid {
    grid-template-columns: repeat(3, 1fr);
  }

  .nl-statusbar {
    position: static;
    height: auto;
    padding: 10px 14px;
    flex-wrap: wrap;
  }
}

.nl-arch-action:hover { background: var(--chrome-4); color: var(--fg-0); }
.nl-arch-action:disabled { opacity: 0.25; cursor: default; }
.nl-arch-action.danger:hover { background: rgba(244,63,94,0.15); border-color: var(--rose); color: var(--rose); }

.nl-arch-connector {
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 18px;
  position: relative;
  flex-shrink: 0;
}

.nl-arch-connector-line {
  width: 1px;
  flex: 1;
  background: var(--bd-hi);
}

.nl-arch-connector-arrow {
  width: 0;
  height: 0;
  border-left: 4px solid transparent;
  border-right: 4px solid transparent;
  border-top: 5px solid var(--bd-hi);
}

/* ── ReactFlow wrapper (kept for reference, unused) ──────────── */
.nl-flow-wrapper {
  width: 100%;
  height: 320px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--bd);
  background: #111;
  margin-bottom: 10px;
  position: relative;
}

/* Override ReactFlow default node styles */
.nl-flow-wrapper .react-flow__node-default {
  background: var(--chrome-3) !important;
  border: 1px solid var(--bd) !important;
  border-radius: 8px !important;
  color: var(--fg-0) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 11px !important;
  padding: 8px 12px !important;
  min-width: 110px !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
}

.nl-flow-wrapper .react-flow__node-default.selected {
  border-color: var(--node-blue) !important;
  box-shadow: 0 0 0 2px rgba(59,130,246,0.3), 0 4px 12px rgba(0,0,0,0.4) !important;
}

.nl-flow-wrapper .react-flow__node-default:hover {
  border-color: var(--bd-hi) !important;
}

.nl-flow-wrapper .react-flow__edge-path {
  stroke: var(--node-blue) !important;
  stroke-width: 1.5 !important;
  opacity: 0.6 !important;
}

.nl-flow-wrapper .react-flow__edge.animated .react-flow__edge-path {
  stroke-dasharray: 8 !important;
  animation: dashdraw 0.6s linear infinite !important;
  opacity: 0.8 !important;
}

@keyframes dashdraw {
  from { stroke-dashoffset: 16; }
  to { stroke-dashoffset: 0; }
}

.nl-flow-wrapper .react-flow__handle {
  width: 8px !important;
  height: 8px !important;
  border-radius: 50% !important;
  border: 2px solid var(--chrome-2) !important;
}

.nl-flow-wrapper .react-flow__handle-top    { background: var(--node-teal) !important; }
.nl-flow-wrapper .react-flow__handle-bottom { background: var(--node-orange) !important; }

.nl-flow-wrapper .react-flow__controls {
  background: var(--chrome-3) !important;
  border: 1px solid var(--bd) !important;
  border-radius: 6px !important;
  box-shadow: none !important;
}

.nl-flow-wrapper .react-flow__controls-button {
  background: transparent !important;
  border-bottom: 1px solid var(--bd) !important;
  color: var(--fg-2) !important;
  fill: var(--fg-2) !important;
}

.nl-flow-wrapper .react-flow__controls-button:hover {
  background: var(--chrome-4) !important;
}

.nl-flow-wrapper .react-flow__minimap {
  background: var(--chrome-2) !important;
  border: 1px solid var(--bd) !important;
  border-radius: 6px !important;
}

/* Node label content */
.nl-flow-node-label {
  text-align: center;
  line-height: 1.4;
}

.nl-flow-node-label strong {
  display: block;
  font-size: 11px;
  font-weight: 600;
  color: var(--fg-0);
}

.nl-flow-node-meta {
  font-size: 9px;
  color: var(--fg-4);
  margin-top: 2px;
  font-family: 'JetBrains Mono', monospace;
}

/* ── Flow explain text ───────────────────────────────────────── */
.nl-flow-explain {
  font-size: 10px;
  color: var(--fg-4);
  line-height: 1.5;
  margin-bottom: 10px;
}

/* ── Block chip list (text fallback) ─────────────────────────── */
.nl-block-fallback {
  display: none; /* hidden — only shown if ReactFlow fails to mount */
}

/* ── Generic copy text ───────────────────────────────────────── */
.nl-copy {
  font-size: 11px;
  color: var(--fg-3);
  line-height: 1.5;
}
`;
