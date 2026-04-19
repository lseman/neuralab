import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import {
  Activity,
  AlertTriangle,
  ChevronRight,
  Code2,
  Copy,
  Cpu,
  Database,
  Eye,
  Gauge,
  Layers,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Shuffle,
  Sparkles,
  StepForward,
  Target,
  X,
  Zap,
} from 'lucide-react';

import {
  ACT_NAMES,
  DATASETS,
  DEFAULT_CONFIG,
  OPTIMIZER_OPTIONS,
  SCHEDULER_OPTIONS,
  bceLoss,
  createNetwork,
  exportPyTorch,
  makeGrid,
  mseLoss,
  softmaxCELoss,
} from './core.js';
import { STYLES } from './styles.js';
import { DatasetThumb, DecisionBoundary, GradFlowBars, NeuronMap } from './visuals.jsx';

// ── Node color by layer type ──────────────────────────────────────────────
const NODE_COLORS = {
  input:     'var(--node-teal)',
  dense:     'var(--node-blue)',
  dropout:   'var(--node-violet)',
  layernorm: 'var(--node-amber)',
  batchnorm: 'var(--node-amber)',
  rmsnorm:   'var(--node-amber)',
  output:    'var(--node-orange)',
};

// ── Section accordion ─────────────────────────────────────────────────────
function Section({ icon: Icon, title, badge, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="nl-section">
      <div className="nl-section-hd" onClick={() => setOpen(v => !v)} style={{ cursor: 'pointer' }}>
        <Icon size={11} color="var(--fg-4)" />
        <span className="nl-section-hd-label">{title}</span>
        {badge ? <span className="badge">{badge}</span> : null}
        <ChevronRight size={11} color="var(--fg-4)"
          style={{ transform: open ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s', marginLeft: 2 }} />
      </div>
      {open ? <div className="nl-section-body">{children}</div> : null}
    </div>
  );
}

// ── Flow node card ────────────────────────────────────────────────────────
function FlowNode({ color, title, type, inputLabel, outputLabel, children, actions }) {
  return (
    <div className="nl-node">
      <div className="nl-node-header">
        <div className="nl-node-dot" style={{ background: color }} />
        <span className="nl-node-title">{title}</span>
        {actions}
        {type ? <span className="nl-node-type">{type}</span> : null}
      </div>
      {children ? <div className="nl-node-body">{children}</div> : null}
      <div className="nl-node-ports">
        <span className="nl-port input-port">
          <span className="nl-port-dot" style={{ borderColor: 'var(--node-teal)', background: 'var(--node-teal)' }} />
          {inputLabel ?? 'in'}
        </span>
        <span className="nl-port output-port">
          {outputLabel ?? 'out'}
          <span className="nl-port-dot" style={{ borderColor: 'var(--node-orange)', background: 'var(--node-orange)' }} />
        </span>
      </div>
    </div>
  );
}

// ── Connector between nodes ───────────────────────────────────────────────
function Connector() {
  return (
    <div className="nl-connector">
      <div className="nl-connector-dot" />
    </div>
  );
}

// ── Topbar metric ─────────────────────────────────────────────────────────
function TMetric({ label, value, color }) {
  return (
    <div className="nl-tmetric">
      <span className="k">{label}</span>
      <span className="v" style={color ? { color } : undefined}>{value}</span>
    </div>
  );
}

// ── Canvas panel ──────────────────────────────────────────────────────────
function CanvasPanel({ icon: Icon, title, right, children }) {
  return (
    <div className="nl-canvas-panel">
      <div className="nl-canvas-panel-hd">
        {Icon ? <Icon size={11} /> : null}
        <span>{title}</span>
        {right ? <div style={{ marginLeft: 'auto' }}>{right}</div> : null}
      </div>
      <div className="nl-canvas-panel-body">{children}</div>
    </div>
  );
}

// ── Main app ──────────────────────────────────────────────────────────────
export default function NeuralabApp() {
  const [datasetName, setDatasetName] = useState('spiral');
  const [noise, setNoise] = useState(0.1);
  const [numSamples, setNumSamples] = useState(200);
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [useFourier, setUseFourier] = useState(false);
  const [numBands, setNumBands] = useState(4);
  const [optimizer, setOptimizer] = useState('adam');
  const [baseLr, setBaseLr] = useState(0.03);
  const [weightDecay, setWeightDecay] = useState(0);
  const [scheduler, setScheduler] = useState('const');
  const [speed, setSpeed] = useState(1);
  const [batchSize, setBatchSize] = useState(32);
  const [isRunning, setIsRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [lossHistory, setLossHistory] = useState([]);
  const [metrics, setMetrics] = useState({ loss: 0, testLoss: 0, acc: 0, testAcc: 0, gradNorm: 0, deadFrac: 0 });
  const [tick, setTick] = useState(0);
  const [showCode, setShowCode] = useState(false);

  const networkRef = useRef(null);
  const trainDataRef = useRef(null);
  const testDataRef = useRef(null);
  const runningRef = useRef(false);
  const stepRef = useRef(0);
  const rafRef = useRef(null);
  const lossBufferRef = useRef([]);
  const configRef = useRef(config);
  configRef.current = config;
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const dragIndexRef = useRef(null);

  const datasetMeta = DATASETS[datasetName];
  const task = datasetMeta.task;
  const numClasses = task === 'multiclass' ? 3 : 2;
  const outputDim = task === 'multiclass' ? 3 : 1;
  const optimizerMeta = OPTIMIZER_OPTIONS.find(o => o.value === optimizer);

  const selectedNode = useMemo(() => {
    if (!selectedNodeId) return null;
    if (selectedNodeId === 'input') {
      return {
        nodeType: 'input',
        title: 'Input',
        subtitle: useFourier ? `Fourier ×${numBands}` : 'x₀ x₁',
        config: { useFourier, numBands },
      };
    }
    if (selectedNodeId === 'output') {
      return {
        nodeType: 'output',
        title: 'Output',
        subtitle: task === 'binary' ? 'BCE / Sigmoid' : task === 'multiclass' ? 'CE / Softmax' : 'MSE / Linear',
      };
    }
    if (selectedNodeId.startsWith('layer-')) {
      const index = Number(selectedNodeId.split('-')[1]);
      const layer = config[index];
      if (!layer) return null;
      const title = layer.type === 'dense'
        ? `Dense · ${layer.units}u`
        : layer.type === 'dropout'
          ? `Dropout · ${layer.rate.toFixed(2)}`
          : layer.type === 'layernorm'
            ? 'LayerNorm'
            : layer.type === 'batchnorm'
              ? 'BatchNorm'
              : layer.type === 'rmsnorm'
                ? 'RMSNorm'
                : layer.type;
      const subtitle = layer.type === 'dense'
        ? `${layer.activation}${layer.residual ? ' • skip' : ''}`
        : layer.type === 'dropout'
          ? `rate ${layer.rate.toFixed(2)}`
          : 'normalization';
      return { nodeType: layer.type, title, subtitle, layerIndex: index, config: layer };
    }
    return null;
  }, [selectedNodeId, config, useFourier, numBands, outputDim, task]);

  // ── Data ────────────────────────────────────────────────────────────────
  const regenerateData = useCallback(() => {
    const all = datasetMeta.fn(numSamples, noise);
    const idx = all.X.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    const nTrain = Math.floor(idx.length * 0.75);
    trainDataRef.current = { X: idx.slice(0, nTrain).map(i => all.X[i]), y: idx.slice(0, nTrain).map(i => all.y[i]), classes: all.classes };
    testDataRef.current  = { X: idx.slice(nTrain).map(i => all.X[i]),  y: idx.slice(nTrain).map(i => all.y[i]),  classes: all.classes };
    setTick(v => v + 1);
  }, [datasetMeta, noise, numSamples]);

  useEffect(() => { regenerateData(); }, [regenerateData]);

  // ── Network ──────────────────────────────────────────────────────────────
  const rebuildNetwork = useCallback(() => {
    networkRef.current = createNetwork(configRef.current, 2, outputDim, useFourier, numBands);
    stepRef.current = 0;
    setStep(0);
    setLossHistory([]);
    lossBufferRef.current = [];
    networkRef.current.forward(makeGrid(), false);
    setTick(v => v + 1);
  }, [numBands, outputDim, useFourier]);

  useEffect(() => { rebuildNetwork(); }, [config, rebuildNetwork]);

  // ── LR schedule ─────────────────────────────────────────────────────────
  const getCurrentLr = useCallback(s => {
    if (scheduler === 'const') return baseLr;
    const p = Math.min(1, s / 2000);
    if (scheduler === 'cosine') return baseLr * 0.5 * (1 + Math.cos(Math.PI * p));
    if (scheduler === 'step') return baseLr * Math.pow(0.5, Math.floor(p * 3));
    if (scheduler === 'warmup_cosine') {
      if (p < 0.05) return baseLr * (p / 0.05);
      const q = (p - 0.05) / 0.95;
      return baseLr * 0.5 * (1 + Math.cos(Math.PI * q));
    }
    return baseLr;
  }, [baseLr, scheduler]);

  // ── Training ─────────────────────────────────────────────────────────────
  const trainOneStep = useCallback(() => {
    const net = networkRef.current, data = trainDataRef.current;
    if (!net || !data) return null;
    const B = Math.min(batchSize, data.X.length);
    const si = Array.from({ length: B }, () => Math.floor(Math.random() * data.X.length));
    net.zeroGrad();
    const out = net.forward(si.map(i => data.X[i]), true);
    const yb = si.map(i => data.y[i]);
    const lossInfo = task === 'binary' ? bceLoss(out, yb) : task === 'multiclass' ? softmaxCELoss(out, yb, numClasses) : mseLoss(out, yb);
    net.backward(lossInfo.dOut);
    net.step(optimizer, getCurrentLr(stepRef.current), weightDecay);
    stepRef.current += 1;
    return lossInfo.loss;
  }, [batchSize, getCurrentLr, numClasses, optimizer, task, weightDecay]);

  const evaluate = useCallback(() => {
    const net = networkRef.current;
    const train = trainDataRef.current, test = testDataRef.current;
    if (!net || !train || !test) return null;
    const evalSet = data => {
      const out = net.forward(data.X, false);
      let loss = 0, correct = 0;
      if (task === 'binary') {
        for (let i = 0; i < data.X.length; i++) {
          const p = Math.max(1e-7, Math.min(1 - 1e-7, 1 / (1 + Math.exp(-out[i][0]))));
          loss += -(data.y[i] * Math.log(p) + (1 - data.y[i]) * Math.log(1 - p));
          if ((p > 0.5 ? 1 : 0) === data.y[i]) correct++;
        }
      } else if (task === 'multiclass') {
        for (let i = 0; i < data.X.length; i++) {
          let mx = -Infinity, am = 0;
          for (let k = 0; k < numClasses; k++) if (out[i][k] > mx) { mx = out[i][k]; am = k; }
          let sum = 0;
          for (let k = 0; k < numClasses; k++) sum += Math.exp(out[i][k] - mx);
          loss += -Math.log(Math.max(1e-7, Math.exp(out[i][data.y[i]] - mx) / sum));
          if (am === data.y[i]) correct++;
        }
      } else {
        for (let i = 0; i < data.X.length; i++) loss += (out[i][0] - data.y[i]) ** 2;
      }
      return { loss: loss / data.X.length, acc: task === 'regression' ? 0 : correct / data.X.length };
    };
    const tr = evalSet(train), te = evalSet(test);
    net.forward(makeGrid(), false);
    let dead = 0, total = 0;
    for (let li = 0; li < net.layers.length - 1; li++) {
      const L = net.layers[li];
      if (L.type !== 'dense' || !net.activations[li + 1]) continue;
      const acts = net.activations[li + 1];
      for (let j = 0; j < L.dout; j++) {
        let maxAbs = 0;
        for (let i = 0; i < acts.length; i++) maxAbs = Math.max(maxAbs, Math.abs(acts[i][j]));
        if (maxAbs < 1e-3) dead++;
        total++;
      }
    }
    let gradNorm = 0;
    for (const L of net.layers) gradNorm += L.gradNorm() ** 2;
    return { loss: tr.loss, testLoss: te.loss, acc: tr.acc, testAcc: te.acc, gradNorm: Math.sqrt(gradNorm), deadFrac: total > 0 ? dead / total : 0 };
  }, [numClasses, task]);

  // ── Animation loop ───────────────────────────────────────────────────────
  useEffect(() => {
    runningRef.current = isRunning;
    if (!isRunning) return;
    const frame = () => {
      if (!runningRef.current) return;
      for (let i = 0; i < speed; i++) {
        const loss = trainOneStep();
        if (loss !== null) lossBufferRef.current.push({ step: stepRef.current, loss });
      }
      const result = evaluate();
      if (result) {
        setMetrics(result);
        setStep(stepRef.current);
        if (lossBufferRef.current.length > 0) {
          const recent = lossBufferRef.current[lossBufferRef.current.length - 1];
          lossBufferRef.current = [];
          setLossHistory(prev => {
            const next = [...prev, { step: recent.step, train: result.loss, test: result.testLoss }];
            return next.length > 200 ? next.slice(-200) : next;
          });
        }
        setTick(v => v + 1);
      }
      rafRef.current = requestAnimationFrame(frame);
    };
    rafRef.current = requestAnimationFrame(frame);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [evaluate, isRunning, speed, trainOneStep]);

  // ── Handlers ─────────────────────────────────────────────────────────────
  const handlePlayPause = () => setIsRunning(v => !v);
  const handleReset = () => {
    setIsRunning(false); runningRef.current = false;
    rebuildNetwork();
    setMetrics({ loss: 0, testLoss: 0, acc: 0, testAcc: 0, gradNorm: 0, deadFrac: 0 });
  };
  const handleStep = () => {
    trainOneStep();
    const r = evaluate();
    if (r) { setMetrics(r); setStep(stepRef.current); setTick(v => v + 1); }
  };

  useEffect(() => {
    const h = e => {
      if (['INPUT','SELECT','TEXTAREA'].includes(e.target?.tagName)) return;
      if (e.key === ' ') { e.preventDefault(); handlePlayPause(); }
      if (e.key === 'r') handleReset();
      if (e.key === 's') handleStep();
      if (e.key === 'd') regenerateData();
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  });

  // ── Layer CRUD ────────────────────────────────────────────────────────────
  const addLayer = type => {
    const layer = type === 'dense' ? { type: 'dense', units: 8, activation: 'relu', residual: false }
      : type === 'dropout' ? { type: 'dropout', rate: 0.1 } : { type };
    setConfig(c => [...c, layer]);
  };
  const updateLayer = (i, patch) => setConfig(c => c.map((l, j) => j === i ? { ...l, ...patch } : l));
  const removeLayer = i => setConfig(c => c.filter((_, j) => j !== i));
  const moveLayer = (i, dir) => setConfig(c => {
    const a = [...c], j = i + dir;
    if (j < 0 || j >= a.length) return a;
    [a[i], a[j]] = [a[j], a[i]]; return a;
  });

  // ── Derived ───────────────────────────────────────────────────────────────
  const allLayers = networkRef.current?.layers ?? [];
  const hiddenLayers = allLayers.slice(0, -1).filter(L => L?.type);
  const gradNorms   = hiddenLayers.map(L => L.gradNorm());
  const weightNorms = hiddenLayers.map(L => L.weightNorm());
  const totalParams = allLayers.reduce((s, L) => L.type === 'dense' ? s + L.din * L.dout + L.dout : s, 0);
  const pyCode = useMemo(
    () => exportPyTorch(config, 2, task, numClasses, useFourier, numBands, optimizer, baseLr, weightDecay),
    [baseLr, config, numBands, numClasses, optimizer, task, useFourier, weightDecay],
  );

  return (
    <div className="nl-root">
      <style dangerouslySetInnerHTML={{ __html: STYLES }} />
      <div className="nl-canvas-bg" />

      {/* ── TOPBAR ──────────────────────────────────────────────────────── */}
      <header className="nl-topbar">
        <div className="nl-topbar-brand">
          <div className="nl-brand-icon">
            <Zap size={15} color="#fff" strokeWidth={2.5} />
          </div>
          <div className="nl-brand-text">
            <span className="nl-title">Neuralab</span>
            <span className="nl-subtitle">flow-block architecture</span>
          </div>
        </div>

        <div className="nl-topbar-controls">
          <button className={`nl-btn${isRunning ? '' : ' primary'}`} onClick={handlePlayPause} style={{ minWidth: 86 }}>
            {isRunning ? <><Pause size={13} /> Pause</> : <><Play size={13} /> Run</>}
          </button>
          <button className="nl-btn icon" onClick={handleStep} disabled={isRunning} title="Single step (s)">
            <StepForward size={13} />
          </button>
          <button className="nl-btn icon" onClick={handleReset} title="Reset weights (r)">
            <RotateCcw size={13} />
          </button>
          <button className="nl-btn icon" onClick={regenerateData} title="New data (d)">
            <Shuffle size={13} />
          </button>
          <div style={{ width: 1, height: 20, background: 'var(--bd)', margin: '0 4px' }} />
          <div className={`nl-chip${isRunning ? ' running' : ''}`}>
            {isRunning ? '● live' : '○ idle'}
          </div>
          <div className="nl-chip blue mono">{totalParams.toLocaleString()} params</div>
        </div>

        <div className="nl-topbar-metrics">
          <TMetric label="Step" value={step.toLocaleString()} />
          <TMetric label="LR" value={getCurrentLr(step).toExponential(1)} color="var(--node-teal)" />
          <TMetric label={task === 'regression' ? 'Train MSE' : 'Train Loss'} value={metrics.loss.toFixed(3)} />
          <TMetric label={task === 'regression' ? 'Test MSE' : 'Test Loss'} value={metrics.testLoss.toFixed(3)}
            color={metrics.testLoss > metrics.loss + 0.15 ? 'var(--amber)' : undefined} />
          {task !== 'regression' ? <>
            <TMetric label="Train Acc" value={`${(metrics.acc * 100).toFixed(1)}%`}
              color={metrics.acc > 0.85 ? 'var(--green)' : undefined} />
            <TMetric label="Test Acc" value={`${(metrics.testAcc * 100).toFixed(1)}%`}
              color={metrics.testAcc > 0.85 ? 'var(--green)' : undefined} />
          </> : <>
            <TMetric label="‖∇θ‖" value={metrics.gradNorm.toExponential(1)} />
            <TMetric label="Dead %" value={`${(metrics.deadFrac * 100).toFixed(0)}%`}
              color={metrics.deadFrac > 0.3 ? 'var(--rose)' : metrics.deadFrac > 0.1 ? 'var(--amber)' : 'var(--green)'} />
          </>}
          <div style={{ marginLeft: 'auto', padding: '0 12px', display: 'flex', alignItems: 'center', borderLeft: '1px solid var(--bd)' }}>
            <button className="nl-btn" onClick={() => setShowCode(true)}>
              <Code2 size={13} /> Export
            </button>
          </div>
        </div>
      </header>

      {/* ── LEFT SIDEBAR: Architecture flow ─────────────────────────────── */}
      <aside className="nl-sidebar">
        <div className="nl-sidebar-scroll nl-scroll">
          <Section icon={Layers} title="Architecture" badge={`${config.length + 2} blocks`}>
            <div className="nl-arch-stack">
              {/* Input node */}
              <div
                className={`nl-arch-block nl-arch-block--input${selectedNodeId === 'input' ? ' selected' : ''}`}
                onClick={() => setSelectedNodeId(id => id === 'input' ? null : 'input')}
              >
                <div className="nl-arch-block-dot" style={{ background: 'var(--node-teal)' }} />
                <div className="nl-arch-block-body">
                  <span className="nl-arch-block-title">Input</span>
                  <span className="nl-arch-block-meta">{useFourier ? `Fourier ×${numBands}` : 'x₀ · x₁'}</span>
                </div>
                <span className="nl-arch-block-tag">in</span>
              </div>

              {/* Hidden layers */}
              {config.map((layer, i) => {
                const id = `layer-${i}`;
                const title = layer.type === 'dense' ? `Dense · ${layer.units}u`
                  : layer.type === 'dropout' ? `Dropout`
                  : layer.type === 'layernorm' ? 'LayerNorm'
                  : layer.type === 'batchnorm' ? 'BatchNorm'
                  : layer.type === 'rmsnorm' ? 'RMSNorm'
                  : layer.type;
                const meta = layer.type === 'dense'
                  ? `${layer.activation}${layer.residual ? ' · skip' : ''}`
                  : layer.type === 'dropout' ? `rate ${layer.rate.toFixed(2)}`
                  : 'norm';
                const color = NODE_COLORS[layer.type] ?? 'var(--node-blue)';
                return (
                  <React.Fragment key={id}>
                    <div className="nl-arch-connector">
                      <div className="nl-arch-connector-line" />
                      <div className="nl-arch-connector-arrow" />
                    </div>
                    <div
                      className={`nl-arch-block${selectedNodeId === id ? ' selected' : ''}`}
                      onClick={() => setSelectedNodeId(cur => cur === id ? null : id)}
                      draggable
                      onDragStart={() => { dragIndexRef.current = i; }}
                      onDragOver={e => e.preventDefault()}
                      onDrop={() => {
                        const from = dragIndexRef.current;
                        if (from != null && from !== i) { moveLayer(from, i - from); dragIndexRef.current = null; }
                      }}
                    >
                      <div className="nl-arch-block-dot" style={{ background: color }} />
                      <div className="nl-arch-block-body">
                        <span className="nl-arch-block-title">{title}</span>
                        <span className="nl-arch-block-meta">{meta}</span>
                      </div>
                      <div className="nl-arch-block-actions">
                        <button className="nl-arch-action" title="Move up" onClick={e => { e.stopPropagation(); moveLayer(i, -1); }}
                          disabled={i === 0}>▲</button>
                        <button className="nl-arch-action" title="Move down" onClick={e => { e.stopPropagation(); moveLayer(i, 1); }}
                          disabled={i === config.length - 1}>▼</button>
                        <button className="nl-arch-action danger" title="Remove" onClick={e => { e.stopPropagation(); removeLayer(i); setSelectedNodeId(null); }}>×</button>
                      </div>
                    </div>
                  </React.Fragment>
                );
              })}

              {/* Connector to output */}
              <div className="nl-arch-connector">
                <div className="nl-arch-connector-line" />
                <div className="nl-arch-connector-arrow" />
              </div>

              {/* Output node */}
              <div
                className={`nl-arch-block nl-arch-block--output${selectedNodeId === 'output' ? ' selected' : ''}`}
                onClick={() => setSelectedNodeId(id => id === 'output' ? null : 'output')}
              >
                <div className="nl-arch-block-dot" style={{ background: 'var(--node-orange)' }} />
                <div className="nl-arch-block-body">
                  <span className="nl-arch-block-title">Output</span>
                  <span className="nl-arch-block-meta">
                    {task === 'binary' ? 'BCE · sigmoid' : task === 'multiclass' ? 'CE · softmax' : 'MSE · linear'} · {outputDim}d
                  </span>
                </div>
                <span className="nl-arch-block-tag">out</span>
              </div>
            </div>

            <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--fg-4)', marginBottom: 2 }}>
                Add block
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
                {[
                  ['dense', 'Dense', 'var(--node-blue)'],
                  ['dropout', 'Dropout', 'var(--node-violet)'],
                  ['layernorm', 'LayerNorm', 'var(--node-amber)'],
                  ['batchnorm', 'BatchNorm', 'var(--node-amber)'],
                  ['rmsnorm', 'RMSNorm', 'var(--node-amber)'],
                ].map(([type, label, color]) => (
                  <button key={type} className="nl-btn" onClick={() => addLayer(type)}
                    style={{ justifyContent: 'flex-start', gap: 6, fontSize: 11 }}>
                    <div style={{ width: 6, height: 6, borderRadius: '50%', background: color, flexShrink: 0 }} />
                    <Plus size={10} /> {label}
                  </button>
                ))}
              </div>
            </div>
          </Section>

          <Section icon={Database} title="Dataset" badge={task} defaultOpen={true}>
            <div className="nl-dataset-grid" style={{ marginBottom: 10 }}>
              {Object.keys(DATASETS).map(key => (
                <button key={key} className={`nl-dataset-btn ${datasetName === key ? 'active' : ''}`}
                  onClick={() => setDatasetName(key)}>
                  <DatasetThumb type={key} size={30} />
                  <span className="nl-dataset-name">{DATASETS[key].name}</span>
                </button>
              ))}
            </div>
            <div className="nl-stack" style={{ gap: 8 }}>
              <div>
                <div className="nl-row between" style={{ marginBottom: 4 }}>
                  <span className="nl-label">Noise</span>
                  <span className="nl-value-display">{noise.toFixed(2)}</span>
                </div>
                <input className="nl-slider" type="range" min="0" max="0.5" step="0.01" value={noise}
                  onChange={e => setNoise(+e.target.value)} />
              </div>
              <div>
                <div className="nl-row between" style={{ marginBottom: 4 }}>
                  <span className="nl-label">Samples</span>
                  <span className="nl-value-display">{numSamples}</span>
                </div>
                <input className="nl-slider" type="range" min="40" max="400" step="20" value={numSamples}
                  onChange={e => setNumSamples(+e.target.value)} />
              </div>
            </div>
          </Section>

          <Section icon={Eye} title="Shortcuts" defaultOpen={false}>
            <div className="nl-stack" style={{ gap: 6 }}>
              {[['space','play / pause'],['s','single step'],['r','reset weights'],['d','resample data']].map(([k,v]) => (
                <div key={k} className="nl-row between">
                  <span className="nl-kbd">{k}</span>
                  <span style={{ fontSize: 11, color: 'var(--fg-3)' }}>{v}</span>
                </div>
              ))}
            </div>
          </Section>
        </div>
      </aside>

      {/* ── RIGHT SIDEBAR: Controls ──────────────────────────────────────── */}
      <aside className="nl-sidebar-right">
        <div className="nl-sidebar-scroll nl-scroll">
          <Section icon={Cpu} title="Optimizer" defaultOpen={true}>
            <div className="nl-stack">
              <div>
                <span className="nl-label" style={{ display: 'block', marginBottom: 6 }}>Algorithm</span>
                <div className="nl-opt-pills">
                  {OPTIMIZER_OPTIONS.map(o => (
                    <button key={o.value} className={`nl-opt-pill${optimizer === o.value ? ' active' : ''}`}
                      onClick={() => setOptimizer(o.value)}>
                      {o.value}
                    </button>
                  ))}
                </div>
                {optimizerMeta?.detail && (
                  <div style={{ marginTop: 6, fontSize: 10, color: 'var(--fg-4)', lineHeight: 1.5 }}>
                    {optimizerMeta.detail}
                  </div>
                )}
              </div>

              <div>
                <div className="nl-row between" style={{ marginBottom: 4 }}>
                  <span className="nl-label">Learning Rate</span>
                  <span className="nl-value-display" style={{ color: 'var(--node-teal)' }}>{baseLr.toFixed(4)}</span>
                </div>
                <input className="nl-slider" type="range" min="-5" max="0" step="0.1"
                  value={Math.log10(baseLr)} onChange={e => setBaseLr(10 ** +e.target.value)} />
                <div className="nl-row between" style={{ marginTop: 2 }}>
                  <span style={{ fontFamily: 'JetBrains Mono', fontSize: 8, color: 'var(--fg-4)' }}>1e-5</span>
                  <span style={{ fontFamily: 'JetBrains Mono', fontSize: 8, color: 'var(--fg-4)' }}>1e0</span>
                </div>
              </div>

              <div>
                <span className="nl-label" style={{ display: 'block', marginBottom: 5 }}>Schedule</span>
                <select className="nl-select mono" value={scheduler} onChange={e => setScheduler(e.target.value)}>
                  {SCHEDULER_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>

              <div>
                <div className="nl-row between" style={{ marginBottom: 4 }}>
                  <span className="nl-label">Weight Decay</span>
                  <span className="nl-value-display">{weightDecay.toExponential(1)}</span>
                </div>
                <input className="nl-slider" type="range" min="-6" max="-1" step="0.2"
                  value={weightDecay === 0 ? -6 : Math.log10(weightDecay)}
                  onChange={e => setWeightDecay(+e.target.value === -6 ? 0 : 10 ** +e.target.value)} />
              </div>

              <div>
                <div className="nl-row between" style={{ marginBottom: 4 }}>
                  <span className="nl-label">Batch Size</span>
                  <span className="nl-value-display">{batchSize}</span>
                </div>
                <input className="nl-slider" type="range" min="8" max="128" step="8"
                  value={batchSize} onChange={e => setBatchSize(+e.target.value)} />
              </div>

              <div>
                <div className="nl-row between" style={{ marginBottom: 4 }}>
                  <span className="nl-label">Steps / Frame</span>
                  <span className="nl-value-display">{speed}×</span>
                </div>
                <input className="nl-slider" type="range" min="1" max="20"
                  value={speed} onChange={e => setSpeed(+e.target.value)} />
              </div>
            </div>
          </Section>

          <Section icon={Target} title="Selected block" defaultOpen={true}>
            {selectedNode ? (
              <div className="nl-stack" style={{ gap: 10 }}>
                <div className="nl-row between">
                  <span className="nl-label">Block</span>
                  <span className="nl-value-display">{selectedNode.title}</span>
                </div>
                <div className="nl-copy">{selectedNode.subtitle}</div>
                {selectedNode.nodeType === 'input' ? (
                  <div className="nl-stack">
                    <label className="nl-check">
                      <input type="checkbox" checked={useFourier} onChange={e => setUseFourier(e.target.checked)} />
                      Fourier projection
                    </label>
                    {useFourier && (
                      <div>
                        <div className="nl-row between" style={{ marginBottom: 4 }}>
                          <span className="nl-label">Bands</span>
                          <span className="nl-value-display">{numBands}</span>
                        </div>
                        <input className="nl-slider" type="range" min="1" max="8" value={numBands}
                          onChange={e => setNumBands(+e.target.value)} />
                      </div>
                    )}
                  </div>
                ) : selectedNode.nodeType === 'output' ? (
                  <div className="nl-copy">Output block — head type follows the dataset task. Add layers above to grow the network.</div>
                ) : (
                  <div className="nl-stack">
                    {selectedNode.nodeType === 'dense' && (
                      <>
                        <div className="nl-row between" style={{ marginBottom: 4 }}>
                          <span className="nl-label">Units</span>
                          <span className="nl-value-display">{selectedNode.config.units}</span>
                        </div>
                        <input className="nl-slider" type="range" min="1" max="32" value={selectedNode.config.units}
                          onChange={e => updateLayer(selectedNode.layerIndex, { units: +e.target.value })} />
                        <select className="nl-select mono" value={selectedNode.config.activation}
                          onChange={e => updateLayer(selectedNode.layerIndex, { activation: e.target.value })}>
                          {ACT_NAMES.map(a => <option key={a} value={a}>{a}</option>)}
                        </select>
                        <label className="nl-check">
                          <input type="checkbox" checked={selectedNode.config.residual}
                            onChange={e => updateLayer(selectedNode.layerIndex, { residual: e.target.checked })} />
                          Residual skip
                        </label>
                      </>
                    )}
                    {selectedNode.nodeType === 'dropout' && (
                      <>
                        <div className="nl-row between" style={{ marginBottom: 4 }}>
                          <span className="nl-label">Rate</span>
                          <span className="nl-value-display">{selectedNode.config.rate.toFixed(2)}</span>
                        </div>
                        <input className="nl-slider" type="range" min="0" max="0.8" step="0.05"
                          value={selectedNode.config.rate}
                          onChange={e => updateLayer(selectedNode.layerIndex, { rate: +e.target.value })} />
                      </>
                    )}
                    {['layernorm', 'batchnorm', 'rmsnorm'].includes(selectedNode.nodeType) && (
                      <div className="nl-copy">Normalization block. Reorder or remove to adjust model topology.</div>
                    )}
                    <button className="nl-btn danger" onClick={() => {
                      if (selectedNode.layerIndex != null) {
                        removeLayer(selectedNode.layerIndex);
                        setSelectedNodeId(null);
                      }
                    }}>
                      Remove block
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="nl-copy">Click a block in the flow to edit its settings.</div>
            )}
          </Section>

          <Section icon={Target} title="Diagnostics" defaultOpen={true}>
            <div className="nl-diag-row">
              <span className="nl-diag-key">Optimizer</span>
              <span className="nl-diag-val">{optimizer}</span>
            </div>
            <div className="nl-diag-row">
              <span className="nl-diag-key">Parameters</span>
              <span className="nl-diag-val">{totalParams.toLocaleString()}</span>
            </div>
            <div className="nl-diag-row">
              <span className="nl-diag-key">Optimizer t</span>
              <span className="nl-diag-val">{networkRef.current?.t ?? 0}</span>
            </div>
            <div className="nl-diag-row">
              <span className="nl-diag-key">‖∇θ‖</span>
              <span className="nl-diag-val">{metrics.gradNorm.toExponential(2)}</span>
            </div>
            <div className="nl-diag-row">
              <span className="nl-diag-key">Dead neurons</span>
              <span className="nl-diag-val" style={{ color: metrics.deadFrac > 0.3 ? 'var(--rose)' : metrics.deadFrac > 0.1 ? 'var(--amber)' : 'var(--green)' }}>
                {(metrics.deadFrac * 100).toFixed(0)}%
                {metrics.deadFrac > 0.3 ? <AlertTriangle size={10} style={{ display: 'inline', marginLeft: 4 }} /> : null}
              </span>
            </div>
            <div className="nl-diag-row">
              <span className="nl-diag-key">Train/test gap</span>
              <span className="nl-diag-val" style={{ color: Math.abs(metrics.testLoss - metrics.loss) > 0.2 ? 'var(--amber)' : 'var(--green)' }}>
                {(metrics.testLoss - metrics.loss).toFixed(3)}
              </span>
            </div>
          </Section>

          <Section icon={Sparkles} title="Task" defaultOpen={false}>
            <div className="nl-diag-row">
              <span className="nl-diag-key">Dataset</span>
              <span className="nl-diag-val">{datasetMeta.name}</span>
            </div>
            <div className="nl-diag-row">
              <span className="nl-diag-key">Task</span>
              <span className="nl-diag-val" style={{ color: 'var(--node-teal)' }}>{task}</span>
            </div>
            <div className="nl-diag-row">
              <span className="nl-diag-key">Loss fn</span>
              <span className="nl-diag-val">{task === 'binary' ? 'BCE' : task === 'multiclass' ? 'CE' : 'MSE'}</span>
            </div>
            <div className="nl-diag-row">
              <span className="nl-diag-key">Output</span>
              <span className="nl-diag-val">{task === 'binary' ? 'sigmoid' : task === 'multiclass' ? 'softmax' : 'linear'}</span>
            </div>
            <div style={{ marginTop: 10, padding: '8px 10px', background: 'var(--chrome-3)', borderRadius: 6, fontSize: 10, color: 'var(--fg-4)', lineHeight: 1.55, borderLeft: '2px solid var(--node-blue)' }}>
              All backprop and optimizer steps run in-browser. No autograd library — hand-written gradients.
            </div>
          </Section>
        </div>
      </aside>

      {/* ── CANVAS CENTER ───────────────────────────────────────────────── */}
      <main className="nl-canvas nl-scroll" style={{ overflowY: 'auto', paddingBottom: 36 }}>

        {/* Top row: decision surface + neuron fields */}
        <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 10, flexShrink: 0 }}>

          {/* Decision boundary panel */}
          <CanvasPanel icon={Activity} title={task === 'regression' ? 'Prediction Line' : 'Prediction Surface'}
            right={
              <div className="nl-legend">
                {task === 'binary' && <>
                  <span><span className="nl-legend-dot" style={{ background: 'var(--node-teal)' }} />class 0</span>
                  <span><span className="nl-legend-dot" style={{ background: 'var(--node-orange)' }} />class 1</span>
                </>}
                {task === 'multiclass' && <>
                  {[['var(--node-teal)','c0'],['var(--amber)','c1'],['var(--green)','c2']].map(([c,l]) => (
                    <span key={l}><span className="nl-legend-dot" style={{ background: c }} />{l}</span>
                  ))}
                </>}
                {task === 'regression' && <span style={{ color: 'var(--fg-4)' }}>blue → orange = −1 → +1</span>}
              </div>
            }
          >
            <DecisionBoundary network={networkRef.current} dataset={trainDataRef.current}
              task={task} numClasses={numClasses} tick={tick} />
            <div className="mono" style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 9, color: 'var(--fg-4)' }}>
              <span>-2.25</span><span>{task === 'regression' ? 'x₀ → y' : 'x₀ · x₁'}</span><span>2.25</span>
            </div>
          </CanvasPanel>

          {/* Neuron activation fields */}
          <CanvasPanel icon={Eye} title="Neuron Fields">
            <div className="nl-scroll" style={{ overflowX: 'auto' }}>
              <div style={{ display: 'flex', gap: 14, minWidth: 'max-content' }}>
                {allLayers.map((L, li) => {
                  if (L.type !== 'dense') return null;
                  const isOutput = li === allLayers.length - 1;
                  return (
                    <div key={li} style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                      <div className="mono" style={{
                        fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.06em',
                        color: isOutput ? 'var(--node-orange)' : 'var(--node-teal)',
                      }}>
                        {isOutput ? 'output' : `L${li + 1}`} · {L.dout}u
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: L.dout > 8 ? '1fr 1fr' : '1fr', gap: 2 }}>
                        {Array.from({ length: L.dout }).map((_, ni) => (
                          <NeuronMap key={ni} network={networkRef.current} layerIdx={li} neuronIdx={ni} tick={tick} />
                        ))}
                      </div>
                      {!isOutput && (
                        <div className="mono" style={{ fontSize: 8, color: 'var(--fg-4)', textAlign: 'center' }}>{L.act}</div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </CanvasPanel>
        </div>

        {/* Bottom row: loss curve + gradient flow */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 10, flexShrink: 0 }}>

          {/* Loss curve */}
          <CanvasPanel icon={Activity} title="Loss Curve"
            right={
              <div style={{ display: 'flex', gap: 12, fontSize: 9, fontFamily: 'JetBrains Mono', color: 'var(--fg-4)' }}>
                <span style={{ color: 'var(--node-orange)' }}>— train</span>
                <span style={{ color: 'var(--node-teal)' }}>- - test</span>
              </div>
            }
          >
            <div style={{ height: 180 }}>
              {lossHistory.length > 1 ? (
                <ResponsiveContainer>
                  <LineChart data={lossHistory} margin={{ top: 4, right: 8, left: -14, bottom: 0 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="4 4" />
                    <XAxis dataKey="step" stroke="var(--fg-4)" tick={{ fontSize: 9, fontFamily: 'JetBrains Mono' }} />
                    <YAxis stroke="var(--fg-4)" tick={{ fontSize: 9, fontFamily: 'JetBrains Mono' }} domain={['auto', 'auto']} />
                    <Tooltip
                      contentStyle={{ background: '#111', border: '1px solid var(--bd-hi)', borderRadius: 6, fontSize: 10, fontFamily: 'JetBrains Mono' }}
                      labelStyle={{ color: 'var(--fg-2)' }}
                    />
                    <Line type="monotone" dataKey="train" stroke="var(--node-orange)" dot={false} strokeWidth={1.5} />
                    <Line type="monotone" dataKey="test" stroke="var(--node-teal)" dot={false} strokeWidth={1.5} strokeDasharray="5 3" />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ height: '100%', display: 'grid', placeItems: 'center' }}>
                  <div style={{ fontSize: 11, color: 'var(--fg-4)', textAlign: 'center' }}>
                    Press <span className="nl-kbd">space</span> to start training
                  </div>
                </div>
              )}
            </div>
          </CanvasPanel>

          {/* Gradient flow */}
          <CanvasPanel icon={Gauge} title="Gradient Flow">
            {hiddenLayers.length > 0 ? (
              <div>
                <div style={{ display: 'flex', marginBottom: 8, fontSize: 8, fontFamily: 'JetBrains Mono', color: 'var(--fg-4)', gap: 12 }}>
                  <span style={{ width: 48 }}>layer</span>
                  <span style={{ flex: 1 }}><span style={{ color: 'var(--node-orange)' }}>grad</span> / <span style={{ color: 'var(--node-blue)' }}>weight</span></span>
                  <span style={{ width: 50, textAlign: 'right' }}>‖g‖</span>
                </div>
                <GradFlowBars gradNorms={gradNorms} weightNorms={weightNorms} layers={hiddenLayers} />
                <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                  {[
                    ['‖∇θ‖', metrics.gradNorm.toExponential(2), undefined],
                    ['Dead', `${(metrics.deadFrac * 100).toFixed(0)}%`, metrics.deadFrac > 0.3 ? 'var(--rose)' : metrics.deadFrac > 0.1 ? 'var(--amber)' : 'var(--green)'],
                  ].map(([k, v, c]) => (
                    <div key={k} style={{ background: 'var(--chrome-3)', borderRadius: 6, padding: '7px 10px', border: '1px solid var(--bd)' }}>
                      <div style={{ fontSize: 9, color: 'var(--fg-4)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{k}</div>
                      <div className="mono" style={{ fontSize: 13, fontWeight: 600, color: c ?? 'var(--fg-0)', marginTop: 3 }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ fontSize: 11, color: 'var(--fg-4)', textAlign: 'center', padding: '24px 0' }}>
                Add hidden layers to inspect gradients.
              </div>
            )}
          </CanvasPanel>
        </div>
      </main>

      {/* ── STATUS BAR ──────────────────────────────────────────────────── */}
      <div className="nl-statusbar">
        <span>{DATASETS[datasetName].name}</span>
        <span>·</span>
        <span className="mono">{task}</span>
        <span>·</span>
        <span className="mono">{optimizer}</span>
        <span>·</span>
        <span className="mono">lr={getCurrentLr(step).toExponential(1)}</span>
        <span>·</span>
        <span className="mono">step {step.toLocaleString()}</span>
        {isRunning && <><span>·</span><span style={{ color: 'var(--green)' }}>● training</span></>}
      </div>

      {/* ── CODE EXPORT MODAL ────────────────────────────────────────────── */}
      {showCode && (
        <div className="nl-modal" onClick={() => setShowCode(false)}>
          <div className="nl-modal-card" onClick={e => e.stopPropagation()}>
            <div className="nl-modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Code2 size={15} color="var(--node-teal)" />
                <span className="nl-modal-title">Export to PyTorch</span>
              </div>
              <div style={{ display: 'flex', gap: 6 }}>
                <button className="nl-btn" onClick={() => navigator.clipboard.writeText(pyCode)}>
                  <Copy size={13} /> Copy
                </button>
                <button className="nl-btn icon" onClick={() => setShowCode(false)}>
                  <X size={13} />
                </button>
              </div>
            </div>
            <pre className="nl-modal-body nl-scroll">{pyCode}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
