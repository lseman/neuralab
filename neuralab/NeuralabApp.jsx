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
import { DatasetThumb, DecisionBoundary, GradFlowBars } from './visuals.jsx';

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

function activationColor(value, min, max) {
  const range = Math.max(1e-6, max - min);
  const t = (value - min) / range;
  const r = Math.round(82 + t * (255 - 82));
  const g = Math.round(199 + t * (109 - 199));
  const b = Math.round(255 + t * (45 - 255));
  return `rgb(${r}, ${g}, ${b})`;
}

function NetworkDiagram({ network, config, updateLayer, useFourier, numBands, outputDim, task, tick }) {
  const denseLayers = useMemo(() => {
    if (!network) return [];
    return network.layers.filter(layer => layer.type === 'dense');
  }, [network]);

  const layers = useMemo(() => {
    const inputCount = Math.max(1, Math.min(network?.inputDim ?? 2, 8));
    const inputLabel = network?.inputDim === 1 ? 'x₀' : 'x₀, x₁';
    const nodes = [{ type: 'input', count: inputCount, label: inputLabel, layerIndex: -1 }];

    denseLayers.forEach((layer, index) => {
      const isOutput = index === denseLayers.length - 1;
      nodes.push({
        type: isOutput ? 'output' : 'dense',
        count: layer.dout,
        label: isOutput ? (task === 'regression' ? 'linear' : 'output') : layer.act,
        weights: layer.W,
        din: layer.din,
        dout: layer.dout,
        neuronIndices: Array.from({ length: layer.dout }, (_, i) => i),
        layerIndex: index,
        isOutput,
      });
    });

    return nodes;
  }, [denseLayers, network, useFourier, numBands, task]);

  const width = Math.max(372, 116 + Math.max(layers.length - 1, 0) * 90);
  const marginX = 36;
  const marginY = 38;
  const cardWidth = 26;
  const cardHeight = 26;
  const cardGap = 10;
  const inputDim = network?.inputDim ?? 2;
  const maxCount = Math.max(1, ...layers.map(layer => Number.isFinite(layer.count) ? layer.count : 1));
  const height = Math.max(228, marginY * 2 + maxCount * cardHeight + Math.max(0, maxCount - 1) * cardGap);

  const columns = useMemo(() => layers.map((layer, li) => {
    const rawCount = Number.isFinite(layer.count) ? layer.count : 1;
    const count = Math.max(1, rawCount);
    const x = marginX + li * ((width - marginX * 2) / Math.max(layers.length - 1, 1));
    const contentHeight = count * cardHeight + Math.max(0, count - 1) * cardGap;
    const topOffset = (height - contentHeight) / 2;
    return {
      layer,
      x,
      nodes: Array.from({ length: count }).map((_, ni) => ({
        cx: x,
        cy: topOffset + cardHeight / 2 + ni * (cardHeight + cardGap),
        actualIndex: layer.type === 'input' ? ni : (layer.neuronIndices?.[ni] ?? ni),
        label: null,
      })),
    };
  }), [cardGap, cardHeight, height, layers, marginX]);

  const edges = useMemo(() => {
    const nextEdges = [];
    for (let li = 0; li < columns.length - 1; li++) {
      const src = columns[li];
      const dst = columns[li + 1];
      const weights = dst.layer.weights;
      let maxWeight = 1;
      if (weights) {
        for (let a = 0; a < Math.min(src.nodes.length, weights.length); a++) {
          for (let b = 0; b < Math.min(dst.nodes.length, weights[0].length); b++) {
            maxWeight = Math.max(maxWeight, Math.abs(weights[a][b]));
          }
        }
      }
      for (let a = 0; a < src.nodes.length; a++) {
        for (let b = 0; b < dst.nodes.length; b++) {
          const weight = weights?.[a]?.[b] ?? 0;
          const strength = weights ? Math.min(1, Math.abs(weight) / maxWeight) : 0.12;
          nextEdges.push({
            from: src.nodes[a],
            to: dst.nodes[b],
            width: 0.7 + 0.9 * strength,
            color: dst.layer.type === 'output'
              ? `rgba(249,115,22,${0.14 + 0.22 * strength})`
              : `rgba(148,163,184,${0.12 + 0.18 * strength})`,
          });
        }
      }
    }
    return nextEdges;
  }, [columns]);

  const previewData = useMemo(() => {
    const previews = {};
    if (!network) return previews;

    const previewRes = inputDim === 1 ? 18 : 5;
    const previewInputs = makeGrid(previewRes, inputDim);
    const out = network.forward(previewInputs, false);
    const denseActivationOffsets = [];
    network.layers.forEach((layer, index) => {
      if (layer.type === 'dense') denseActivationOffsets.push(index + 1);
    });

    columns.forEach((column, columnIndex) => {
      column.nodes.forEach((node, nodeIndex) => {
        let values;
        if (column.layer.type === 'input') {
          values = previewInputs.map(sample => sample[node.actualIndex] ?? 0);
        } else {
          const activationIndex = denseActivationOffsets[columnIndex - 1];
          const activations = column.layer.isOutput ? out : network.activations[activationIndex];
          values = activations?.map(row => row[node.actualIndex] ?? 0) ?? [];
        }

        let min = Infinity;
        let max = -Infinity;
        for (const value of values) {
          if (value < min) min = value;
          if (value > max) max = value;
        }
        if (!values.length) {
          min = 0;
          max = 1;
        }
        previews[`${columnIndex}-${nodeIndex}`] = { values, min, max, previewRes };
      });
    });

    return previews;
  }, [columns, inputDim, network, tick]);

  return (
    <div className="nl-network-diagram">
      <div className="nl-network-svg-wrap">
        <div className="nl-network-stage" style={{ width }}>
          <div className="nl-network-toolbar">
            {columns.map((column, ci) => (
              <div
                key={ci}
                className="nl-network-layer-control"
                style={{ left: column.x, transform: 'translateX(-50%)' }}
              >
                {column.layer.type === 'dense' && !column.layer.isOutput ? (
                  <div className="nl-layer-spin">
                    <button
                      type="button"
                      className="nl-btn mini"
                      onClick={() => updateLayer(column.layer.layerIndex, { units: Math.max(1, column.layer.dout - 1) })}
                    >
                      −
                    </button>
                    <button
                      type="button"
                      className="nl-btn mini"
                      onClick={() => updateLayer(column.layer.layerIndex, { units: column.layer.dout + 1 })}
                    >
                      +
                    </button>
                  </div>
                ) : <div className="nl-layer-spacer" />}
              </div>
            ))}
          </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="nl-network-svg" style={{ width, height }}>
        {edges.map((edge, index) => (
          <path
            key={index}
            d={`M ${edge.from.cx} ${edge.from.cy} C ${edge.from.cx + (edge.to.cx - edge.from.cx) * 0.42} ${edge.from.cy}, ${edge.to.cx - (edge.to.cx - edge.from.cx) * 0.34} ${edge.to.cy}, ${edge.to.cx} ${edge.to.cy}`}
            stroke={edge.color}
            strokeWidth={edge.width}
            strokeLinecap="round"
            fill="none"
          />
        ))}
        {columns.map((column, ci) => (
          <g key={ci}>
            <line
              x1={column.x}
              y1="24"
              x2={column.x}
              y2={height - 24}
              stroke="rgba(255,255,255,0.08)"
              strokeDasharray="3 8"
            />
            {column.nodes.map((node, ni) => (
              <g key={ni}>
                <rect
                  x={node.cx - cardWidth / 2}
                  y={node.cy - cardHeight / 2}
                  width={cardWidth}
                  height={cardHeight}
                  rx="7"
                  fill="rgba(255,255,255,0.04)"
                  stroke={column.layer.type === 'output' ? 'rgba(249,115,22,0.65)' : column.layer.type === 'input' ? 'rgba(20,184,166,0.65)' : 'rgba(59,130,246,0.55)'}
                  strokeWidth="1"
                />
                {inputDim === 1 ? (
                  <>
                    <line
                      x1={node.cx - cardWidth / 2 + 4}
                      y1={node.cy}
                      x2={node.cx + cardWidth / 2 - 4}
                      y2={node.cy}
                      stroke="rgba(255,255,255,0.12)"
                      strokeWidth="1"
                    />
                    <path
                      d={(previewData[`${ci}-${ni}`]?.values ?? []).map((value, index, arr) => {
                        const min = previewData[`${ci}-${ni}`]?.min ?? 0;
                        const max = previewData[`${ci}-${ni}`]?.max ?? 1;
                        const x = node.cx - cardWidth / 2 + 4 + (index / Math.max(arr.length - 1, 1)) * (cardWidth - 8);
                        const y = node.cy + cardHeight / 2 - 4 - ((value - min) / Math.max(max - min, 1e-6)) * (cardHeight - 8);
                        return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
                      }).join(' ')}
                      fill="none"
                      stroke={column.layer.type === 'output' ? '#f97316' : column.layer.type === 'input' ? '#14b8a6' : '#60a5fa'}
                      strokeWidth="1.2"
                      strokeLinecap="round"
                    />
                  </>
                ) : (
                  (() => {
                    const preview = previewData[`${ci}-${ni}`];
                    if (!preview) return null;
                    const cells = [];
                    const innerPad = 4;
                    const cell = (cardWidth - innerPad * 2) / Math.max(preview.previewRes, 1);
                    const values = preview.values ?? [];
                    for (let row = 0; row < preview.previewRes; row++) {
                      for (let col = 0; col < preview.previewRes; col++) {
                        const value = values[row * preview.previewRes + col] ?? preview.min ?? 0;
                        cells.push(
                          <rect
                            key={`${row}-${col}`}
                            x={node.cx - cardWidth / 2 + innerPad + col * cell}
                            y={node.cy - cardHeight / 2 + innerPad + row * cell}
                            width={cell + 0.4}
                            height={cell + 0.4}
                            fill={activationColor(value, preview.min, preview.max)}
                            opacity="0.94"
                          />,
                        );
                      }
                    }
                    return cells;
                  })()
                )}
              </g>
            ))}
            <text x={column.x} y="18" textAnchor="middle" fontSize="9" fill="rgba(226,232,240,0.86)" fontFamily="JetBrains Mono" letterSpacing="0.16em">
              {column.layer.type === 'input' ? 'INPUT' : column.layer.type === 'output' ? 'OUTPUT' : `BLOCK ${ci}`}
            </text>
            <text x={column.x} y={height - 10} textAnchor="middle" fontSize="10" fill="rgba(148,163,184,0.82)" fontFamily="JetBrains Mono">
              {column.layer.label}
            </text>
            {column.nodes.map((node, ni) => node.label ? (
              <text key={`label-${ni}`} x={node.cx} y={node.cy + 3} textAnchor="middle" fontSize="8" fill="rgba(241,245,249,0.92)" fontFamily="JetBrains Mono">
                {node.label}
              </text>
            ) : null)}
          </g>
        ))}
      </svg>
        </div>
      </div>
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
      const inputLabel = task === 'regression' ? 'x₀' : 'x₀ x₁';
      return {
        nodeType: 'input',
        title: 'Input',
        subtitle: useFourier ? `Fourier ×${numBands}` : inputLabel,
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
    const inputDim = task === 'regression' ? 1 : 2;
    networkRef.current = createNetwork(configRef.current, inputDim, outputDim, useFourier, numBands);
    stepRef.current = 0;
    setStep(0);
    setLossHistory([]);
    lossBufferRef.current = [];
    networkRef.current.forward(makeGrid(inputDim), false);
    setTick(v => v + 1);
  }, [numBands, outputDim, useFourier, task]);

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
    net.forward(makeGrid(net.inputDim ?? 2), false);
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
  const inputDim = networkRef.current?.inputDim ?? 2;
  const pyCode = useMemo(
    () => exportPyTorch(config, inputDim, task, numClasses, useFourier, numBands, optimizer, baseLr, weightDecay),
    [baseLr, config, inputDim, numBands, numClasses, optimizer, task, useFourier, weightDecay],
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
                  <span className="nl-arch-block-meta">{useFourier ? `Fourier ×${numBands}` : task === 'regression' ? 'x₀' : 'x₀ · x₁'}</span>
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
                <select className="nl-select mono" value={optimizer} onChange={e => setOptimizer(e.target.value)}>
                  {OPTIMIZER_OPTIONS.map(o => (
                    <option key={o.value} value={o.value}>{o.label || o.value}</option>
                  ))}
                </select>
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

        {/* Top row: prediction + network graph */}
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
              task={task} numClasses={numClasses} tick={tick} inputDim={networkRef.current?.inputDim ?? 2} />
            <div className="mono" style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 9, color: 'var(--fg-4)' }}>
              <span>-2.25</span><span>{task === 'regression' ? 'x₀ → y' : 'x₀ · x₁'}</span><span>2.25</span>
            </div>
          </CanvasPanel>

          <CanvasPanel icon={Layers} title="Network Graph">
            <NetworkDiagram network={networkRef.current} config={config} updateLayer={updateLayer} useFourier={useFourier}
              numBands={numBands} outputDim={outputDim} task={task} tick={tick} />
          </CanvasPanel>
        </div>

        {/* Bottom row: loss curve + gradient flow */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 10, marginTop: 10, flexShrink: 0 }}>

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
