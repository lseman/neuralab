import React, { useEffect, useRef } from 'react';

import { DATASETS, GRID_RES, makeGrid } from './core.js';

export function DatasetThumb({ type, size = 28 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const ratio = window.devicePixelRatio || 1;
    canvas.width = size * ratio;
    canvas.height = size * ratio;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const sample = DATASETS[type].fn(60);
    const isRegression = DATASETS[type].task === 'regression';
    for (const i in sample.X) {
      const x = sample.X[i][0];
      const y = sample.X[i][1];
      const px = (x / 4 + 0.5) * size;
      const py = (-y / 4 + 0.5) * size;
      if (isRegression) {
        const value = sample.y[i];
        const t = (value + 1) / 2;
        ctx.fillStyle = `rgba(${82 + t * 180},${133 + t * 36},${255 - t * 120}, 0.9)`;
      } else if (DATASETS[type].task === 'multiclass') {
        const colors = ['#52c7ff', '#f4b53f', '#18c37e'];
        ctx.fillStyle = colors[sample.y[i]];
      } else {
        ctx.fillStyle = sample.y[i] === 0 ? '#52c7ff' : '#ff8447';
      }
      ctx.beginPath();
      ctx.arc(px, py, 1.4, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [size, type]);

  return <canvas ref={canvasRef} />;
}

export function DecisionBoundary({ network, dataset, task, numClasses, tick }) {
  const canvasRef = useRef(null);
  const SIZE = 380;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !network) return;
    const ctx = canvas.getContext('2d');
    const ratio = window.devicePixelRatio || 1;
    canvas.width = SIZE * ratio;
    canvas.height = SIZE * ratio;
    canvas.style.width = `${SIZE}px`;
    canvas.style.height = `${SIZE}px`;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, SIZE, SIZE);

    const grid = makeGrid(GRID_RES);
    const out = network.forward(grid, false);
    const cell = SIZE / GRID_RES;

    for (let i = 0; i < GRID_RES; i++) {
      for (let j = 0; j < GRID_RES; j++) {
        const index = i * GRID_RES + j;
        const o = out[index];
        let r;
        let g;
        let b;
        if (task === 'binary') {
          const p = 1 / (1 + Math.exp(-o[0]));
          r = Math.round(82 + p * (255 - 82));
          g = Math.round(199 + p * (109 - 199));
          b = Math.round(255 + p * (45 - 255));
        } else if (task === 'multiclass') {
          const palette = [[82, 199, 255], [244, 181, 63], [24, 195, 126]];
          let mx = -Infinity;
          for (let k = 0; k < numClasses; k++) if (o[k] > mx) mx = o[k];
          let sum = 0;
          const probs = new Array(numClasses).fill(0);
          for (let k = 0; k < numClasses; k++) {
            probs[k] = Math.exp(o[k] - mx);
            sum += probs[k];
          }
          r = 0;
          g = 0;
          b = 0;
          for (let k = 0; k < numClasses; k++) {
            probs[k] /= sum;
            r += probs[k] * palette[k][0];
            g += probs[k] * palette[k][1];
            b += probs[k] * palette[k][2];
          }
        } else {
          const value = Math.max(-1, Math.min(1, o[0]));
          if (value < 0) {
            const t = -value;
            r = Math.round(240 * (1 - t) + 82 * t);
            g = Math.round(240 * (1 - t) + 199 * t);
            b = Math.round(240 * (1 - t) + 255 * t);
          } else {
            r = Math.round(240 * (1 - value) + 255 * value);
            g = Math.round(240 * (1 - value) + 109 * value);
            b = Math.round(240 * (1 - value) + 45 * value);
          }
        }
        ctx.fillStyle = `rgba(${r | 0}, ${g | 0}, ${b | 0}, 0.88)`;
        ctx.fillRect(j * cell, i * cell, cell + 0.5, cell + 0.5);
      }
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let k = 1; k < 4; k++) {
      const p = (k * SIZE) / 4;
      ctx.beginPath();
      ctx.moveTo(p, 0);
      ctx.lineTo(p, SIZE);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, p);
      ctx.lineTo(SIZE, p);
      ctx.stroke();
    }

    if (!dataset) return;
    for (let i = 0; i < dataset.X.length; i++) {
      const x = dataset.X[i][0];
      const y = dataset.X[i][1];
      const px = (x / 4.5 + 0.5) * SIZE;
      const py = (-y / 4.5 + 0.5) * SIZE;
      if (task === 'binary') {
        ctx.fillStyle = dataset.y[i] === 0 ? '#132c43' : '#4b240d';
        ctx.strokeStyle = dataset.y[i] === 0 ? '#52c7ff' : '#ff9d57';
      } else if (task === 'multiclass') {
        const fills = ['#132c43', '#4b240d', '#103828'];
        const strokes = ['#52c7ff', '#f4b53f', '#18c37e'];
        ctx.fillStyle = fills[dataset.y[i]];
        ctx.strokeStyle = strokes[dataset.y[i]];
      } else {
        ctx.fillStyle = dataset.y[i] < 0 ? '#132c43' : '#4b240d';
        ctx.strokeStyle = '#d7dee9';
      }
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }, [dataset, network, numClasses, task, tick]);

  return <canvas ref={canvasRef} style={{ display: 'block', width: '100%', maxWidth: SIZE, borderRadius: 6 }} />;
}

export function NeuronMap({ network, layerIdx, neuronIdx, tick }) {
  const canvasRef = useRef(null);
  const SIZE = 56;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !network) return;
    const ctx = canvas.getContext('2d');
    const ratio = window.devicePixelRatio || 1;
    canvas.width = SIZE * ratio;
    canvas.height = SIZE * ratio;
    canvas.style.width = `${SIZE}px`;
    canvas.style.height = `${SIZE}px`;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

    const activations = network.activations[layerIdx + 1];
    if (!activations) return;
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < activations.length; i++) {
      const value = activations[i][neuronIdx];
      if (value < min) min = value;
      if (value > max) max = value;
    }
    const range = Math.max(1e-6, max - min);
    const cell = SIZE / GRID_RES;
    for (let i = 0; i < GRID_RES; i++) {
      for (let j = 0; j < GRID_RES; j++) {
        const value = activations[i * GRID_RES + j][neuronIdx];
        const t = (value - min) / range;
        const r = Math.round(82 + t * (255 - 82));
        const g = Math.round(199 + t * (109 - 199));
        const b = Math.round(255 + t * (45 - 255));
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(j * cell, i * cell, cell + 0.5, cell + 0.5);
      }
    }
  }, [layerIdx, network, neuronIdx, tick]);

  return <canvas ref={canvasRef} style={{ display: 'block', borderRadius: 4 }} />;
}

export function GradFlowBars({ gradNorms, weightNorms, layers }) {
  const maxG = Math.max(1e-6, ...gradNorms);
  const maxW = Math.max(1e-6, ...weightNorms);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
      {layers.map((layer, index) => (
        <div key={`${layer.type}-${index}`} className="nl-grad-bar-row">
          <span className="nl-grad-bar-label">
            {layer.type === 'dense' && layer.act ? layer.act.slice(0, 5) : layer.type.slice(0, 5)}
          </span>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
            <div className="nl-grad-bar-track">
              <div className="nl-grad-bar-fill"
                style={{ width: `${(gradNorms[index] / maxG) * 100}%`, background: 'linear-gradient(90deg, rgba(249,115,22,0.5), var(--node-orange))' }} />
            </div>
            <div className="nl-grad-bar-track">
              <div className="nl-grad-bar-fill"
                style={{ width: `${(weightNorms[index] / maxW) * 100}%`, background: 'linear-gradient(90deg, rgba(59,130,246,0.4), var(--node-blue))' }} />
            </div>
          </div>
          <span className="nl-grad-bar-val">{gradNorms[index].toExponential(1)}</span>
        </div>
      ))}
    </div>
  );
}
