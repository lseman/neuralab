import { Activity, Circle, GitBranch, Grid3x3, Shuffle, Sparkles, TrendingUp, Waves } from 'lucide-react';

export const GRID_RES = 60;

const randn = (() => {
  let cached = null;
  return () => {
    if (cached !== null) {
      const value = cached;
      cached = null;
      return value;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    const mag = Math.sqrt(-2 * Math.log(u));
    cached = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
})();

const zerosMat = (rows, cols) => Array.from({ length: rows }, () => new Float32Array(cols));

const matmul = (X, W) => {
  const batch = X.length;
  const din = W.length;
  const dout = W[0].length;
  const out = Array.from({ length: batch }, () => new Float32Array(dout));
  for (let i = 0; i < batch; i++) {
    const xi = X[i];
    const oi = out[i];
    for (let k = 0; k < din; k++) {
      const value = xi[k];
      if (value === 0) continue;
      const wk = W[k];
      for (let j = 0; j < dout; j++) oi[j] += value * wk[j];
    }
  }
  return out;
};

const matmulATB = (A, B) => {
  const batch = A.length;
  const M = A[0].length;
  const N = B[0].length;
  const out = Array.from({ length: M }, () => new Float32Array(N));
  for (let i = 0; i < batch; i++) {
    const ai = A[i];
    const bi = B[i];
    for (let m = 0; m < M; m++) {
      const av = ai[m];
      if (av === 0) continue;
      const om = out[m];
      for (let n = 0; n < N; n++) om[n] += av * bi[n];
    }
  }
  return out;
};

const matmulABT = (A, B) => {
  const batch = A.length;
  const M = B.length;
  const N = B[0].length;
  const out = Array.from({ length: batch }, () => new Float32Array(M));
  for (let i = 0; i < batch; i++) {
    const ai = A[i];
    const oi = out[i];
    for (let m = 0; m < M; m++) {
      const bm = B[m];
      let sum = 0;
      for (let n = 0; n < N; n++) sum += ai[n] * bm[n];
      oi[m] = sum;
    }
  }
  return out;
};

const ACT = {
  relu: { f: (x) => (x > 0 ? x : 0), df: (x, y) => (y > 0 ? 1 : 0) },
  leaky_relu: { f: (x) => (x > 0 ? x : 0.1 * x), df: (x) => (x > 0 ? 1 : 0.1) },
  tanh: { f: Math.tanh, df: (x, y) => 1 - y * y },
  sigmoid: { f: (x) => 1 / (1 + Math.exp(-x)), df: (x, y) => y * (1 - y) },
  gelu: {
    f: (x) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x))),
    df: (x) => {
      const inner = 0.7978845608 * (x + 0.044715 * x * x * x);
      const t = Math.tanh(inner);
      const dInner = 0.7978845608 * (1 + 3 * 0.044715 * x * x);
      return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dInner;
    },
  },
  silu: {
    f: (x) => x / (1 + Math.exp(-x)),
    df: (x) => {
      const s = 1 / (1 + Math.exp(-x));
      return s + x * s * (1 - s);
    },
  },
  elu: {
    f: (x) => (x > 0 ? x : Math.exp(x) - 1),
    df: (x, y) => (x > 0 ? 1 : y + 1),
  },
  linear: { f: (x) => x, df: () => 1 },
};

export const ACT_NAMES = ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'gelu', 'silu', 'elu', 'linear'];

export const OPTIMIZER_OPTIONS = [
  { value: 'adam', label: 'Adam', detail: 'Fast default for most layouts' },
  { value: 'adamw', label: 'AdamW', detail: 'Adam with decoupled weight decay' },
  { value: 'rmsprop', label: 'RMSprop', detail: 'Stable on noisy gradients' },
  { value: 'momentum', label: 'Momentum SGD', detail: 'Classic momentum baseline' },
  { value: 'sgd', label: 'SGD', detail: 'Plain updates, easy to reason about' },
];

export const SCHEDULER_OPTIONS = [
  { value: 'const', label: 'Constant' },
  { value: 'cosine', label: 'Cosine decay' },
  { value: 'step', label: 'Step decay' },
  { value: 'warmup_cosine', label: 'Warmup + cosine' },
];

export class DenseLayer {
  constructor(din, dout, act = 'relu', residual = false) {
    this.type = 'dense';
    this.din = din;
    this.dout = dout;
    this.act = act;
    this.residual = residual && din === dout;
    const reluish = ['relu', 'leaky_relu', 'gelu', 'silu', 'elu'].includes(act);
    const scale = reluish ? Math.sqrt(2 / din) : Math.sqrt(1 / din);
    this.W = Array.from({ length: din }, () => {
      const row = new Float32Array(dout);
      for (let j = 0; j < dout; j++) row[j] = randn() * scale;
      return row;
    });
    this.b = new Float32Array(dout);
    this.dW = zerosMat(din, dout);
    this.db = new Float32Array(dout);
    this.mW = zerosMat(din, dout);
    this.vW = zerosMat(din, dout);
    this.mb = new Float32Array(dout);
    this.vb = new Float32Array(dout);
  }

  forward(X) {
    this.X = X;
    const batch = X.length;
    const pre = matmul(X, this.W);
    this.pre = Array.from({ length: batch }, () => new Float32Array(this.dout));
    this.post = Array.from({ length: batch }, () => new Float32Array(this.dout));
    for (let i = 0; i < batch; i++) {
      const xi = X[i];
      for (let j = 0; j < this.dout; j++) {
        const value = pre[i][j] + this.b[j] + (this.residual ? xi[j] : 0);
        this.pre[i][j] = value;
        this.post[i][j] = ACT[this.act].f(value);
      }
    }
    return this.post;
  }

  backward(dOut) {
    const batch = dOut.length;
    const dPre = Array.from({ length: batch }, () => new Float32Array(this.dout));
    for (let i = 0; i < batch; i++) {
      for (let j = 0; j < this.dout; j++) {
        dPre[i][j] = dOut[i][j] * ACT[this.act].df(this.pre[i][j], this.post[i][j]);
      }
    }
    const dW = matmulATB(this.X, dPre);
    const db = new Float32Array(this.dout);
    for (let i = 0; i < batch; i++) {
      for (let j = 0; j < this.dout; j++) db[j] += dPre[i][j];
    }
    const dX = matmulABT(dPre, this.W);
    if (this.residual) {
      for (let i = 0; i < batch; i++) {
        for (let j = 0; j < this.dout; j++) dX[i][j] += dPre[i][j];
      }
    }
    for (let i = 0; i < this.din; i++) {
      for (let j = 0; j < this.dout; j++) this.dW[i][j] += dW[i][j];
    }
    for (let j = 0; j < this.dout; j++) this.db[j] += db[j];
    return dX;
  }

  zeroGrad() {
    for (let i = 0; i < this.din; i++) this.dW[i].fill(0);
    this.db.fill(0);
  }

  gradNorm() {
    let sum = 0;
    for (let i = 0; i < this.din; i++) for (let j = 0; j < this.dout; j++) sum += this.dW[i][j] ** 2;
    for (let j = 0; j < this.dout; j++) sum += this.db[j] ** 2;
    return Math.sqrt(sum);
  }

  weightNorm() {
    let sum = 0;
    for (let i = 0; i < this.din; i++) for (let j = 0; j < this.dout; j++) sum += this.W[i][j] ** 2;
    return Math.sqrt(sum);
  }
}

export class DropoutLayer {
  constructor(rate = 0.1) {
    this.type = 'dropout';
    this.rate = rate;
  }

  forward(X, training) {
    this.din = X[0].length;
    this.dout = this.din;
    if (!training || this.rate === 0) {
      this.mask = null;
      return X;
    }
    const keep = 1 - this.rate;
    this.mask = X.map((row) => {
      const mask = new Float32Array(row.length);
      for (let j = 0; j < row.length; j++) mask[j] = Math.random() < keep ? 1 / keep : 0;
      return mask;
    });
    return X.map((row, i) => {
      const out = new Float32Array(row.length);
      for (let j = 0; j < row.length; j++) out[j] = row[j] * this.mask[i][j];
      return out;
    });
  }

  backward(dOut) {
    if (!this.mask) return dOut;
    return dOut.map((row, i) => {
      const out = new Float32Array(row.length);
      for (let j = 0; j < row.length; j++) out[j] = row[j] * this.mask[i][j];
      return out;
    });
  }

  zeroGrad() {}
  gradNorm() { return 0; }
  weightNorm() { return 0; }
}

class NormBase {
  zeroGrad() {
    this.dgamma.fill(0);
    this.dbeta.fill(0);
  }

  gradNorm() {
    let sum = 0;
    for (let j = 0; j < this.din; j++) sum += this.dgamma[j] ** 2 + this.dbeta[j] ** 2;
    return Math.sqrt(sum);
  }

  weightNorm() {
    let sum = 0;
    for (let j = 0; j < this.din; j++) sum += this.gamma[j] ** 2 + this.beta[j] ** 2;
    return Math.sqrt(sum);
  }
}

export class LayerNormLayer extends NormBase {
  constructor(dim) {
    super();
    this.type = 'layernorm';
    this.din = dim;
    this.dout = dim;
    this.gamma = new Float32Array(dim).fill(1);
    this.beta = new Float32Array(dim);
    this.dgamma = new Float32Array(dim);
    this.dbeta = new Float32Array(dim);
    this.mg = new Float32Array(dim);
    this.vg = new Float32Array(dim);
    this.mbt = new Float32Array(dim);
    this.vbt = new Float32Array(dim);
  }

  forward(X) {
    const B = X.length;
    const D = this.din;
    const eps = 1e-5;
    this.X = X;
    this.mu = new Float32Array(B);
    this.rv = new Float32Array(B);
    this.xhat = Array.from({ length: B }, () => new Float32Array(D));
    const out = Array.from({ length: B }, () => new Float32Array(D));
    for (let i = 0; i < B; i++) {
      let mean = 0;
      for (let j = 0; j < D; j++) mean += X[i][j];
      mean /= D;
      let variance = 0;
      for (let j = 0; j < D; j++) variance += (X[i][j] - mean) ** 2;
      variance /= D;
      const rv = 1 / Math.sqrt(variance + eps);
      this.mu[i] = mean;
      this.rv[i] = rv;
      for (let j = 0; j < D; j++) {
        this.xhat[i][j] = (X[i][j] - mean) * rv;
        out[i][j] = this.gamma[j] * this.xhat[i][j] + this.beta[j];
      }
    }
    return out;
  }

  backward(dOut) {
    const B = dOut.length;
    const D = this.din;
    const dX = Array.from({ length: B }, () => new Float32Array(D));
    for (let i = 0; i < B; i++) {
      const dxhat = new Float32Array(D);
      let sumDxhat = 0;
      let sumDxhatXhat = 0;
      for (let j = 0; j < D; j++) {
        this.dgamma[j] += dOut[i][j] * this.xhat[i][j];
        this.dbeta[j] += dOut[i][j];
        dxhat[j] = dOut[i][j] * this.gamma[j];
        sumDxhat += dxhat[j];
        sumDxhatXhat += dxhat[j] * this.xhat[i][j];
      }
      for (let j = 0; j < D; j++) {
        dX[i][j] = this.rv[i] * (dxhat[j] - sumDxhat / D - this.xhat[i][j] * sumDxhatXhat / D);
      }
    }
    return dX;
  }
}

export class BatchNormLayer extends NormBase {
  constructor(dim, momentum = 0.1) {
    super();
    this.type = 'batchnorm';
    this.din = dim;
    this.dout = dim;
    this.gamma = new Float32Array(dim).fill(1);
    this.beta = new Float32Array(dim);
    this.runningMean = new Float32Array(dim);
    this.runningVar = new Float32Array(dim).fill(1);
    this.dgamma = new Float32Array(dim);
    this.dbeta = new Float32Array(dim);
    this.mg = new Float32Array(dim);
    this.vg = new Float32Array(dim);
    this.mbt = new Float32Array(dim);
    this.vbt = new Float32Array(dim);
    this.momentum = momentum;
  }

  forward(X, training) {
    const B = X.length;
    const D = this.din;
    const eps = 1e-5;
    this.X = X;
    this.xhat = Array.from({ length: B }, () => new Float32Array(D));
    const out = Array.from({ length: B }, () => new Float32Array(D));
    if (training) {
      const mean = new Float32Array(D);
      const variance = new Float32Array(D);
      for (let j = 0; j < D; j++) {
        for (let i = 0; i < B; i++) mean[j] += X[i][j];
        mean[j] /= B;
      }
      for (let j = 0; j < D; j++) {
        for (let i = 0; i < B; i++) {
          const delta = X[i][j] - mean[j];
          variance[j] += delta * delta;
        }
        variance[j] /= B;
        this.runningMean[j] = this.momentum * mean[j] + (1 - this.momentum) * this.runningMean[j];
        this.runningVar[j] = this.momentum * variance[j] + (1 - this.momentum) * this.runningVar[j];
      }
      this.rv = new Float32Array(D);
      for (let j = 0; j < D; j++) this.rv[j] = 1 / Math.sqrt(variance[j] + eps);
      for (let i = 0; i < B; i++) {
        for (let j = 0; j < D; j++) {
          this.xhat[i][j] = (X[i][j] - mean[j]) * this.rv[j];
          out[i][j] = this.gamma[j] * this.xhat[i][j] + this.beta[j];
        }
      }
      return out;
    }
    this.rv = new Float32Array(D);
    for (let j = 0; j < D; j++) this.rv[j] = 1 / Math.sqrt(this.runningVar[j] + eps);
    for (let i = 0; i < B; i++) {
      for (let j = 0; j < D; j++) {
        this.xhat[i][j] = (X[i][j] - this.runningMean[j]) * this.rv[j];
        out[i][j] = this.gamma[j] * this.xhat[i][j] + this.beta[j];
      }
    }
    return out;
  }

  backward(dOut) {
    const B = dOut.length;
    const D = this.din;
    const dX = Array.from({ length: B }, () => new Float32Array(D));
    for (let i = 0; i < B; i++) {
      for (let j = 0; j < D; j++) {
        this.dgamma[j] += dOut[i][j] * this.xhat[i][j];
        this.dbeta[j] += dOut[i][j];
        dX[i][j] = dOut[i][j] * this.gamma[j];
      }
    }
    for (let j = 0; j < D; j++) {
      let sumDX = 0;
      let sumDXxhat = 0;
      for (let i = 0; i < B; i++) {
        sumDX += dX[i][j];
        sumDXxhat += dX[i][j] * this.xhat[i][j];
      }
      for (let i = 0; i < B; i++) {
        dX[i][j] = this.rv[j] * (dX[i][j] - sumDX / B - this.xhat[i][j] * sumDXxhat / B);
      }
    }
    return dX;
  }
}

export class RMSNormLayer extends NormBase {
  constructor(dim) {
    super();
    this.type = 'rmsnorm';
    this.din = dim;
    this.dout = dim;
    this.gamma = new Float32Array(dim).fill(1);
    this.beta = new Float32Array(dim);
    this.dgamma = new Float32Array(dim);
    this.dbeta = new Float32Array(dim);
    this.mg = new Float32Array(dim);
    this.vg = new Float32Array(dim);
    this.mbt = new Float32Array(dim);
    this.vbt = new Float32Array(dim);
  }

  forward(X) {
    const B = X.length;
    const D = this.din;
    const eps = 1e-5;
    this.X = X;
    this.xhat = Array.from({ length: B }, () => new Float32Array(D));
    const out = Array.from({ length: B }, () => new Float32Array(D));
    for (let i = 0; i < B; i++) {
      let rms = 0;
      for (let j = 0; j < D; j++) rms += X[i][j] * X[i][j];
      const inv = 1 / Math.sqrt(rms / D + eps);
      for (let j = 0; j < D; j++) {
        this.xhat[i][j] = X[i][j] * inv;
        out[i][j] = this.gamma[j] * this.xhat[i][j] + this.beta[j];
      }
    }
    return out;
  }

  backward(dOut) {
    const B = dOut.length;
    const D = this.din;
    const dX = Array.from({ length: B }, () => new Float32Array(D));
    for (let i = 0; i < B; i++) {
      let rms = 0;
      for (let j = 0; j < D; j++) rms += this.X[i][j] * this.X[i][j];
      const inv = 1 / Math.sqrt(rms / D + 1e-5);
      let sum = 0;
      for (let j = 0; j < D; j++) {
        this.dgamma[j] += dOut[i][j] * this.xhat[i][j];
        this.dbeta[j] += dOut[i][j];
        sum += dOut[i][j] * this.xhat[i][j];
      }
      const mean = sum / D;
      for (let j = 0; j < D; j++) dX[i][j] = this.gamma[j] * inv * (dOut[i][j] - mean * this.xhat[i][j]);
    }
    return dX;
  }
}

export class FourierFeaturesLayer {
  constructor(din, numBands = 4) {
    this.type = 'fourier';
    this.din = din;
    this.numBands = numBands;
    this.dout = din * 2 * numBands;
  }

  forward(X) {
    const B = X.length;
    const K = this.numBands;
    const D = this.din;
    const out = Array.from({ length: B }, () => new Float32Array(this.dout));
    for (let i = 0; i < B; i++) {
      for (let d = 0; d < D; d++) {
        for (let k = 0; k < K; k++) {
          const f = 2 ** k * Math.PI;
          out[i][d * 2 * K + 2 * k] = Math.sin(f * X[i][d]);
          out[i][d * 2 * K + 2 * k + 1] = Math.cos(f * X[i][d]);
        }
      }
    }
    return out;
  }

  backward(dOut) { return dOut; }
  zeroGrad() {}
  gradNorm() { return 0; }
  weightNorm() { return 0; }
}

export class Network {
  constructor(layers) {
    this.layers = layers;
    this.t = 0;
  }

  forward(X, training = true) {
    this.activations = [X];
    let value = X;
    for (const layer of this.layers) {
      value = layer.forward(value, training);
      this.activations.push(value);
    }
    return value;
  }

  backward(dOut) {
    let delta = dOut;
    for (let i = this.layers.length - 1; i >= 0; i--) delta = this.layers[i].backward(delta);
    return delta;
  }

  zeroGrad() {
    for (const layer of this.layers) layer.zeroGrad();
  }

  step(opt, lr, wd = 0) {
    this.t += 1;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    for (const layer of this.layers) {
      if (layer.type === 'dense') {
        for (let i = 0; i < layer.din; i++) {
          for (let j = 0; j < layer.dout; j++) {
            let g = layer.dW[i][j];
            if (opt === 'sgd') {
              layer.W[i][j] -= lr * (g + wd * layer.W[i][j]);
            } else if (opt === 'momentum') {
              layer.mW[i][j] = 0.9 * layer.mW[i][j] + g;
              layer.W[i][j] -= lr * (layer.mW[i][j] + wd * layer.W[i][j]);
            } else if (opt === 'rmsprop') {
              layer.vW[i][j] = 0.9 * layer.vW[i][j] + 0.1 * g * g;
              layer.W[i][j] -= lr * g / (Math.sqrt(layer.vW[i][j]) + eps) + lr * wd * layer.W[i][j];
            } else if (opt === 'adam') {
              if (wd !== 0) g += wd * layer.W[i][j];
              layer.mW[i][j] = beta1 * layer.mW[i][j] + (1 - beta1) * g;
              layer.vW[i][j] = beta2 * layer.vW[i][j] + (1 - beta2) * g * g;
              const mhat = layer.mW[i][j] / (1 - Math.pow(beta1, this.t));
              const vhat = layer.vW[i][j] / (1 - Math.pow(beta2, this.t));
              layer.W[i][j] -= lr * mhat / (Math.sqrt(vhat) + eps);
            } else if (opt === 'adamw') {
              layer.mW[i][j] = beta1 * layer.mW[i][j] + (1 - beta1) * g;
              layer.vW[i][j] = beta2 * layer.vW[i][j] + (1 - beta2) * g * g;
              const mhat = layer.mW[i][j] / (1 - Math.pow(beta1, this.t));
              const vhat = layer.vW[i][j] / (1 - Math.pow(beta2, this.t));
              layer.W[i][j] -= lr * (mhat / (Math.sqrt(vhat) + eps) + wd * layer.W[i][j]);
            }
          }
        }
        for (let j = 0; j < layer.dout; j++) {
          const g = layer.db[j];
          if (opt === 'sgd') {
            layer.b[j] -= lr * g;
          } else if (opt === 'momentum') {
            layer.mb[j] = 0.9 * layer.mb[j] + g;
            layer.b[j] -= lr * layer.mb[j];
          } else if (opt === 'rmsprop') {
            layer.vb[j] = 0.9 * layer.vb[j] + 0.1 * g * g;
            layer.b[j] -= lr * g / (Math.sqrt(layer.vb[j]) + eps);
          } else {
            layer.mb[j] = 0.9 * layer.mb[j] + 0.1 * g;
            layer.vb[j] = 0.999 * layer.vb[j] + 0.001 * g * g;
            const mhat = layer.mb[j] / (1 - Math.pow(0.9, this.t));
            const vhat = layer.vb[j] / (1 - Math.pow(0.999, this.t));
            layer.b[j] -= lr * mhat / (Math.sqrt(vhat) + eps);
          }
        }
      } else if (['layernorm', 'batchnorm', 'rmsnorm'].includes(layer.type)) {
        for (let j = 0; j < layer.din; j++) {
          let g = layer.dgamma[j];
          if (opt === 'adam' || opt === 'adamw') {
            layer.mg[j] = 0.9 * layer.mg[j] + 0.1 * g;
            layer.vg[j] = 0.999 * layer.vg[j] + 0.001 * g * g;
            const mhat = layer.mg[j] / (1 - Math.pow(0.9, this.t));
            const vhat = layer.vg[j] / (1 - Math.pow(0.999, this.t));
            layer.gamma[j] -= lr * mhat / (Math.sqrt(vhat) + eps);

            g = layer.dbeta[j];
            layer.mbt[j] = 0.9 * layer.mbt[j] + 0.1 * g;
            layer.vbt[j] = 0.999 * layer.vbt[j] + 0.001 * g * g;
            const mhat2 = layer.mbt[j] / (1 - Math.pow(0.9, this.t));
            const vhat2 = layer.vbt[j] / (1 - Math.pow(0.999, this.t));
            layer.beta[j] -= lr * mhat2 / (Math.sqrt(vhat2) + eps);
          } else {
            layer.gamma[j] -= lr * g;
            layer.beta[j] -= lr * layer.dbeta[j];
          }
        }
      }
    }
  }
}

export function bceLoss(logits, yTrue) {
  const B = logits.length;
  let loss = 0;
  const dOut = Array.from({ length: B }, () => new Float32Array(1));
  for (let i = 0; i < B; i++) {
    const z = logits[i][0];
    const p = 1 / (1 + Math.exp(-z));
    const pc = Math.max(1e-7, Math.min(1 - 1e-7, p));
    const y = yTrue[i];
    loss += -(y * Math.log(pc) + (1 - y) * Math.log(1 - pc));
    dOut[i][0] = (p - y) / B;
  }
  return { loss: loss / B, dOut };
}

export function mseLoss(yPred, yTrue) {
  const B = yPred.length;
  let loss = 0;
  const dOut = Array.from({ length: B }, () => new Float32Array(1));
  for (let i = 0; i < B; i++) {
    const error = yPred[i][0] - yTrue[i];
    loss += error * error;
    dOut[i][0] = (2 * error) / B;
  }
  return { loss: loss / B, dOut };
}

export function softmaxCELoss(logits, yTrue, K) {
  const B = logits.length;
  let loss = 0;
  const dOut = Array.from({ length: B }, () => new Float32Array(K));
  for (let i = 0; i < B; i++) {
    let mx = -Infinity;
    for (let k = 0; k < K; k++) if (logits[i][k] > mx) mx = logits[i][k];
    let sum = 0;
    const exps = new Float32Array(K);
    for (let k = 0; k < K; k++) {
      exps[k] = Math.exp(logits[i][k] - mx);
      sum += exps[k];
    }
    for (let k = 0; k < K; k++) exps[k] /= sum;
    loss += -Math.log(Math.max(1e-7, exps[yTrue[i]]));
    for (let k = 0; k < K; k++) dOut[i][k] = (exps[k] - (k === yTrue[i] ? 1 : 0)) / B;
  }
  return { loss: loss / B, dOut };
}

function genCircle(n = 200, noise = 0.1) {
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const r = Math.random() < 0.5 ? 0.3 + Math.random() * 0.2 : 0.7 + Math.random() * 0.2;
    const a = Math.random() * 2 * Math.PI;
    const x0 = r * Math.cos(a) + (Math.random() - 0.5) * noise;
    const x1 = r * Math.sin(a) + (Math.random() - 0.5) * noise;
    X.push(new Float32Array([x0 * 2, x1 * 2]));
    y.push(r < 0.5 ? 0 : 1);
  }
  return { X, y, classes: 2 };
}

function genMoons(n = 200, noise = 0.15) {
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2;
    const t = Math.random() * Math.PI;
    let x0;
    let x1;
    if (c === 0) {
      x0 = Math.cos(t);
      x1 = Math.sin(t);
    } else {
      x0 = 1 - Math.cos(t);
      x1 = 0.5 - Math.sin(t);
    }
    x0 += (Math.random() - 0.5) * noise;
    x1 += (Math.random() - 0.5) * noise;
    X.push(new Float32Array([(x0 - 0.5) * 2, x1 * 2]));
    y.push(c);
  }
  return { X, y, classes: 2 };
}

function genSpiral(n = 200, noise = 0.1) {
  const X = [];
  const y = [];
  const perClass = Math.floor(n / 2);
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < perClass; i++) {
      const r = i / perClass;
      const t = 1.75 * i / perClass * 2 * Math.PI + c * Math.PI;
      const x0 = r * Math.sin(t) + (Math.random() - 0.5) * noise;
      const x1 = r * Math.cos(t) + (Math.random() - 0.5) * noise;
      X.push(new Float32Array([x0 * 2, x1 * 2]));
      y.push(c);
    }
  }
  return { X, y, classes: 2 };
}

function genXor(n = 200, noise = 0.1) {
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const x0 = (Math.random() - 0.5) * 4;
    const x1 = (Math.random() - 0.5) * 4;
    X.push(new Float32Array([
      x0 + (Math.random() - 0.5) * noise,
      x1 + (Math.random() - 0.5) * noise,
    ]));
    y.push(x0 * x1 > 0 ? 1 : 0);
  }
  return { X, y, classes: 2 };
}

function genChecker(n = 250, noise = 0.05) {
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const x0 = (Math.random() - 0.5) * 4;
    const x1 = (Math.random() - 0.5) * 4;
    X.push(new Float32Array([
      x0 + (Math.random() - 0.5) * noise,
      x1 + (Math.random() - 0.5) * noise,
    ]));
    y.push((Math.floor(x0 + 2) + Math.floor(x1 + 2)) % 2);
  }
  return { X, y, classes: 2 };
}

function genGaussians(n = 240, noise = 0.25) {
  const centers = [[-1.2, -1], [1.2, -1], [0, 1.3]];
  const X = [];
  const y = [];
  const per = Math.floor(n / 3);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < per; i++) {
      X.push(new Float32Array([
        centers[c][0] + randn() * noise,
        centers[c][1] + randn() * noise,
      ]));
      y.push(c);
    }
  }
  return { X, y, classes: 3 };
}

function genSine(n = 200, noise = 0.1) {
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const x0 = (Math.random() - 0.5) * 4;
    X.push(new Float32Array([x0, 0]));
    y.push(Math.sin(2 * x0) + randn() * noise);
  }
  return { X, y, classes: 0 };
}

function genRipple(n = 200, noise = 0.08) {
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const x0 = (Math.random() - 0.5) * 4;
    X.push(new Float32Array([x0, 0]));
    y.push(Math.sin(3 * x0) / (1 + 0.4 * Math.abs(x0)) + randn() * noise);
  }
  return { X, y, classes: 0 };
}

function genSaddle(n = 200, noise = 0.08) {
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const x0 = (Math.random() - 0.5) * 4;
    X.push(new Float32Array([x0, 0]));
    y.push(0.4 * x0 * x0 - 0.3 * x0 + randn() * noise);
  }
  return { X, y, classes: 0 };
}

export const DATASETS = {
  circle: { name: 'Circle', fn: genCircle, task: 'binary', Icon: Circle },
  moons: { name: 'Moons', fn: genMoons, task: 'binary', Icon: Waves },
  spiral: { name: 'Spiral', fn: genSpiral, task: 'binary', Icon: Shuffle },
  xor: { name: 'XOR', fn: genXor, task: 'binary', Icon: GitBranch },
  checker: { name: 'Checker', fn: genChecker, task: 'binary', Icon: Grid3x3 },
  gaussians: { name: '3 Gaussians', fn: genGaussians, task: 'multiclass', Icon: Sparkles },
  sine: { name: 'Sine Wave', fn: genSine, task: 'regression', Icon: Activity },
  ripple: { name: 'Noisy Sine', fn: genRipple, task: 'regression', Icon: Sparkles },
  saddle: { name: 'Quadratic Curve', fn: genSaddle, task: 'regression', Icon: TrendingUp },
};

export const DEFAULT_CONFIG = [
  { type: 'dense', units: 8, activation: 'tanh', residual: false },
  { type: 'dense', units: 8, activation: 'tanh', residual: false },
];

export function makeGrid(res = GRID_RES) {
  const X = [];
  for (let i = 0; i < res; i++) {
    for (let j = 0; j < res; j++) {
      const x0 = (j / (res - 1) - 0.5) * 4.5;
      const x1 = -(i / (res - 1) - 0.5) * 4.5;
      X.push(new Float32Array([x0, x1]));
    }
  }
  return X;
}

export function createNetwork(config, inputDim, outputDim, useFourier, numBands) {
  const layers = [];
  let din = inputDim;
  if (useFourier) {
    const fourier = new FourierFeaturesLayer(inputDim, numBands);
    layers.push(fourier);
    din = fourier.dout;
  }
  for (const layer of config) {
    if (layer.type === 'dense') {
      layers.push(new DenseLayer(din, layer.units, layer.activation, layer.residual));
      din = layer.units;
    } else if (layer.type === 'dropout') {
      layers.push(new DropoutLayer(layer.rate));
    } else if (layer.type === 'layernorm') {
      layers.push(new LayerNormLayer(din));
    } else if (layer.type === 'batchnorm') {
      layers.push(new BatchNormLayer(din));
    } else if (layer.type === 'rmsnorm') {
      layers.push(new RMSNormLayer(din));
    }
  }
  layers.push(new DenseLayer(din, outputDim, 'linear', false));
  return new Network(layers);
}

export function exportPyTorch(config, inputDim, task, numClasses, useFourier, numBands, optName, lr, wd) {
  let code = '# Auto-generated from Neuralab\\nimport torch\\nimport torch.nn as nn\\nimport torch.nn.functional as F\\n\\n';
  if (useFourier) {
    code += `class FourierFeatures(nn.Module):\\n    def __init__(self, dim, num_bands=${numBands}):\\n`;
    code += `        super().__init__(); self.num_bands = num_bands\\n`;
    code += `        self.register_buffer('freqs', (2 ** torch.arange(num_bands)) * torch.pi)\\n`;
    code += `    def forward(self, x):\\n        x = x.unsqueeze(-1) * self.freqs\\n`;
    code += '        return torch.cat([x.sin(), x.cos()], dim=-1).flatten(-2)\\n\\n';
  }
  code += 'class Net(nn.Module):\\n    def __init__(self):\\n        super().__init__()\\n';
  let din = inputDim;
  if (useFourier) {
    code += `        self.fourier = FourierFeatures(${inputDim}, ${numBands})\\n`;
    din = inputDim * 2 * numBands;
  }
  let denseIdx = 0;
  let auxIdx = 0;
  for (const layer of config) {
    if (layer.type === 'dense') {
      code += `        self.fc${denseIdx} = nn.Linear(${din}, ${layer.units})\\n`;
      din = layer.units;
      denseIdx++;
    } else if (layer.type === 'layernorm') {
      code += `        self.ln${auxIdx} = nn.LayerNorm(${din})\\n`;
      auxIdx++;
    } else if (layer.type === 'batchnorm') {
      code += `        self.bn${auxIdx} = nn.BatchNorm1d(${din})\\n`;
      auxIdx++;
    } else if (layer.type === 'rmsnorm') {
      code += `        self.rms${auxIdx} = nn.RMSNorm(${din})\\n`;
      auxIdx++;
    } else if (layer.type === 'dropout') {
      code += `        self.dp${auxIdx} = nn.Dropout(${layer.rate})\\n`;
      auxIdx++;
    }
  }
  const outUnits = task === 'multiclass' ? numClasses : 1;
  code += `        self.out = nn.Linear(${din}, ${outUnits})\\n\\n`;
  code += '    def forward(self, x):\\n';
  if (useFourier) code += '        x = self.fourier(x)\\n';
  const actMap = {
    relu: 'F.relu',
    leaky_relu: 'F.leaky_relu',
    tanh: 'torch.tanh',
    sigmoid: 'torch.sigmoid',
    gelu: 'F.gelu',
    silu: 'F.silu',
    elu: 'F.elu',
    linear: '',
  };
  denseIdx = 0;
  auxIdx = 0;
  for (const layer of config) {
    if (layer.type === 'dense') {
      const act = actMap[layer.activation];
      if (layer.residual) {
        code += `        h = self.fc${denseIdx}(x); x = ${act ? `${act}(h + x)` : 'h + x'}\\n`;
      } else {
        code += `        x = ${act ? `${act}(self.fc${denseIdx}(x))` : `self.fc${denseIdx}(x)`}\\n`;
      }
      denseIdx++;
    } else if (layer.type === 'layernorm') {
      code += `        x = self.ln${auxIdx}(x)\\n`;
      auxIdx++;
    } else if (layer.type === 'batchnorm') {
      code += `        x = self.bn${auxIdx}(x)\\n`;
      auxIdx++;
    } else if (layer.type === 'rmsnorm') {
      code += `        x = self.rms${auxIdx}(x)\\n`;
      auxIdx++;
    } else if (layer.type === 'dropout') {
      code += `        x = self.dp${auxIdx}(x)\\n`;
      auxIdx++;
    }
  }
  if (task === 'binary') code += '        return torch.sigmoid(self.out(x))\\n';
  else code += '        return self.out(x)\\n';

  const optMap = {
    sgd: 'torch.optim.SGD(model.parameters(), lr=%lr)',
    momentum: 'torch.optim.SGD(model.parameters(), lr=%lr, momentum=0.9)',
    rmsprop: 'torch.optim.RMSprop(model.parameters(), lr=%lr)',
    adam: 'torch.optim.Adam(model.parameters(), lr=%lr, weight_decay=%wd)',
    adamw: 'torch.optim.AdamW(model.parameters(), lr=%lr, weight_decay=%wd)',
  };

  code += '\\nmodel = Net()\\n';
  code += `optim = ${optMap[optName].replace('%lr', lr).replace('%wd', wd)}\\n`;
  if (task === 'binary') code += 'loss_fn = nn.BCELoss()\\n';
  else if (task === 'multiclass') code += 'loss_fn = nn.CrossEntropyLoss()\\n';
  else code += 'loss_fn = nn.MSELoss()\\n';
  return code;
}
