import React from 'react';
import ReactDOM from 'react-dom/client';
import Neuralab from './neuralab.jsx';

const root = document.getElementById('root');
if (!root) throw new Error('Root element not found');

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <Neuralab />
  </React.StrictMode>
);
