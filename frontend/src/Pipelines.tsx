import { useState, useEffect } from 'react';
import { Settings, Play, Database, Server, RefreshCw, CheckCircle, AlertTriangle } from 'lucide-react';

interface TileData {
  title: string;
  value: string;
  description: string;
  details: string;
}

export default function Pipelines() {
  const [tiles, setTiles] = useState<TileData[]>([]);
  const [ingestStatus, setIngestStatus] = useState<{ status: 'idle' | 'loading' | 'success' | 'error', message: string }>({ status: 'idle', message: '' });
  const [validateStatus, setValidateStatus] = useState<{ status: 'idle' | 'loading' | 'success' | 'error', summary?: any }>({ status: 'idle' });

  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/config');
      const data = await res.json();
      setTiles(data.tiles || []);
    } catch (e) {
      console.error('Failed to fetch config', e);
    }
  };

  const handleIngest = async () => {
    setIngestStatus({ status: 'loading', message: 'Running ingestion pipelines...' });
    try {
      const res = await fetch('http://127.0.0.1:8000/ingest', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setIngestStatus({ status: 'success', message: data.message });
      } else {
        setIngestStatus({ status: 'error', message: data.detail || 'Ingestion failed' });
      }
    } catch (e: any) {
      setIngestStatus({ status: 'error', message: e.message || 'Network error' });
    }
  };

  const handleValidate = async () => {
    setValidateStatus({ status: 'loading' });
    try {
      const res = await fetch('http://127.0.0.1:8000/validate', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setValidateStatus({ status: 'success', summary: data.summary });
        fetchConfig();
      } else {
        setValidateStatus({ status: 'error', summary: data.detail || 'Validation failed' });
      }
    } catch (e: any) {
      setValidateStatus({ status: 'error', summary: e.message || 'Network error' });
    }
  };

  return (
    <div className="pipelines-container">
      <div className="pipelines-header">
        <h1><Settings className="icon" size={28} /> Control Panel & Pipelines</h1>
        <p>Manage RAG system configuration, update vector & graph databases, and evaluate pipeline performance.</p>
      </div>

      <div className="tiles-section">
        <h2>RAG Evaluation Metrics</h2>
        <div className="tiles-grid">
          {tiles.length === 0 ? (
            <div className="loading-state">Loading parameters...</div>
          ) : (
            tiles.map((tile, idx) => (
              <div key={idx} className="config-tile">
                <h3>{tile.title}</h3>
                <div className="tile-value">{tile.value}</div>
                <div className="tile-desc">{tile.description}</div>
                <div className="tile-details">{tile.details}</div>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="actions-section">
        <h2>Data Operations</h2>
        <div className="actions-grid">
          {/* Ingestion Card */}
          <div className="action-card">
            <div className="action-card-header">
              <Database size={24} className="icon-blue" />
              <h3>Re-run Ingestion</h3>
            </div>
            <p>Processes raw files, chunks text, generates embeddings, and populates Qdrant and Neo4j databases.</p>
            <button 
              className={`btn-primary ${ingestStatus.status === 'loading' ? 'loading' : ''}`}
              onClick={handleIngest}
              disabled={ingestStatus.status === 'loading'}
            >
              {ingestStatus.status === 'loading' ? <RefreshCw className="spin" size={16} /> : <Play size={16} />}
              {ingestStatus.status === 'loading' ? 'Ingesting...' : 'Start Ingestion'}
            </button>
            {ingestStatus.status === 'success' && (
              <div className="status-msg success"><CheckCircle size={16} /> {ingestStatus.message}</div>
            )}
            {ingestStatus.status === 'error' && (
              <div className="status-msg error"><AlertTriangle size={16} /> {ingestStatus.message}</div>
            )}
          </div>

          {/* Validation Card */}
          <div className="action-card">
            <div className="action-card-header">
              <Server size={24} className="icon-purple" />
              <h3>Evaluate Pipeline</h3>
            </div>
            <p>Runs the LLM-as-a-judge benchmark suite to evaluate retrieval precision, latency, and groundedness.</p>
            <button 
              className={`btn-secondary ${validateStatus.status === 'loading' ? 'loading' : ''}`}
              onClick={handleValidate}
              disabled={validateStatus.status === 'loading'}
            >
              {validateStatus.status === 'loading' ? <RefreshCw className="spin" size={16} /> : <Play size={16} />}
              {validateStatus.status === 'loading' ? 'Evaluating...' : 'Run Benchmark'}
            </button>
            {validateStatus.status === 'success' && validateStatus.summary && (
              <div className="validation-results">
                <div className="metric"><span>Precision:</span> <strong>{validateStatus.summary.avg_precision}%</strong></div>
                <div className="metric"><span>Groundedness:</span> <strong>{validateStatus.summary.avg_groundedness}%</strong></div>
                <div className="metric"><span>Accuracy:</span> <strong>{validateStatus.summary.avg_accuracy}%</strong></div>
                <div className="metric"><span>Safety Rate:</span> <strong>{validateStatus.summary.safety_rate}%</strong></div>
              </div>
            )}
            {validateStatus.status === 'error' && (
              <div className="status-msg error"><AlertTriangle size={16} /> {typeof validateStatus.summary === 'string' ? validateStatus.summary : 'Error'}</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
