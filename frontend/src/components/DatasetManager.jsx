import React, { useState } from 'react';
import { Database, FileSpreadsheet, Upload, Play, Loader2, CheckCircle2, BarChart2, Hash, AlertCircle } from 'lucide-react';

export function DatasetManager({ projectId, onTrain, busy, metrics, inspection }) {
  const [file, setFile] = useState(null);
  const [useDemo, setUseDemo] = useState(true);

  function handleFileChange(e) {
    const selected = e.target.files?.[0] || null;
    if (selected) {
      setFile(selected);
      setUseDemo(false);
    }
  }

  function handleTrainClick() {
    onTrain(useDemo ? null : file);
  }

  return (
    <section className="panel data-panel">
      <div className="panel-head">
        <div>
          <div className="kicker">04 · DATASET & REAL ML TRAINING</div>
          <h2>Dataset Inspection & Training</h2>
        </div>
        <Database className="panel-icon" size={24} />
      </div>

      <div className="dataset-selection-card">
        <div className="selection-toggle">
          <label className={`toggle-option ${useDemo ? 'selected' : ''}`}>
            <input
              type="radio"
              name="dataset-mode"
              checked={useDemo}
              onChange={() => { setUseDemo(true); setFile(null); }}
            />
            <FileSpreadsheet size={16} /> Use Bundled Legal Demo Dataset
          </label>
          <label className={`toggle-option ${!useDemo ? 'selected' : ''}`}>
            <input
              type="radio"
              name="dataset-mode"
              checked={!useDemo}
              onChange={() => setUseDemo(false)}
            />
            <Upload size={16} /> Upload Custom CSV Dataset
          </label>
        </div>

        {!useDemo && (
          <div className="file-input-wrapper">
            <input type="file" accept=".csv" onChange={handleFileChange} />
            <div className="file-drop-info">
              <Upload size={20} />
              <span>{file ? file.name : 'Choose a CSV file with "text" and "label" columns'}</span>
            </div>
          </div>
        )}

        <button
          className="primary-btn train-btn"
          onClick={handleTrainClick}
          disabled={!!busy || (!useDemo && !file)}
        >
          {busy === 'train' ? (
            <>
              <Loader2 className="spin" size={18} /> Training ML Model…
            </>
          ) : (
            <>
              <Play size={18} /> Start Real ML Training
            </>
          )}
        </button>
      </div>

      {inspection && (
        <div className="inspection-card">
          <div className="card-title"><BarChart2 size={16} /> Dataset Inspection Summary</div>
          <div className="inspection-grid">
            <div className="stat-box">
              <Hash size={16} />
              <div><span>Total Rows</span><strong>{inspection.rows}</strong></div>
            </div>
            <div className="stat-box">
              <div><span>Classes Detected</span><strong>{inspection.classes}</strong></div>
            </div>
            <div className="stat-box">
              <div><span>Text Column</span><strong>{inspection.text_column || 'None'}</strong></div>
            </div>
            <div className="stat-box">
              <div><span>Label Column</span><strong>{inspection.label_column || 'None'}</strong></div>
            </div>
          </div>

          {inspection.class_distribution && (
            <div className="class-distro-box">
              <h4>Class Distribution</h4>
              <div className="distro-tags">
                {Object.entries(inspection.class_distribution).map(([label, count]) => (
                  <span className="distro-tag" key={label}>
                    <strong>{label}:</strong> {count} samples
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {metrics && (
        <div className="metrics-card">
          <div className="card-title"><CheckCircle2 size={16} /> Trained Model Performance Metrics</div>
          <div className="metrics-grid">
            <div className="metric-box">
              <span>Accuracy</span>
              <strong>{(metrics.accuracy * 100).toFixed(1)}%</strong>
            </div>
            <div className="metric-box">
              <span>Weighted F1</span>
              <strong>{metrics.f1.toFixed(3)}</strong>
            </div>
            <div className="metric-box">
              <span>Precision</span>
              <strong>{metrics.precision.toFixed(3)}</strong>
            </div>
            <div className="metric-box">
              <span>Recall</span>
              <strong>{metrics.recall.toFixed(3)}</strong>
            </div>
            <div className="metric-box">
              <span>Training Samples</span>
              <strong>{metrics.samples}</strong>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
