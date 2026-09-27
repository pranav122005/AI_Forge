import React, { useState, useRef } from 'react';
import { api } from '../../api/apiClient';
import {
  IconDatabase,
  IconRefresh,
  IconDownload,
  IconCheckCircle,
} from '../common/Icons';

export const DatasetManager = ({ projectId, onTrain, busy, metrics, inspection }) => {
  const [source, setSource] = useState('demo');
  const [file, setFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleTrainClick = () => {
    if (source === 'demo') {
      onTrain(null);
    } else {
      onTrain(file);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-header-title">
          <IconDatabase size={16} />
          <span className="kicker">DATASET & ML MODEL TRAINING</span>
        </div>
      </div>
      <div className="dataset-section">
        <div className="dataset-toggle">
          <button
            className={`toggle-option ${source === 'demo' ? 'toggle-option-active' : ''}`}
            onClick={() => setSource('demo')}
          >
            Built-in Legal Classification Dataset
          </button>
          <button
            className={`toggle-option ${source === 'custom' ? 'toggle-option-active' : ''}`}
            onClick={() => setSource('custom')}
          >
            Upload Custom CSV Dataset
          </button>
        </div>

        {source === 'custom' && (
          <div
            className="file-upload-area"
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              accept=".csv"
              className="hidden-input"
              ref={fileInputRef}
              onChange={handleFileChange}
            />
            <span className="file-upload-text">
              {file ? file.name : 'Click to select CSV file (text, label)'}
            </span>
          </div>
        )}

        <button
          className="btn-primary"
          onClick={handleTrainClick}
          disabled={busy || (source === 'custom' && !file)}
        >
          {busy ? (
            <>
              <IconRefresh size={14} className="icon-spin mr-1" />
              <span>TRAINING MODEL...</span>
            </>
          ) : (
            <span>TRAIN MODEL PIPELINE</span>
          )}
        </button>

        {inspection && (
          <div className="inspection-grid">
            <div className="stat-box">
              <span className="stat-label">SAMPLES</span>
              <span className="stat-value">{inspection.num_samples}</span>
            </div>
            <div className="stat-box">
              <span className="stat-label">CLASSES</span>
              <span className="stat-value">{inspection.num_classes}</span>
            </div>
            <div className="stat-box">
              <span className="stat-label">COLUMNS</span>
              <span className="stat-value">{inspection.columns?.join(', ')}</span>
            </div>
            {inspection.class_distribution && (
              <div className="stat-box distro-box">
                <span className="stat-label">CLASS DISTRIBUTION</span>
                <div className="distro-tags">
                  {Object.entries(inspection.class_distribution).map(([cls, count]) => (
                    <span key={cls} className="distro-tag">{cls}: {count}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {metrics && (
          <div className="metrics-grid">
            <div className="metric-box">
              <span className="metric-label">Accuracy</span>
              <span className="metric-value">{metrics.accuracy?.toFixed(4)}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">F1 Score</span>
              <span className="metric-value">{(metrics.f1_score || metrics.f1)?.toFixed(4)}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Precision</span>
              <span className="metric-value">{metrics.precision?.toFixed(4)}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Recall</span>
              <span className="metric-value">{metrics.recall?.toFixed(4)}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
