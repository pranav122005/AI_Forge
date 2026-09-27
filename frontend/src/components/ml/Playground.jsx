import React, { useState, useRef } from 'react';
import { api } from '../../api/apiClient';
import {
  IconFlask,
  IconRefresh,
  IconEye,
  IconCheckCircle,
} from '../common/Icons';

export const Playground = ({ projectId, onPredictText, onPredictImage, busy, textPrediction, imagePrediction, hasVision }) => {
  const [tab, setTab] = useState('text');
  const [textInput, setTextInput] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleTextPredict = () => {
    if (textInput.trim()) {
      onPredictText(textInput);
    }
  };

  const handleImagePredict = () => {
    if (imageFile) {
      onPredictImage(imageFile);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setImageFile(e.target.files[0]);
    }
  };

  const renderResults = (prediction) => {
    if (!prediction) {
      return (
        <div className="empty-results">
          Awaiting inference request...
        </div>
      );
    }

    return (
      <div className="result-card">
        <div className="prediction-label-row">
          <span className="result-kicker">PREDICTED CLASS</span>
          <div className="prediction-label">
            {prediction.prediction || prediction.label || 'Unknown'}
          </div>
        </div>
        <div className="confidence-bar">
          <div
            className="confidence-value"
            style={{ width: `${(prediction.confidence || 0) * 100}%` }}
          ></div>
        </div>
        <span className="confidence-text">{((prediction.confidence || 0) * 100).toFixed(1)}% Confidence Score</span>

        {prediction.top_predictions && (
          <div className="top-predictions">
            <span className="predictions-header">CONFIDENCE DISTRIBUTION</span>
            {prediction.top_predictions.map((p, i) => (
              <div key={i} className="prediction-row">
                <span className="pred-name">{p.label}</span>
                <span className="pred-val">{((p.confidence || 0) * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        )}

        {prediction.extracted_text && (
          <div className="extracted-text-section">
            <h4>EXTRACTED OCR TEXT</h4>
            <p>{prediction.extracted_text}</p>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-header-title">
          <IconFlask size={16} />
          <span className="kicker">INFERENCE PLAYGROUND</span>
        </div>
      </div>

      {hasVision && (
        <div className="playground-tabs">
          <button
            className={`tab-btn ${tab === 'text' ? 'tab-btn-active' : ''}`}
            onClick={() => setTab('text')}
          >
            Text Classification
          </button>
          <button
            className={`tab-btn ${tab === 'image' ? 'tab-btn-active' : ''}`}
            onClick={() => setTab('image')}
          >
            Vision + OCR
          </button>
        </div>
      )}

      <div className="playground-layout">
        <div className="playground-form">
          {tab === 'text' || !hasVision ? (
            <>
              <textarea
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder="Enter sample input text to run classification inference..."
                rows={5}
                className="input-field"
              />
              <button
                className="btn-primary"
                onClick={handleTextPredict}
                disabled={busy || !textInput.trim()}
              >
                {busy ? (
                  <>
                    <IconRefresh size={14} className="icon-spin mr-1" />
                    <span>RUNNING INFERENCE...</span>
                  </>
                ) : (
                  <span>RUN PREDICTION</span>
                )}
              </button>
            </>
          ) : (
            <>
              <div
                className="file-upload-area"
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  type="file"
                  accept="image/*"
                  className="hidden-input"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                />
                <span className="file-upload-text">
                  {imageFile ? imageFile.name : 'Click to select document image'}
                </span>
              </div>
              <button
                className="btn-primary"
                onClick={handleImagePredict}
                disabled={busy || !imageFile}
              >
                {busy ? (
                  <>
                    <IconRefresh size={14} className="icon-spin mr-1" />
                    <span>EXTRACTING & PREDICTING...</span>
                  </>
                ) : (
                  <span>PREDICT FROM IMAGE</span>
                )}
              </button>
            </>
          )}
        </div>

        <div className="playground-results">
          {tab === 'text' || !hasVision ? renderResults(textPrediction) : renderResults(imagePrediction)}
        </div>
      </div>
    </div>
  );
};
