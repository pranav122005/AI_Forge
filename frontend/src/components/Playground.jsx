import React, { useState } from 'react';
import { Server, Play, Loader2, FileText, Image as ImageIcon, CheckCircle2, ScanText } from 'lucide-react';

export function Playground({ projectId, onPredictText, onPredictImage, busy, textPrediction, imagePrediction, hasVision }) {
  const [activeTab, setActiveTab] = useState('text');
  const [inputText, setInputText] = useState('The petitioner respectfully submits this petition before the competent court and seeks appropriate relief.');
  const [imageFile, setImageFile] = useState(null);

  function handleTextSubmit(e) {
    e.preventDefault();
    if (!inputText.trim() || busy) return;
    onPredictText(inputText.trim());
  }

  function handleImageSubmit(e) {
    e.preventDefault();
    if (!imageFile || busy) return;
    onPredictImage(imageFile);
  }

  return (
    <section className="panel api-panel">
      <div className="panel-head">
        <div>
          <div className="kicker">05 · LIVE MODEL PLAYGROUND & INFERENCE</div>
          <h2>Test Generated AI Model</h2>
        </div>
        <Server className="panel-icon" size={24} />
      </div>

      {hasVision && (
        <div className="playground-tabs">
          <button
            className={`tab-btn ${activeTab === 'text' ? 'active' : ''}`}
            onClick={() => setActiveTab('text')}
          >
            <FileText size={15} /> Text Classification
          </button>
          <button
            className={`tab-btn ${activeTab === 'image' ? 'active' : ''}`}
            onClick={() => setActiveTab('image')}
          >
            <ImageIcon size={15} /> Vision + OCR Pipeline
          </button>
        </div>
      )}

      {activeTab === 'text' ? (
        <div className="api-layout">
          <form className="playground-form" onSubmit={handleTextSubmit}>
            <div className="endpoint-badge">
              <span className="badge tone-purple">POST</span>
              <code>/api/predict</code>
            </div>
            <textarea
              rows={4}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Enter text to classify..."
              disabled={!!busy}
            />
            <button className="primary-btn" type="submit" disabled={!!busy || !inputText.trim()}>
              {busy === 'predictText' ? (
                <>
                  <Loader2 className="spin" size={16} /> Running Prediction…
                </>
              ) : (
                <>
                  Run Text Prediction <Play size={15} />
                </>
              )}
            </button>
          </form>

          <div className="result-card">
            {textPrediction ? (
              <div className="prediction-display">
                <div className="result-header">
                  <CheckCircle2 size={16} /> Inference Result
                </div>
                <div className="prediction-label">{textPrediction.prediction}</div>
                <div className="confidence-meter">
                  <span>Confidence Score</span>
                  <strong>{(textPrediction.confidence * 100).toFixed(1)}%</strong>
                </div>

                {textPrediction.top_predictions && (
                  <div className="top3-list">
                    <h4>Top Predictions</h4>
                    {textPrediction.top_predictions.map((p) => (
                      <div key={p.label} className="top3-row">
                        <span>{p.label}</span>
                        <span>{(p.confidence * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="empty-result">
                <FileText size={28} />
                <strong>Ready for Text Prediction</strong>
                <span>Train the model first, then submit a text input to receive real inference scores.</span>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="api-layout">
          <form className="playground-form" onSubmit={handleImageSubmit}>
            <div className="endpoint-badge">
              <span className="badge tone-purple">POST</span>
              <code>/api/predict-image</code>
            </div>
            <div className="file-input-wrapper">
              <input type="file" accept="image/*" onChange={(e) => setImageFile(e.target.files?.[0] || null)} />
              <div className="file-drop-info">
                <ImageIcon size={20} />
                <span>{imageFile ? imageFile.name : 'Choose a document image file (PNG / JPG / Scanned)'}</span>
              </div>
            </div>
            <button className="primary-btn" type="submit" disabled={!!busy || !imageFile}>
              {busy === 'predictImage' ? (
                <>
                  <Loader2 className="spin" size={16} /> Running Vision + OCR Pipeline…
                </>
              ) : (
                <>
                  Run Vision Pipeline <Play size={15} />
                </>
              )}
            </button>
          </form>

          <div className="result-card">
            {imagePrediction ? (
              <div className="prediction-display">
                <div className="result-header">
                  <ScanText size={16} /> Extracted Text & Classification
                </div>
                {imagePrediction.extracted_text && (
                  <div className="ocr-extracted-box">
                    <strong>Extracted Text via OCR:</strong>
                    <p>"{imagePrediction.extracted_text}"</p>
                  </div>
                )}
                {imagePrediction.prediction ? (
                  <>
                    <div className="prediction-label">{imagePrediction.prediction}</div>
                    <div className="confidence-meter">
                      <span>Confidence Score</span>
                      <strong>{(imagePrediction.confidence * 100).toFixed(1)}%</strong>
                    </div>
                  </>
                ) : (
                  <div className="empty-result">No text extracted from image.</div>
                )}
              </div>
            ) : (
              <div className="empty-result">
                <ImageIcon size={28} />
                <strong>Ready for Image Pipeline</strong>
                <span>Upload an image file to run OpenCV preprocessing, OCR text extraction, and classifier.</span>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
