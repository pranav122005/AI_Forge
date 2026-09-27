import React, { useState, useRef, useEffect } from 'react';
import { api } from '../../api/apiClient';
import {
  IconTerminal,
  IconCpu,
  IconSettings,
  IconZap,
  IconCheckCircle,
  IconXCircle,
  IconSearch,
  IconPackage,
  IconDownload,
  IconWorkspace,
  IconRefresh,
  IconClock,
} from '../common/Icons';

const SUGGESTIONS = [
  'Build a high-performance sentiment analysis API with FastAPI and scikit-learn',
  'Train a text classification model on support tickets with precision/recall metrics',
  'Create a modular REST API with JWT authentication, rate limiting, and pytest suite',
  'Build a document classifier with file upload, OCR parsing, and JSON endpoint',
];

const LIFECYCLE_STAGES = [
  { id: 'request', label: 'AI REQUEST' },
  { id: 'analyzing', label: 'REQUIREMENT ANALYSIS' },
  { id: 'planning', label: 'ARCHITECTURE PLANNING' },
  { id: 'generating', label: 'CODE GENERATION' },
  { id: 'testing', label: 'TEST EXECUTION' },
  { id: 'repair', label: 'AUTO REPAIR' },
  { id: 'verification', label: 'SECURITY VERIFICATION' },
  { id: 'packaging', label: 'PACKAGING' },
  { id: 'completed', label: 'COMPLETED' },
];

export function CommandCenter({
  onGenerate,
  onCodegen,
  busy,
  messages,
  activeProject,
  onViewWorkspace,
  activeProvider,
  activeModel,
  onProviderChange,
  onViewProviders,
}) {
  const [inputText, setInputText] = useState('');
  const chatEndRef = useRef(null);

  const providerNames = {
    gemini: 'Google Gemini',
    openai: 'OpenAI',
    anthropic: 'Anthropic Claude',
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  function handleSubmit(e) {
    if (e) e.preventDefault();
    if (!inputText.trim() || busy) return;
    onCodegen(inputText.trim());
    setInputText('');
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  const renderEngineSelector = () => (
    <div className="engine-selector-bar">
      <div className="engine-selector-info">
        <span className="engine-selector-label">ACTIVE ENGINE:</span>
        <button
          type="button"
          className="engine-pill-btn"
          onClick={onViewProviders}
          title="Switch or configure LLM providers"
        >
          <IconCpu size={14} className="engine-icon" />
          <span className="engine-name">{providerNames[activeProvider] || activeProvider?.toUpperCase() || 'Google Gemini'}</span>
          <span className="engine-model-sub">[ {activeModel || 'gemini-3.8-flash'} ]</span>
          <IconSettings size={13} className="engine-gear-icon" />
        </button>
      </div>
      <div className="engine-status-ready">
        <span className="status-dot status-dot-online" />
        <span className="status-text">ONLINE</span>
      </div>
    </div>
  );

  /* ── Hero (empty state) ── */
  if (!messages || messages.length === 0) {
    return (
      <div className="command-center">
        <div className="command-hero">
          <div className="command-hero-badge">
            <span className="hero-dot"></span>
            <span>AUTONOMOUS SCIENTIFIC AI ENGINEERING WORKSTATION</span>
          </div>
          <h1 className="command-title">AIForge</h1>
          <h2 className="command-subtitle">Agentic Model Training & Production Code Synthesis</h2>
          <p className="command-tagline">
            Provide system specifications in natural language. AIForge plans architecture, selects dependencies, writes full multi-file codebases, executes automated test suites, repairs defects, and verifies security sandboxing.
          </p>
        </div>

        <div className="command-input-area">
          {renderEngineSelector()}
          <textarea
            className="command-textarea"
            placeholder="Describe your AI system requirements (e.g. Build a sentiment analysis API with FastAPI, text preprocessing, ML model training, evaluation metrics, and unit tests)..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!!busy}
            rows={4}
            aria-label="Project requirement input"
          />
          <div className="command-actions">
            <div className="command-input-hint">
              <span>Press <kbd>Enter</kbd> to execute, <kbd>Shift + Enter</kbd> for new line</span>
            </div>
            <button className="btn-primary" onClick={handleSubmit} disabled={!!busy || !inputText.trim()}>
              {busy ? (
                <>
                  <IconRefresh size={14} className="icon-spin" />
                  <span>SYNTHESIZING...</span>
                </>
              ) : (
                <>
                  <IconZap size={14} />
                  <span>GENERATE PROJECT</span>
                </>
              )}
            </button>
          </div>
        </div>

        <div className="suggestion-container">
          <div className="suggestion-header">
            <IconTerminal size={13} />
            <span>PRESET SYSTEM ARCHITECTURES</span>
          </div>
          <div className="suggestion-chips">
            {SUGGESTIONS.map((s) => (
              <button key={s} className="suggestion-chip" onClick={() => setInputText(s)} disabled={!!busy}>
                <span className="chip-prefix">&gt;</span>
                <span className="chip-text">{s}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* ── Chat stream ── */
  return (
    <div className="command-center command-center-chat">
      <div className="chat-stream">
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-message ${msg.role === 'user' ? 'chat-message-user' : 'chat-message-ai'}`}>
            {msg.role === 'ai' && (
              <div className="chat-avatar">
                <IconTerminal size={14} />
              </div>
            )}
            <div className="chat-bubble">
              {msg.role === 'ai' && msg.stage && (
                <div className={`chat-stage stage-${msg.stage}`}>
                  {msg.stage === 'analyzing' && (
                    <>
                      <IconSearch size={13} />
                      <span>ANALYZING REQUIREMENTS...</span>
                    </>
                  )}
                  {msg.stage === 'generating' && (
                    <>
                      <IconZap size={13} className="icon-pulse" />
                      <span>SYNTHESIZING ARCHITECTURE & CODE...</span>
                    </>
                  )}
                  {msg.stage === 'completed' && (
                    <>
                      <IconCheckCircle size={13} className="text-green" />
                      <span>EXECUTION COMPLETED</span>
                    </>
                  )}
                  {msg.stage === 'failed' && (
                    <>
                      <IconXCircle size={13} className="text-red" />
                      <span>EXECUTION FAILED</span>
                    </>
                  )}
                </div>
              )}

              <div className="chat-content">{msg.content}</div>

              {msg.role === 'ai' && msg.stage === 'completed' && msg.result && (
                <div className="chat-result-card">
                  <div className="chat-result-meta">
                    <div className="meta-item">
                      <span className="meta-label">PROJECT ID:</span>
                      <code className="meta-value">{msg.result.project_id}</code>
                    </div>
                    <div className="meta-item">
                      <span className="meta-label">FILES GENERATED:</span>
                      <span className="meta-value">{msg.result.generated_files?.length || 0} files</span>
                    </div>
                    <div className="meta-item">
                      <span className="meta-label">LIFECYCLE STATUS:</span>
                      <span className="meta-value status-badge">{msg.result.status || 'READY'}</span>
                    </div>
                    {msg.result.test_result && (
                      <div className="meta-item">
                        <span className="meta-label">UNIT TESTS:</span>
                        <span className={`meta-value ${msg.result.test_result.success ? 'text-green' : 'text-red'}`}>
                          {msg.result.test_result.success ? 'PASS (All Passed)' : 'FAIL (Needs Repair)'}
                        </span>
                      </div>
                    )}
                  </div>
                  <div className="chat-result-actions">
                    <button className="btn-primary" onClick={onViewWorkspace}>
                      <IconWorkspace size={14} />
                      <span>OPEN WORKSPACE</span>
                    </button>
                    {msg.result.zip_path && (
                      <a
                        className="btn-secondary"
                        href={api.getCodegenDownloadUrl(msg.result.project_id)}
                        download
                      >
                        <IconDownload size={14} />
                        <span>DOWNLOAD ZIP</span>
                      </a>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      <div className="command-input-area command-input-bottom">
        {renderEngineSelector()}
        <textarea
          className="command-textarea"
          placeholder="Enter follow-up instructions, architectural additions, or modifications..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={!!busy}
          rows={2}
          aria-label="Follow-up input"
        />
        <div className="command-actions">
          <button className="btn-primary" onClick={handleSubmit} disabled={!!busy || !inputText.trim()}>
            {busy ? (
              <>
                <IconRefresh size={14} className="icon-spin" />
                <span>PROCESSING...</span>
              </>
            ) : (
              <>
                <IconZap size={14} />
                <span>SUBMIT COMMAND</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
