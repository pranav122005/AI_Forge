import React, { useState, useEffect } from 'react';
import { api } from '../../api/apiClient';
import {
  IconCpu,
  IconSettings,
  IconCheckCircle,
  IconXCircle,
  IconRefresh,
  IconEye,
  IconLock,
  IconZap,
  IconCheck,
  IconClose,
} from '../common/Icons';

export function ProviderControlCenter({ activeProvider, activeModel, onProviderChanged, onError }) {
  const [providers, setProviders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [testingProvider, setTestingProvider] = useState('');
  const [testResults, setTestResults] = useState({});
  const [modalProvider, setModalProvider] = useState(null);
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [modelInput, setModelInput] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState('');

  const providerDescriptions = {
    gemini: 'Google Gemini 2.5/3.8 Flash & Pro models with ultra-fast structured JSON generation and reasoning.',
    openai: 'OpenAI GPT-4o, GPT-4o-mini, and GPT-5 models with advanced codegen and reasoning capabilities.',
    anthropic: 'Anthropic Claude 3.5/3.7 Sonnet & Haiku models with industry-leading code architecture and refactoring.',
  };

  async function loadProviders() {
    try {
      setLoading(true);
      const list = await api.getLLMProviders();
      setProviders(list);
    } catch (err) {
      if (onError) onError(err.message || 'Failed to load LLM providers.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadProviders();
  }, []);

  async function handleSetActive(providerId, model) {
    try {
      await api.setLLMActive(providerId, model);
      if (onProviderChanged) onProviderChanged(providerId, model);
      await loadProviders();
    } catch (err) {
      if (onError) onError(err.message || 'Failed to activate provider.');
    }
  }

  async function handleTest(providerId, key = null, model = null) {
    setTestingProvider(providerId);
    try {
      const payload = {};
      if (key) payload.api_key = key;
      if (model) payload.model = model;
      const res = await api.testLLMProvider(providerId, payload);
      setTestResults(prev => ({ ...prev, [providerId]: res }));
    } catch (err) {
      setTestResults(prev => ({
        ...prev,
        [providerId]: { status: 'FAILED', message: err.message || 'Test failed' },
      }));
    } finally {
      setTestingProvider('');
    }
  }

  function openConfigureModal(p) {
    setModalProvider(p);
    setApiKeyInput('');
    setModelInput(p.default_model || (p.models && p.models[0]) || '');
    setShowKey(false);
    setSaveSuccess('');
  }

  function closeConfigureModal() {
    setModalProvider(null);
    setApiKeyInput('');
    setModelInput('');
    setSaveSuccess('');
  }

  async function handleSaveConfig(e) {
    if (e) e.preventDefault();
    if (!modalProvider) return;
    setSaving(true);
    setSaveSuccess('');
    try {
      const config = { model: modelInput };
      if (apiKeyInput.trim()) config.api_key = apiKeyInput.trim();
      await api.configureLLMProvider(modalProvider.id, config);
      setSaveSuccess('Configuration saved successfully!');
      await loadProviders();
      if (modalProvider.id === activeProvider && onProviderChanged) {
        onProviderChanged(modalProvider.id, modelInput);
      }
      setTimeout(() => {
        closeConfigureModal();
      }, 1000);
    } catch (err) {
      if (onError) onError(err.message || 'Failed to save configuration.');
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="provider-control-center">
      <div className="provider-header">
        <div className="provider-header-title">
          <div className="provider-badge-kicker">
            <IconCpu size={14} />
            <span>NEURAL BACKEND MANAGEMENT</span>
          </div>
          <h2>AI Provider Control Center</h2>
          <p>
            Connect and manage multiple LLM backends (Google Gemini, OpenAI, Anthropic Claude).
            Select default models, test live connectivity, and switch active engines seamlessly.
          </p>
        </div>
        <div className="active-engine-badge">
          <span className="status-dot status-dot-online"></span>
          <span className="active-label">ACTIVE ENGINE:</span>
          <strong>{activeProvider?.toUpperCase() || 'GEMINI'}</strong>
          <span className="active-model-tag">{activeModel || 'gemini-3.8-flash'}</span>
        </div>
      </div>

      {loading ? (
        <div className="loading-state">
          <IconRefresh size={24} className="icon-spin" />
          <p>Scanning provider configurations...</p>
        </div>
      ) : (
        <div className="provider-grid">
          {providers.map((p) => {
            const isActive = p.active || p.id === activeProvider;
            const isConfigured = p.configured;
            const testResult = testResults[p.id];
            const desc = providerDescriptions[p.id] || 'Production LLM Provider.';

            return (
              <div key={p.id} className={`provider-card ${isActive ? 'provider-card-active' : ''} ${isConfigured ? 'provider-card-configured' : 'provider-card-unconfigured'}`}>
                <div className="provider-card-header">
                  <div className="provider-title-group">
                    <div className="provider-avatar-box">
                      <IconCpu size={18} />
                    </div>
                    <div>
                      <h3 className="provider-name">{p.name}</h3>
                      <span className="provider-id-tag">ID: {p.id}</span>
                    </div>
                  </div>
                  <div className="provider-badges">
                    {isActive && <span className="badge-active">ACTIVE ENGINE</span>}
                    <span className={`badge-status ${isConfigured ? 'status-connected' : 'status-unconfigured'}`}>
                      {isConfigured ? '● CONNECTED' : '○ NOT CONFIGURED'}
                    </span>
                  </div>
                </div>

                <p className="provider-desc">{desc}</p>

                <div className="provider-model-select-group">
                  <label htmlFor={`model-select-${p.id}`}>SELECTED MODEL:</label>
                  <select
                    id={`model-select-${p.id}`}
                    className="provider-model-select"
                    value={p.default_model}
                    onChange={(e) => {
                      const newModel = e.target.value;
                      api.configureLLMProvider(p.id, { model: newModel }).then(() => {
                        loadProviders();
                        if (isActive && onProviderChanged) onProviderChanged(p.id, newModel);
                      });
                    }}
                  >
                    {p.models?.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>

                {testResult && (
                  <div className={`provider-test-result ${testResult.status === 'SUCCESS' ? 'test-pass' : 'test-fail'}`}>
                    <div className="test-status-line">
                      <div className="test-status-indicator">
                        {testResult.status === 'SUCCESS' ? <IconCheckCircle size={13} className="text-green" /> : <IconXCircle size={13} className="text-red" />}
                        <strong>{testResult.status === 'SUCCESS' ? 'CONNECTED' : 'FAILED'}</strong>
                      </div>
                      {testResult.latency_ms && <span>{testResult.latency_ms}ms</span>}
                    </div>
                    <div className="test-message">{testResult.message}</div>
                  </div>
                )}

                <div className="provider-card-actions">
                  {!isActive && isConfigured && (
                    <button
                      className="btn-primary btn-sm"
                      onClick={() => handleSetActive(p.id, p.default_model)}
                    >
                      <IconZap size={12} className="mr-1" />
                      Set Active
                    </button>
                  )}
                  {isActive && (
                    <button className="btn-secondary btn-sm" disabled>
                      <IconCheck size={12} className="mr-1" />
                      Active
                    </button>
                  )}
                  <button
                    className="btn-secondary btn-sm"
                    onClick={() => openConfigureModal(p)}
                  >
                    <IconSettings size={12} className="mr-1" />
                    Configure Key
                  </button>
                  <button
                    className="btn-ghost btn-sm"
                    onClick={() => handleTest(p.id)}
                    disabled={testingProvider === p.id}
                  >
                    {testingProvider === p.id ? (
                      <>
                        <IconRefresh size={12} className="icon-spin mr-1" />
                        Testing...
                      </>
                    ) : (
                      <>
                        <IconZap size={12} className="mr-1" />
                        Test
                      </>
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Configure Modal */}
      {modalProvider && (
        <div className="modal-backdrop" onClick={closeConfigureModal}>
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-group">
                <IconSettings size={18} className="modal-icon" />
                <div>
                  <h3>Configure {modalProvider.name}</h3>
                  <p className="modal-subtitle">Provider ID: <code>{modalProvider.id}</code></p>
                </div>
              </div>
              <button className="modal-close" onClick={closeConfigureModal}>
                <IconClose size={16} />
              </button>
            </div>

            <form onSubmit={handleSaveConfig} className="modal-form">
              <div className="form-group">
                <label>API Key</label>
                <div className="input-password-wrapper">
                  <input
                    type={showKey ? 'text' : 'password'}
                    className="form-control"
                    placeholder={modalProvider.configured ? '●●●●●●●● (Configured - enter new key to replace)' : 'Enter API Key (sk-...)'}
                    value={apiKeyInput}
                    onChange={(e) => setApiKeyInput(e.target.value)}
                    autoComplete="off"
                  />
                  <button
                    type="button"
                    className="btn-toggle-key"
                    onClick={() => setShowKey(!showKey)}
                    title={showKey ? 'Hide Key' : 'Show Key'}
                  >
                    <IconEye size={14} />
                  </button>
                </div>
                <small className="form-hint">
                  <IconLock size={12} className="mr-1" />
                  Keys are stored in server memory. They are never written to disk, committed, or exposed.
                </small>
              </div>

              <div className="form-group">
                <label>Default Model</label>
                <select
                  className="form-control"
                  value={modelInput}
                  onChange={(e) => setModelInput(e.target.value)}
                >
                  {modalProvider.models?.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>

              {saveSuccess && (
                <div className="alert-success">
                  <IconCheckCircle size={13} className="mr-1" />
                  {saveSuccess}
                </div>
              )}

              <div className="modal-actions">
                <button
                  type="button"
                  className="btn-ghost"
                  onClick={() => handleTest(modalProvider.id, apiKeyInput || null, modelInput || null)}
                  disabled={testingProvider === modalProvider.id}
                >
                  {testingProvider === modalProvider.id ? 'Testing...' : 'Test Connection'}
                </button>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={closeConfigureModal}
                  disabled={saving}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn-primary"
                  disabled={saving}
                >
                  {saving ? 'Saving...' : 'Save Settings'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
