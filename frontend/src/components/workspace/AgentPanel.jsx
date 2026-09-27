import React, { useState, useEffect, useRef } from 'react';
import { api } from '../../api/apiClient';
import {
  IconBot,
  IconCpu,
  IconCheckCircle,
  IconXCircle,
  IconAlertTriangle,
  IconRefresh,
  IconDownload,
  IconEdit,
  IconTerminal,
  IconClock,
  IconCheck,
} from '../common/Icons';

export function AgentPanel({
  projectId,
  agentRunState,
  onAgentRunStateChange,
  busy,
  onBusyChange,
  onRefreshFiles,
  onError,
  activeProvider,
  activeModel,
}) {
  const [instruction, setInstruction] = useState('');
  const [modifyInstruction, setModifyInstruction] = useState('');
  const [modifyResult, setModifyResult] = useState(null);
  const [modifyProgress, setModifyProgress] = useState(0);
  const [eventLogs, setEventLogs] = useState([]);
  const pollRef = useRef(null);

  const presets = [
    'Add JWT authentication and protected routes',
    'Add Dockerfile and docker-compose deployment',
    'Add /health and readiness check endpoints',
    'Add comprehensive pytest unit test coverage',
  ];

  const formatTimestamp = () => {
    const d = new Date();
    return d.toTimeString().split(' ')[0];
  };

  const addLog = (text, type = 'info') => {
    setEventLogs(prev => [...prev.slice(-49), { time: formatTimestamp(), text, type, id: Math.random() }]);
  };

  function formatErrorMessage(msg) {
    if (!msg) return 'Unknown error occurred.';
    const s = String(msg).toLowerCase();
    if (s.includes('429') || s.includes('quota') || s.includes('rate limit'))
      return 'API Rate Limit / Quota Exceeded. Please wait before retrying.';
    if (s.includes('504') || s.includes('timeout'))
      return 'Subprocess execution timed out safely.';
    if (s.includes('rolled_back') || s.includes('restored'))
      return 'Agent modification failed; project state safely restored.';
    return String(msg);
  }

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  /* ── Agent Run ── */
  async function handleRunAgent(e) {
    e.preventDefault();
    if (!instruction.trim() || busy) return;

    onBusyChange('agent');
    onAgentRunStateChange(null);
    setModifyResult(null);
    addLog(`Initiating autonomous agent run for: "${instruction}"`, 'start');

    try {
      const state = await api.runCodegenAgent(projectId, instruction, activeProvider, activeModel);
      onAgentRunStateChange(state);
      addLog(`Agent initialized with run ID: ${state.run_id}`, 'info');

      if (state.error) {
        const formattedErr = formatErrorMessage(state.error);
        onError(formattedErr);
        addLog(`Agent error: ${formattedErr}`, 'error');
      }

      if (!['COMPLETED', 'FAILED', 'ROLLED_BACK'].includes(state.status)) {
        startPolling(state.run_id);
      } else {
        onBusyChange('');
        onRefreshFiles();
        addLog(`Agent completed with status: ${state.status}`, state.status === 'COMPLETED' ? 'success' : 'error');
      }
    } catch (err) {
      const formattedErr = formatErrorMessage(err?.message || err);
      onError(formattedErr);
      addLog(`Agent failed: ${formattedErr}`, 'error');
      onBusyChange('');
    }
  }

  function startPolling(runId) {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const state = await api.getCodegenAgentState(projectId, runId);
        onAgentRunStateChange(state);

        if (state.stage) {
          addLog(`Agent stage update: ${state.stage}`, 'stage');
        }

        if (state.error) {
          const formattedErr = formatErrorMessage(state.error);
          onError(formattedErr);
          addLog(`Error detected: ${formattedErr}`, 'error');
        }

        if (['COMPLETED', 'FAILED', 'ROLLED_BACK'].includes(state.status)) {
          clearInterval(pollRef.current);
          pollRef.current = null;
          onRefreshFiles();
          onBusyChange('');
          addLog(`Agent finished with status: ${state.status}`, state.status === 'COMPLETED' ? 'success' : 'error');
        }
      } catch (err) {
        clearInterval(pollRef.current);
        pollRef.current = null;
        const formattedErr = formatErrorMessage(err?.message || err);
        onError(formattedErr);
        addLog(`Polling failed: ${formattedErr}`, 'error');
        onBusyChange('');
      }
    }, 1500);
  }

  /* ── Modify ── */
  async function handleModify(e) {
    e.preventDefault();
    if (!modifyInstruction.trim() || busy) return;

    onBusyChange('modify');
    setModifyProgress(1);
    setModifyResult(null);
    addLog(`Running project modification: "${modifyInstruction}"`, 'start');

    const timer = setInterval(() => {
      setModifyProgress(p => (p < 4 ? p + 1 : p));
    }, 1500);

    try {
      const res = await api.modifyCodegenProject(projectId, modifyInstruction, activeProvider, activeModel);
      clearInterval(timer);
      setModifyProgress(0);
      setModifyResult(res);
      onRefreshFiles();
      addLog(`Modification completed: ${res.status}`, res.status === 'SUCCESS' ? 'success' : 'error');
    } catch (err) {
      clearInterval(timer);
      setModifyProgress(0);
      const formattedErr = formatErrorMessage(err?.message || err);
      onError(formattedErr);
      addLog(`Modification error: ${formattedErr}`, 'error');
    } finally {
      onBusyChange('');
    }
  }

  function getTestSuccess(tr) {
    if (!tr) return false;
    return tr.success === true || tr.success === 'true';
  }

  const statusClass = (status) => {
    if (status === 'COMPLETED') return 'tone-green';
    if (status === 'FAILED' || status === 'ROLLED_BACK') return 'tone-red';
    return 'tone-cyan';
  };

  const stagesList = [
    { key: 'analyzed', label: 'ANALYSIS', done: Boolean(agentRunState) },
    { key: 'planning', label: 'PLANNING', done: Boolean(agentRunState?.plan) },
    { key: 'executing', label: 'EXECUTION', done: ((agentRunState?.files_created?.length || 0) + (agentRunState?.files_modified?.length || 0)) > 0 },
    { key: 'testing', label: 'TESTING', done: getTestSuccess(agentRunState?.test_result) },
    { key: 'verification', label: 'VERIFICATION', done: Boolean(agentRunState?.verification_result?.success) },
    { key: 'packaging', label: 'PACKAGING', done: Boolean(agentRunState?.zip_available) },
  ];

  return (
    <div className="agent-panel">
      {/* Header */}
      <div className="agent-header">
        <div className="agent-title-group">
          <IconBot size={16} className="agent-bot-icon" />
          <div>
            <h3 className="agent-title">AI ENGINEERING AGENT</h3>
            <span className="agent-subtitle">AUTONOMOUS DEVELOPMENT LOOP</span>
          </div>
        </div>
        <div className="agent-engine-tag">
          <IconCpu size={12} />
          <span>{(activeProvider || 'GEMINI').toUpperCase()}</span>
        </div>
      </div>

      <p className="agent-desc">
        Direct autonomous coding loop. Inspects workspace, designs implementation plans, generates code, runs test suites, repairs failures, and creates verified packages.
      </p>

      {/* Presets */}
      <div className="agent-presets">
        <div className="preset-label">QUICK OBJECTIVES:</div>
        <div className="preset-list">
          {presets.map(p => (
            <button key={p} className="preset-chip" onClick={() => setInstruction(p)} disabled={!!busy}>
              <span className="preset-arrow">&gt;</span>
              <span>{p}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Agent Input */}
      <form className="agent-form" onSubmit={handleRunAgent}>
        <div className="agent-input-wrapper">
          <input
            type="text"
            className="agent-input"
            placeholder="e.g. Add JWT authentication middleware and token verification..."
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            disabled={!!busy}
          />
        </div>
        <button type="submit" className="btn-primary agent-submit" disabled={!!busy || !instruction.trim()}>
          {busy === 'agent' ? (
            <>
              <IconRefresh size={14} className="icon-spin" />
              <span>RUNNING AGENT...</span>
            </>
          ) : (
            <>
              <IconBot size={14} />
              <span>EXECUTE AGENT</span>
            </>
          )}
        </button>
      </form>

      {/* ── Agent Results & Telemetry ── */}
      {agentRunState && (
        <div className="agent-results">
          <div className="agent-status-header">
            <div className="status-indicator-box">
              <span className={`status-pill ${statusClass(agentRunState.status)}`}>
                {agentRunState.status}
              </span>
            </div>
            <code className="agent-run-id">RUN: {agentRunState.run_id}</code>
          </div>

          {/* Live Stage Tracker */}
          <div className="agent-stage-tracker">
            <div className="tracker-title">PIPELINE EXECUTION STAGES:</div>
            <div className="tracker-steps">
              {stagesList.map((st) => (
                <div key={st.key} className={`tracker-step ${st.done ? 'step-done' : 'step-pending'}`}>
                  <span className="step-bullet">{st.done ? '●' : '○'}</span>
                  <span className="step-name">{st.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Structured Telemetry Grid */}
          <div className="agent-telemetry-grid">
            <div className="telemetry-card">
              <span className="telemetry-label">CREATED</span>
              <span className="telemetry-value text-green">+{agentRunState.files_created?.length || 0}</span>
            </div>
            <div className="telemetry-card">
              <span className="telemetry-label">MODIFIED</span>
              <span className="telemetry-value text-cyan">~{agentRunState.files_modified?.length || 0}</span>
            </div>
            <div className="telemetry-card">
              <span className="telemetry-label">REPAIRS</span>
              <span className="telemetry-value">{agentRunState.repair_attempts || 0}</span>
            </div>
            <div className="telemetry-card">
              <span className="telemetry-label">TESTS</span>
              <span className={`telemetry-value ${getTestSuccess(agentRunState.test_result) ? 'text-green' : 'text-amber'}`}>
                {getTestSuccess(agentRunState.test_result) ? 'PASS' : 'PENDING'}
              </span>
            </div>
          </div>

          {/* Plan Section */}
          {agentRunState.plan && (
            <div className="agent-plan">
              <div className="plan-summary">
                <IconTerminal size={13} />
                <span>PLAN: {agentRunState.plan.summary} ({agentRunState.plan.steps?.length || 0} steps)</span>
              </div>
              <div className="plan-steps-list">
                {(agentRunState.plan.steps || []).map((step, i) => (
                  <div key={i} className="plan-step">
                    <span className="plan-step-id">[{step.id}]</span>
                    <span className="plan-step-action">{step.action}</span>
                    <span className="plan-step-desc">{step.description}</span>
                    {step.files?.length > 0 && (
                      <span className="plan-step-files">({step.files.join(', ')})</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Verification + Download */}
          <div className={`agent-verification ${agentRunState.verification_result?.success ? 'verification-pass' : 'verification-fail'}`}>
            <div className="verification-label">
              {agentRunState.verification_result?.success ? (
                <>
                  <IconCheckCircle size={14} className="text-green" />
                  <span>VERIFICATION: PASSED</span>
                </>
              ) : (
                <>
                  <IconAlertTriangle size={14} className="text-amber" />
                  <span>VERIFICATION: {agentRunState.verification_result?.status || 'PENDING'}</span>
                </>
              )}
            </div>
            {agentRunState.zip_available && (
              <a
                className="btn-secondary agent-download-btn"
                href={api.getCodegenDownloadUrl(projectId)}
                download
              >
                <IconDownload size={13} />
                <span>DOWNLOAD ZIP</span>
              </a>
            )}
          </div>
        </div>
      )}

      {/* ── Live Event Stream ── */}
      <div className="agent-event-stream">
        <div className="stream-header">
          <IconClock size={12} />
          <span>LIVE AGENT EVENT STREAM</span>
        </div>
        <div className="stream-logs">
          {eventLogs.length === 0 ? (
            <div className="stream-empty">Awaiting agent commands...</div>
          ) : (
            eventLogs.map((log) => (
              <div key={log.id} className={`stream-log-entry log-${log.type}`}>
                <span className="log-time">[{log.time}]</span>
                <span className="log-text">{log.text}</span>
              </div>
            ))
          )}
        </div>
      </div>

      {/* ── Modify Section ── */}
      <div className="agent-divider" />

      <div className="modify-section">
        <div className="modify-header">
          <IconEdit size={14} />
          <h4>DIRECT PROJECT MODIFIER</h4>
        </div>
        <form className="modify-form" onSubmit={handleModify}>
          <input
            type="text"
            className="modify-input"
            placeholder="e.g. Replace LogisticRegression with RandomForestClassifier"
            value={modifyInstruction}
            onChange={(e) => setModifyInstruction(e.target.value)}
            disabled={!!busy}
          />
          <button type="submit" className="btn-secondary modify-btn" disabled={!!busy || !modifyInstruction.trim()}>
            {busy === 'modify' ? (
              <>
                <IconRefresh size={13} className="icon-spin" />
                <span>MODIFYING...</span>
              </>
            ) : (
              <>
                <IconEdit size={13} />
                <span>APPLY CHANGE</span>
              </>
            )}
          </button>
        </form>

        {modifyProgress > 0 && (
          <div className="modify-progress">
            {['ANALYZING', 'LOCATING', 'APPLYING', 'VERIFYING'].map((label, i) => (
              <div key={label} className={`progress-step ${modifyProgress > i ? 'progress-step-active' : ''}`}>
                <span className="step-num">{i + 1}</span>
                <span>{label}</span>
              </div>
            ))}
          </div>
        )}

        {modifyResult && (
          <div className="agent-results">
            <div className={`agent-verification ${modifyResult.status === 'SUCCESS' ? 'verification-pass' : 'verification-fail'}`}>
              <span>STATUS: {modifyResult.status}</span>
            </div>
            <div className="modify-summary">{modifyResult.summary}</div>
          </div>
        )}
      </div>
    </div>
  );
}
