import React from 'react';
import { Play, Loader2, CheckCircle2, XCircle, Terminal, Layers, Box } from 'lucide-react';

export function AgentRuntimeView({ projectId, onRunAgent, busy, agentResult }) {
  return (
    <section className="panel agent-panel">
      <div className="panel-head">
        <div>
          <div className="kicker">06 · EXECUTABLE AGENT RUNTIME</div>
          <h2>Agent Execution Engine</h2>
        </div>
        <Terminal className="panel-icon" size={24} />
      </div>

      <div className="agent-trigger-card">
        <div>
          <h3>Execute Registered Agent Pipeline</h3>
          <p>Runs the generated project pipeline steps sequentially through the backend AgentRuntime.</p>
        </div>
        <button
          className="primary-btn"
          onClick={() => onRunAgent('Run agent pipeline with default test payload')}
          disabled={!!busy}
        >
          {busy === 'agent' ? (
            <>
              <Loader2 className="spin" size={16} /> Running Agent Pipeline…
            </>
          ) : (
            <>
              <Play size={16} /> Execute Agent Pipeline
            </>
          )}
        </button>
      </div>

      {agentResult && (
        <div className="agent-result-container">
          <div className="result-status-header">
            <div className="status-title">
              {agentResult.status === 'completed' ? (
                <span className="badge tone-green"><CheckCircle2 size={14} /> Pipeline Execution Completed</span>
              ) : (
                <span className="badge tone-red"><XCircle size={14} /> Pipeline Execution Failed</span>
              )}
            </div>
            {agentResult.error && (
              <div className="error-banner">
                <strong>Failed Component ({agentResult.failed_component}):</strong> {agentResult.error}
              </div>
            )}
          </div>

          {agentResult.steps && agentResult.steps.length > 0 && (
            <div className="step-logs-card">
              <h4><Layers size={16} /> Execution Step Logs</h4>
              <div className="log-table">
                {agentResult.steps.map((step, idx) => (
                  <div key={idx} className={`log-row ${step.status}`}>
                    <span className="log-idx">Step {idx + 1}</span>
                    <strong className="log-comp">{step.component}</strong>
                    <span className={`badge ${step.status === 'completed' ? 'tone-green' : 'tone-red'}`}>
                      {step.status.toUpperCase()}
                    </span>
                    {step.error && <span className="log-err">{step.error}</span>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {agentResult.result && (
            <div className="artifacts-card">
              <h4><Box size={16} /> Output Artifacts</h4>
              <pre className="json-artifact-view">
                {JSON.stringify(agentResult.result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
