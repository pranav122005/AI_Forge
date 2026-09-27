import React, { useState, useEffect } from 'react';
import { api } from '../api/apiClient';

export function CodegenExplorer({ projectId, initialCodegenResult, requirementText, onGenerateSuccess, metrics }) {
  const [codegenResult, setCodegenResult] = useState(initialCodegenResult || null);
  const [modifyResult, setModifyResult] = useState(null);
  const [fileTree, setFileTree] = useState(null);
  const [selectedFilePath, setSelectedFilePath] = useState('');
  const [fileContent, setFileContent] = useState('');
  const [loadingFile, setLoadingFile] = useState(false);
  const [testResult, setTestResult] = useState(initialCodegenResult?.test_result || null);
  const [repairResult, setRepairResult] = useState(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [userFeedback, setUserFeedback] = useState('');
  const [modifyInstruction, setModifyInstruction] = useState('');
  const [agentInstruction, setAgentInstruction] = useState('');
  const [agentRunState, setAgentRunState] = useState(null);
  const [llmStatus, setLlmStatus] = useState(null);
  const [progressStep, setProgressStep] = useState(0);
  const [copied, setCopied] = useState(false);
  const [projectHistory, setProjectHistory] = useState([]);

  useEffect(() => {
    async function loadLlmStatus() {
      try {
        const cap = await api.getCapabilities();
        if (cap?.llm) {
          setLlmStatus(cap.llm);
        }
      } catch (err) {
        console.error('Failed to load LLM capabilities:', err);
      }
    }
    loadLlmStatus();
  }, []);

  useEffect(() => {
    if (projectId) {
      loadProjectFiles(projectId);
    }
  }, [projectId]);

  async function loadProjectFiles(id) {
    try {
      const tree = await api.getCodegenFiles(id);
      setFileTree(tree);
      const firstFile = findFirstFile(tree);
      if (firstFile) {
        handleFileSelect(firstFile.path);
      }
    } catch (err) {
      console.error('Failed to load file tree:', err);
    }
  }

  function findFirstFile(node) {
    if (!node) return null;
    if (!node.is_dir && node.path !== '.') return node;
    if (node.children) {
      for (const child of node.children) {
        const found = findFirstFile(child);
        if (found) return found;
      }
    }
    return null;
  }

  async function handleFileSelect(path) {
    if (!path || path === '.') return;
    setSelectedFilePath(path);
    setLoadingFile(true);
    try {
      const content = await api.getCodegenFileContent(projectId, path);
      setFileContent(content);
    } catch (err) {
      setFileContent(`// Error reading file: ${err.message}`);
    } finally {
      setLoadingFile(false);
    }
  }

  async function handleTriggerCodegen() {
    if (!requirementText) return;
    setBusy('generating');
    setError('');
    try {
      const res = await api.generateCodegen(requirementText);
      setCodegenResult(res);
      setTestResult(res.test_result);
      if (res.project_id) {
        loadProjectFiles(res.project_id);
        if (onGenerateSuccess) onGenerateSuccess(res);
      }
    } catch (err) {
      setError(err.message || 'Failed to generate code.');
    } finally {
      setBusy('');
    }
  }

  async function handleRunTests() {
    if (!projectId) return;
    setBusy('testing');
    setError('');
    try {
      const res = await api.runCodegenTests(projectId);
      setTestResult(res);
    } catch (err) {
      setError(err.message || 'Failed to run tests.');
    } finally {
      setBusy('');
    }
  }

  async function handleTriggerRepair() {
    if (!projectId) return;
    setBusy('repairing');
    setError('');
    try {
      const res = await api.repairCodegenProject(projectId, testResult?.stderr || '', userFeedback);
      setRepairResult(res);
      if (res.test_result) setTestResult(res.test_result);
      loadProjectFiles(projectId);
    } catch (err) {
      setError(err.message || 'Repair loop failed.');
    } finally {
      setBusy('');
    }
  }

  async function handleModifyProject(e) {
    if (e) e.preventDefault();
    if (!modifyInstruction.trim() || !projectId) return;

    setBusy('modifying');
    setError('');
    setProgressStep(1);

    const stepTimer = setInterval(() => {
      setProgressStep((prev) => (prev < 4 ? prev + 1 : prev));
    }, 1200);

    try {
      const res = await api.modifyCodegenProject(projectId, modifyInstruction);
      setModifyResult(res);
      if (res.history) setProjectHistory(res.history);
      if (res.test_result) setTestResult(res.test_result);
      loadProjectFiles(projectId);
    } catch (err) {
      setError(err.message || 'Failed to modify project.');
    } finally {
      clearInterval(stepTimer);
      setProgressStep(0);
      setBusy('');
    }
  }

  function formatErrorMessage(err) {
    if (!err) return '';
    const msg = typeof err === 'string' ? err : (err.message || String(err));
    if (msg.includes('429') || msg.includes('quota') || msg.includes('rate limit')) {
      return '⚠️ Gemini API Rate Limit / Quota Exceeded. Please wait a moment or use fallback execution.';
    }
    if (msg.includes('504') || msg.includes('timeout') || msg.includes('Timed out')) {
      return '⏱️ Subprocess execution timed out safely.';
    }
    if (msg.includes('ROLLED_BACK') || msg.includes('restored')) {
      return '⚠️ Agent modification failed validation or tests; project state was restored automatically.';
    }
    return msg;
  }

  async function handleRunAgent(e) {
    if (e) e.preventDefault();
    if (!agentInstruction.trim() || !projectId) return;

    setBusy('agent');
    setError('');

    try {
      const initialState = await api.runCodegenAgent(projectId, agentInstruction);
      setAgentRunState(initialState);
      if (initialState.error) {
        setError(formatErrorMessage(initialState.error));
      }

      if (!['COMPLETED', 'FAILED', 'ROLLED_BACK'].includes(initialState.status)) {
        const runId = initialState.run_id;
        const pollInterval = setInterval(async () => {
          try {
            const pollState = await api.getCodegenAgentState(projectId, runId);
            setAgentRunState(pollState);
            if (pollState.error) {
              setError(formatErrorMessage(pollState.error));
            }
            if (['COMPLETED', 'FAILED', 'ROLLED_BACK'].includes(pollState.status)) {
              clearInterval(pollInterval);
              setBusy('');
              loadProjectFiles(projectId);
            }
          } catch (pollErr) {
            console.error('Agent polling error:', pollErr);
            setError(formatErrorMessage(pollErr));
            clearInterval(pollInterval);
            setBusy('');
          }
        }, 1500);
      } else {
        setBusy('');
        loadProjectFiles(projectId);
      }
    } catch (err) {
      setError(formatErrorMessage(err));
      setBusy('');
    }
  }

  function handleCopyCode() {
    if (!fileContent) return;
    navigator.clipboard.writeText(fileContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const renderTreeNodes = (node) => {
    if (!node) return null;
    if (node.is_dir) {
      return (
        <div key={node.path} style={{ marginLeft: '12px' }}>
          <div style={{ fontWeight: '600', color: '#4b5563', padding: '4px 0', fontSize: '0.875rem' }}>
            📁 {node.name}
          </div>
          {node.children && node.children.map((child) => renderTreeNodes(child))}
        </div>
      );
    }

    const isSelected = selectedFilePath === node.path;
    return (
      <div
        key={node.path}
        onClick={() => handleFileSelect(node.path)}
        style={{
          marginLeft: '24px',
          padding: '4px 8px',
          cursor: 'pointer',
          borderRadius: '4px',
          backgroundColor: isSelected ? '#e0e7ff' : 'transparent',
          color: isSelected ? '#4338ca' : '#374151',
          fontSize: '0.85rem',
          fontWeight: isSelected ? '600' : '400',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
        }}
      >
        📄 {node.name}
      </div>
    );
  };

  const currentZipUrl = (codegenResult?.zip_path || modifyResult?.zip_path || agentRunState?.zip_path)
    ? api.getCodegenDownloadUrl(projectId)
    : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header Banner */}
      <div style={{ borderBottom: '2px solid #e5e7eb', paddingBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: '700', color: '#1e293b', margin: 0 }}>
            ⚡ AIForge Code Workspace & Agentic Development Explorer
          </h2>
          <p style={{ color: '#64748b', fontSize: '0.875rem', margin: '4px 0 0 0' }}>
            Project ID: <code style={{ backgroundColor: '#f1f5f9', padding: '2px 6px', borderRadius: '4px' }}>{projectId || 'Not Generated'}</code>
          </p>
        </div>

        {currentZipUrl && (
          <a
            href={currentZipUrl}
            download
            className="btn"
            style={{
              padding: '8px 16px',
              backgroundColor: '#059669',
              color: '#fff',
              textDecoration: 'none',
              borderRadius: '6px',
              fontWeight: '600',
              fontSize: '0.875rem',
            }}
          >
            📦 Download Verified ZIP Artifact
          </a>
        )}
      </div>

      {/* LLM Status Badge */}
      {llmStatus && (
        <div style={{ padding: '10px 14px', backgroundColor: llmStatus.configured ? '#ecfdf5' : '#fffbeb', borderRadius: '6px', border: `1px solid ${llmStatus.configured ? '#a7f3d0' : '#fde68a'}`, fontSize: '0.85rem' }}>
          <span style={{ fontWeight: '700', color: llmStatus.configured ? '#047857' : '#b45309' }}>
            {llmStatus.configured ? `🟢 LLM Provider Connected: ${llmStatus.provider} (${llmStatus.model})` : '🟡 Deterministic Fallback Mode'}
          </span>
        </div>
      )}

      {error && (
        <div style={{ padding: '12px', backgroundColor: '#fef2f2', borderLeft: '4px solid #ef4444', color: '#b91c1c', borderRadius: '4px', marginBottom: '16px' }}>
          {error}
        </div>
      )}

      {/* AI DEVELOPMENT AGENT PANEL */}
      {projectId && (
        <div style={{ border: '2px solid #4f46e5', borderRadius: '12px', padding: '20px', backgroundColor: '#f5f3ff', boxShadow: '0 4px 6px -1px rgba(79, 70, 229, 0.1)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '700', color: '#3730a3', display: 'flex', alignItems: 'center', gap: '8px' }}>
              🤖 AI Development Agent
            </h3>
            <span style={{ fontSize: '0.75rem', padding: '4px 10px', backgroundColor: '#e0e7ff', color: '#4338ca', borderRadius: '12px', fontWeight: '600' }}>
              Autonomous Loop Engine
            </span>
          </div>
          <p style={{ margin: '0 0 14px 0', fontSize: '0.875rem', color: '#4c1d95' }}>
            Give the agent a natural-language engineering goal. The agent will inspect project files, generate a structured plan, apply code edits, test, auto-repair failures, verify syntax, and package the result.
          </p>

          {/* Quick Preset Prompts */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '14px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.75rem', color: '#6b7280', alignSelf: 'center', fontWeight: '600' }}>Quick Prompts:</span>
            {['Add JWT authentication', 'Add Docker support', 'Add health check endpoint'].map((preset) => (
              <button
                key={preset}
                type="button"
                onClick={() => setAgentInstruction(preset)}
                style={{ padding: '4px 10px', backgroundColor: '#ffffff', border: '1px solid #c7d2fe', borderRadius: '16px', fontSize: '0.75rem', color: '#4338ca', cursor: 'pointer', fontWeight: '500' }}
              >
                + {preset}
              </button>
            ))}
          </div>

          <form onSubmit={handleRunAgent} style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            <input
              type="text"
              value={agentInstruction}
              onChange={(e) => setAgentInstruction(e.target.value)}
              placeholder="e.g. Add JWT authentication to this project and add tests"
              style={{ flex: 1, minWidth: '300px', padding: '12px 16px', borderRadius: '8px', border: '1px solid #818cf8', fontSize: '0.95rem', backgroundColor: '#ffffff' }}
            />
            <button
              type="submit"
              disabled={!agentInstruction.trim() || !!busy}
              className="btn"
              style={{
                padding: '12px 24px',
                backgroundColor: '#4338ca',
                color: '#fff',
                border: 'none',
                borderRadius: '8px',
                fontWeight: '700',
                fontSize: '0.95rem',
                cursor: (!agentInstruction.trim() || busy) ? 'not-allowed' : 'pointer',
              }}
            >
              {busy === 'agent' ? '⏳ Agent Working...' : '🤖 Run AI Agent'}
            </button>
          </form>

          {/* Agent Run Progress & Results Display */}
          {agentRunState && (
            <div style={{ marginTop: '20px', backgroundColor: '#ffffff', borderRadius: '10px', padding: '16px', border: '1px solid #ddd6fe' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', borderBottom: '1px solid #f3e8ff', paddingBottom: '8px' }}>
                <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: '700', color: '#312e81' }}>
                  Agent Run Status: <span style={{ color: agentRunState.status === 'COMPLETED' ? '#059669' : agentRunState.status === 'ROLLED_BACK' ? '#dc2626' : '#4338ca' }}>{agentRunState.status}</span>
                </h4>
                <span style={{ fontSize: '0.75rem', color: '#6b7280' }}>
                  Run ID: {agentRunState.run_id}
                </span>
              </div>

              {/* Status Timeline Checklist */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '8px', marginBottom: '16px' }}>
                <div style={{ padding: '6px 10px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: '600', backgroundColor: '#ecfdf5', color: '#047857' }}>✓ Project Analyzed</div>
                <div style={{ padding: '6px 10px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: '600', backgroundColor: '#ecfdf5', color: '#047857' }}>✓ Relevant Files Identified</div>
                <div style={{ padding: '6px 10px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: '600', backgroundColor: agentRunState.plan ? '#ecfdf5' : '#f3f4f6', color: agentRunState.plan ? '#047857' : '#9ca3af' }}>{agentRunState.plan ? '✓ Plan Generated' : '⏳ Generating Plan...'}</div>
                <div style={{ padding: '6px 10px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: '600', backgroundColor: (agentRunState.files_created.length + agentRunState.files_modified.length > 0) ? '#ecfdf5' : '#f3f4f6', color: (agentRunState.files_created.length + agentRunState.files_modified.length > 0) ? '#047857' : '#9ca3af' }}>Changes Applied</div>
                <div style={{ padding: '6px 10px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: '600', backgroundColor: agentRunState.test_result?.success ? '#ecfdf5' : '#f3f4f6', color: agentRunState.test_result?.success ? '#047857' : '#9ca3af' }}>Tests Passed</div>
                <div style={{ padding: '6px 10px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: '600', backgroundColor: agentRunState.verification_result?.success ? '#ecfdf5' : '#f3f4f6', color: agentRunState.verification_result?.success ? '#047857' : '#9ca3af' }}>Verification Passed</div>
                <div style={{ padding: '6px 10px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: '600', backgroundColor: agentRunState.zip_available ? '#ecfdf5' : '#f3f4f6', color: agentRunState.zip_available ? '#047857' : '#9ca3af' }}>ZIP Generated</div>
              </div>

              {/* Structured Plan Steps Display */}
              {agentRunState.plan && (
                <div style={{ marginBottom: '16px', backgroundColor: '#f8fafc', padding: '12px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                  <h5 style={{ margin: '0 0 8px 0', fontSize: '0.85rem', fontWeight: '700', color: '#334155' }}>
                    📌 Structured Agent Execution Plan ({agentRunState.plan.steps.length} Steps): {agentRunState.plan.summary}
                  </h5>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {agentRunState.plan.steps.map((step) => (
                      <div key={step.id} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', backgroundColor: '#ffffff', padding: '6px 10px', borderRadius: '6px', border: '1px solid #f1f5f9' }}>
                        <span style={{ fontWeight: '700', color: '#4f46e5' }}>{step.id}:</span>
                        <span style={{ padding: '2px 6px', backgroundColor: '#e0e7ff', color: '#3730a3', borderRadius: '4px', fontSize: '0.7rem', fontWeight: '700', textTransform: 'uppercase' }}>{step.action}</span>
                        <span style={{ flex: 1, color: '#1e293b' }}>{step.description}</span>
                        {step.files && step.files.length > 0 && (
                          <span style={{ color: '#64748b', fontSize: '0.75rem', fontFamily: 'monospace' }}>[{step.files.join(', ')}]</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Changed Files Overview */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '8px', fontSize: '0.8rem', marginBottom: '12px' }}>
                <div style={{ background: '#f8fafc', padding: '8px 12px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                  <strong>Files Created:</strong> {agentRunState.files_created.length > 0 ? agentRunState.files_created.map(f => `+ ${f}`).join(', ') : 'None'}
                </div>
                <div style={{ background: '#f8fafc', padding: '8px 12px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                  <strong>Files Modified:</strong> {agentRunState.files_modified.length > 0 ? agentRunState.files_modified.map(f => `~ ${f}`).join(', ') : 'None'}
                </div>
                <div style={{ background: '#f8fafc', padding: '8px 12px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                  <strong>Files Deleted:</strong> {agentRunState.files_deleted.length > 0 ? agentRunState.files_deleted.map(f => `- ${f}`).join(', ') : 'None'}
                </div>
                <div style={{ background: '#f8fafc', padding: '8px 12px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                  <strong>Repair Attempts:</strong> {agentRunState.repair_attempts}
                </div>
              </div>

              {/* Verification & ZIP Download */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: agentRunState.verification_result?.success ? '#f0fdf4' : '#fff1f2', padding: '10px 14px', borderRadius: '6px', border: `1px solid ${agentRunState.verification_result?.success ? '#bbf7d0' : '#fecdd3'}` }}>
                <span style={{ fontSize: '0.85rem', fontWeight: '700', color: agentRunState.verification_result?.success ? '#166534' : '#9f1239' }}>
                  Verification Stage: {agentRunState.verification_result?.success ? '✅ PASS' : '❌ FAIL / ROLLED BACK'}
                </span>
                {agentRunState.zip_available && (
                  <a href={api.getCodegenDownloadUrl(projectId)} download className="btn" style={{ padding: '6px 14px', backgroundColor: '#059669', color: '#fff', textDecoration: 'none', borderRadius: '6px', fontSize: '0.8rem', fontWeight: '600' }}>
                    📦 Download Updated ZIP
                  </a>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Code Workspace Grid: File Tree + Source Code View */}
      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: '16px', minHeight: '400px' }}>
        {/* Left: File Tree */}
        <div style={{ border: '1px solid #e5e7eb', borderRadius: '8px', padding: '12px', backgroundColor: '#f9fafb', overflowY: 'auto' }}>
          <h3 style={{ fontSize: '0.85rem', fontWeight: '700', textTransform: 'uppercase', color: '#6b7280', marginBottom: '12px' }}>
            📂 Project File Explorer
          </h3>
          {fileTree ? renderTreeNodes(fileTree) : <div style={{ color: '#9ca3af', fontSize: '0.85rem' }}>No project files generated yet.</div>}
        </div>

        {/* Right: Code Viewer & Test/Metrics Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Source Code Viewer with Copy Action */}
          <div style={{ border: '1px solid #e5e7eb', borderRadius: '8px', overflow: 'hidden', backgroundColor: '#1e293b', flex: 1, display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '8px 16px', backgroundColor: '#0f172a', color: '#94a3b8', fontSize: '0.85rem', borderBottom: '1px solid #334155', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>📄 {selectedFilePath || 'Select a file to inspect'}</span>
              <button
                onClick={handleCopyCode}
                style={{
                  padding: '4px 10px',
                  backgroundColor: '#334155',
                  color: '#e2e8f0',
                  border: 'none',
                  borderRadius: '4px',
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                }}
              >
                {copied ? '✓ Copied!' : '📋 Copy Code'}
              </button>
            </div>
            <pre style={{ margin: 0, padding: '16px', color: '#f8fafc', fontFamily: 'monospace', fontSize: '0.875rem', overflowX: 'auto', flex: 1, minHeight: '260px' }}>
              {loadingFile ? 'Loading file content...' : fileContent || '// Select a generated file from the explorer to view source code'}
            </pre>
          </div>

          {/* Actual Model Metrics Display */}
          {metrics && (
            <div style={{ border: '1px solid #cbd5e1', borderRadius: '8px', padding: '16px', backgroundColor: '#f8fafc' }}>
              <h4 style={{ margin: '0 0 12px 0', fontSize: '0.9rem', fontWeight: '700', color: '#334155' }}>
                📊 Real Trained Model Evaluation Metrics
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', textAlign: 'center' }}>
                <div style={{ padding: '8px', background: '#fff', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Accuracy</div>
                  <div style={{ fontSize: '1rem', fontWeight: '700', color: '#059669' }}>{metrics.accuracy ?? 'N/A'}</div>
                </div>
                <div style={{ padding: '8px', background: '#fff', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Precision</div>
                  <div style={{ fontSize: '1rem', fontWeight: '700', color: '#0284c7' }}>{metrics.precision ?? 'N/A'}</div>
                </div>
                <div style={{ padding: '8px', background: '#fff', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Recall</div>
                  <div style={{ fontSize: '1rem', fontWeight: '700', color: '#6366f1' }}>{metrics.recall ?? 'N/A'}</div>
                </div>
                <div style={{ padding: '8px', background: '#fff', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>F1 Score</div>
                  <div style={{ fontSize: '1rem', fontWeight: '700', color: '#d97706' }}>{metrics.f1 ?? 'N/A'}</div>
                </div>
              </div>
            </div>
          )}

          {/* Pytest Execution Output Panel */}
          {testResult && (
            <div style={{ border: '1px solid #e5e7eb', borderRadius: '8px', padding: '16px', backgroundColor: testResult.success ? '#f0fdf4' : '#fef2f2' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: '700', color: testResult.success ? '#166534' : '#991b1b' }}>
                  {testResult.success ? '✅ Pytest Suite Passed' : '❌ Pytest Suite Failed'} (Exit Code: {testResult.exit_code}, Duration: {testResult.duration_seconds}s)
                </h4>
              </div>
              <pre style={{ margin: 0, padding: '10px', backgroundColor: '#0f172a', color: '#e2e8f0', borderRadius: '6px', fontSize: '0.8rem', overflowX: 'auto', maxHeight: '160px' }}>
                {testResult.stdout || testResult.stderr || 'No output.'}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
