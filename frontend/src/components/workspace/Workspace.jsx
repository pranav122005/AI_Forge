import React, { useState, useEffect, useCallback } from 'react';
import { api } from '../../api/apiClient';
import { FileTree } from './FileTree';
import { CodeViewer } from './CodeViewer';
import { AgentPanel } from './AgentPanel';
import {
  IconCpu,
  IconDownload,
  IconFolder,
  IconDatabase,
  IconCheckCircle,
  IconXCircle,
  IconBot,
  IconTerminal,
} from '../common/Icons';

export function Workspace({
  projectId,
  codegenResult,
  activeProject,
  metrics,
  busy,
  onProjectUpdate,
  activeProvider,
  activeModel,
}) {
  const [fileTree, setFileTree] = useState(null);
  const [selectedFilePath, setSelectedFilePath] = useState('');
  const [fileContent, setFileContent] = useState('');
  const [loadingFile, setLoadingFile] = useState(false);
  const [testResult, setTestResult] = useState(codegenResult?.test_result || null);
  const [agentRunState, setAgentRunState] = useState(null);
  const [agentPanelOpen, setAgentPanelOpen] = useState(true);
  const [error, setError] = useState('');
  const [agentBusy, setAgentBusy] = useState('');

  /* Load file tree */
  const loadProjectFiles = useCallback(async () => {
    if (!projectId) return;
    try {
      const tree = await api.getCodegenFiles(projectId);
      setFileTree(tree);
      if (tree && !selectedFilePath) {
        const first = findFirstFile(tree);
        if (first) {
          setSelectedFilePath(first.path);
        }
      }
    } catch (err) {
      setError(err.message || 'Failed to load file tree.');
    }
  }, [projectId]);

  useEffect(() => {
    loadProjectFiles();
  }, [loadProjectFiles]);

  /* Load file content when selection changes */
  useEffect(() => {
    if (!projectId || !selectedFilePath || selectedFilePath === '.') return;
    let cancelled = false;
    setLoadingFile(true);
    (async () => {
      try {
        const res = await api.getCodegenFileContent(projectId, selectedFilePath);
        if (!cancelled) {
          setFileContent(typeof res === 'string' ? res : (res.content || ''));
        }
      } catch (err) {
        if (!cancelled) setFileContent(`// Error loading file: ${err.message}`);
      } finally {
        if (!cancelled) setLoadingFile(false);
      }
    })();
    return () => { cancelled = true; };
  }, [projectId, selectedFilePath]);

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

  const handleDownloadZip = () => {
    if (!projectId) return;
    const url = api.getCodegenDownloadUrl(projectId);
    window.open(url, '_blank');
  };

  const projectName = activeProject?.project_name || activeProject?.name || projectId || 'Unnamed Project';
  const status = codegenResult?.status || agentRunState?.status || '';
  const engineProvider = (codegenResult?.provider || activeProvider || 'gemini').toUpperCase();
  const engineModel = codegenResult?.model || activeModel || 'gemini-3.8-flash';

  return (
    <div className="workspace">
      <header className="workspace-header">
        <div className="workspace-header-info">
          <h2 className="workspace-title">{projectName}</h2>
          <code className="workspace-id">{projectId}</code>
          <span className="badge tone-cyan" title="AI Engine used for generation/modification">
            <IconCpu size={12} className="mr-1" />
            {engineProvider} · {engineModel}
          </span>
          {status && (
            <span className={`badge ${status === 'completed' || status === 'COMPLETED' ? 'tone-green' : status === 'failed' || status === 'FAILED' || status === 'ROLLED_BACK' ? 'tone-red' : 'tone-amber'}`}>
              STATUS: {status.toUpperCase()}
            </span>
          )}
          {testResult && (
            <span className={`badge ${testResult.success ? 'tone-green' : 'tone-red'}`}>
              {testResult.success ? (
                <>
                  <IconCheckCircle size={11} className="mr-1" />
                  TESTS PASSED
                </>
              ) : (
                <>
                  <IconXCircle size={11} className="mr-1" />
                  TESTS FAILED
                </>
              )}
            </span>
          )}
        </div>
        <div className="workspace-header-actions">
          <button className="btn-secondary" onClick={handleDownloadZip} aria-label="Download ZIP">
            <IconDownload size={13} className="mr-1" />
            <span>EXPORT ZIP</span>
          </button>
          <button className="btn-ghost" onClick={() => setAgentPanelOpen(!agentPanelOpen)} aria-label="Toggle Agent Panel">
            <IconBot size={13} className="mr-1" />
            <span>{agentPanelOpen ? 'HIDE AGENT' : 'SHOW AGENT'}</span>
          </button>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button className="error-dismiss" onClick={() => setError('')}>×</button>
        </div>
      )}

      <div className={`workspace-panels ${agentPanelOpen ? 'with-agent' : ''}`}>
        <div className="workspace-left">
          <div className="panel-section-header">
            <IconFolder size={14} />
            <span>PROJECT FILES</span>
          </div>
          <FileTree tree={fileTree} selectedPath={selectedFilePath} onSelect={setSelectedFilePath} />
        </div>

        <div className="workspace-center">
          <CodeViewer filePath={selectedFilePath} content={fileContent} loading={loadingFile} />

          {metrics && (
            <div className="workspace-metrics">
              <div className="panel-section-header">
                <IconDatabase size={14} />
                <span>MODEL EVALUATION METRICS</span>
              </div>
              <div className="metrics-grid">
                <div className="metric-box">
                  <span className="metric-label">Accuracy</span>
                  <strong className="metric-value">{typeof metrics.accuracy === 'number' ? (metrics.accuracy * 100).toFixed(1) + '%' : 'N/A'}</strong>
                </div>
                <div className="metric-box">
                  <span className="metric-label">Precision</span>
                  <strong className="metric-value">{typeof metrics.precision === 'number' ? metrics.precision.toFixed(3) : 'N/A'}</strong>
                </div>
                <div className="metric-box">
                  <span className="metric-label">Recall</span>
                  <strong className="metric-value">{typeof metrics.recall === 'number' ? metrics.recall.toFixed(3) : 'N/A'}</strong>
                </div>
                <div className="metric-box">
                  <span className="metric-label">F1 Score</span>
                  <strong className="metric-value">{typeof (metrics.f1_score ?? metrics.f1) === 'number' ? (metrics.f1_score ?? metrics.f1).toFixed(3) : 'N/A'}</strong>
                </div>
              </div>
            </div>
          )}

          {testResult && (
            <div className={`workspace-test-output ${testResult.success ? 'test-pass' : 'test-fail'}`}>
              <div className="test-output-header">
                <div className="test-status-text">
                  {testResult.success ? <IconCheckCircle size={14} className="text-green" /> : <IconXCircle size={14} className="text-red" />}
                  <strong>{testResult.success ? 'PYTEST EXECUTION: PASSED' : 'PYTEST EXECUTION: FAILED'}</strong>
                </div>
                <span className="test-meta">EXIT: {testResult.exit_code} · {testResult.duration_seconds}s</span>
              </div>
              <pre className="test-output-body">{testResult.stdout || testResult.stderr || 'No output.'}</pre>
            </div>
          )}
        </div>

        {agentPanelOpen && (
          <div className="workspace-right">
            <AgentPanel
              projectId={projectId}
              agentRunState={agentRunState}
              onAgentRunStateChange={setAgentRunState}
              busy={agentBusy}
              onBusyChange={setAgentBusy}
              onRefreshFiles={loadProjectFiles}
              onError={setError}
              activeProvider={activeProvider}
              activeModel={activeModel}
            />
          </div>
        )}
      </div>
    </div>
  );
}
