import React, { useEffect, useState, useCallback } from 'react';
import { api } from './api/apiClient';

/* ── Layout Components ── */
import { Sidebar } from './components/layout/Sidebar';
import { Topbar } from './components/layout/Topbar';

/* ── View Components ── */
import { CommandCenter } from './components/chat/CommandCenter';
import { Workspace } from './components/workspace/Workspace';
import { ProviderControlCenter } from './components/providers/ProviderControlCenter';
import { DatasetManager } from './components/ml/DatasetManager';
import { Playground } from './components/ml/Playground';
import { Reports } from './components/reports/Reports';
import { SystemStatus } from './components/system/SystemStatus';
import { IconAlertTriangle, IconClose, IconZap, IconWorkspace } from './components/common/Icons';

export function App() {
  /* ── Global State ── */
  const [health, setHealth] = useState(null);
  const [capabilities, setCapabilities] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  /* ── LLM Provider State ── */
  const [activeProvider, setActiveProvider] = useState('gemini');
  const [activeModel, setActiveModel] = useState('gemini-3.8-flash');

  /* ── Project State ── */
  const [activeProject, setActiveProject] = useState(null);
  const [codegenResult, setCodegenResult] = useState(null);

  /* ── ML State ── */
  const [inspection, setInspection] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [textPrediction, setTextPrediction] = useState(null);
  const [imagePrediction, setImagePrediction] = useState(null);
  const [agentResult, setAgentResult] = useState(null);

  /* ── UX State ── */
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [view, setView] = useState('command'); // command | workspace | providers | dataset | playground | reports | system
  const [chatMessages, setChatMessages] = useState([]);

  /* ── Fetch system status & active provider on mount ── */
  useEffect(() => {
    (async () => {
      try {
        const h = await api.getHealth();
        setHealth(h);
        const c = await api.getCapabilities();
        setCapabilities(c);
        const act = await api.getLLMActive();
        if (act?.provider) {
          setActiveProvider(act.provider);
          setActiveModel(act.model || '');
        }
      } catch { /* silent */ }
    })();
  }, []);

  /* ── Chat message helpers ── */
  const addMessage = useCallback((role, content, meta = {}) => {
    setChatMessages(prev => [...prev, { id: Date.now() + Math.random(), role, content, timestamp: Date.now(), ...meta }]);
  }, []);

  const updateLastAI = useCallback((content, meta = {}) => {
    setChatMessages(prev => {
      const copy = [...prev];
      for (let i = copy.length - 1; i >= 0; i--) {
        if (copy[i].role === 'ai') {
          copy[i] = { ...copy[i], content, ...meta };
          break;
        }
      }
      return copy;
    });
  }, []);

  /* ── Handlers ── */
  async function handleGenerate(requirement, projectName) {
    setBusy('generate');
    setError('');
    addMessage('user', requirement);
    addMessage('ai', 'Understanding requirement...', { stage: 'analyzing' });

    try {
      const res = await api.generateProject(requirement, projectName);
      setActiveProject(res);
      setChatMessages(prev => [...prev.slice(0, -1),
        { id: Date.now(), role: 'ai', content: 'Project generated successfully.', stage: 'completed', result: res, timestamp: Date.now() }
      ]);
      setInspection(null);
      setMetrics(null);
      setTextPrediction(null);
      setImagePrediction(null);
      setAgentResult(null);
    } catch (err) {
      const msg = err.message || 'Project generation failed.';
      setError(msg);
      updateLastAI(`Generation failed: ${msg}`, { stage: 'failed' });
    } finally {
      setBusy('');
    }
  }

  async function handleCodegen(requirement, projectName, extraInstructions) {
    setBusy('codegen');
    setError('');
    addMessage('user', requirement);
    const providerLabel = activeProvider ? activeProvider.toUpperCase() : 'AI ENGINE';
    addMessage('ai', `Analyzing requirement with ${providerLabel} (${activeModel || 'default'})...`, { stage: 'analyzing' });

    try {
      updateLastAI(`Generating project code using ${providerLabel}...`, { stage: 'generating' });
      const res = await api.generateCodegen(requirement, projectName, extraInstructions, activeProvider, activeModel);
      setCodegenResult(res);
      setActiveProject({ project_id: res.project_id, project_name: projectName || 'aiforge-project', status: res.status });

      const testStatus = res.test_result?.success ? 'Tests passed' : 'Tests failed';
      setChatMessages(prev => [...prev.slice(0, -1),
        { id: Date.now(), role: 'ai', content: `Project generated with ${providerLabel}. ${res.generated_files?.length || 0} files created. ${testStatus}.`, stage: 'completed', result: res, timestamp: Date.now() }
      ]);
      setView('workspace');
    } catch (err) {
      const msg = err.message || 'Code generation failed.';
      setError(msg);
      updateLastAI(`Code generation failed: ${msg}`, { stage: 'failed' });
    } finally {
      setBusy('');
    }
  }

  async function handleTrain(fileObject) {
    const pid = activeProject?.project_id;
    if (!pid) return;
    setBusy('train');
    setError('');
    try {
      const res = await api.trainModel(pid, fileObject);
      if (res.inspection) setInspection(res.inspection);
      if (res.metrics) setMetrics(res.metrics);
      setActiveProject(prev => ({ ...prev, lifecycle_status: 'TRAINED' }));
    } catch (err) {
      setError(err.message || 'Training failed.');
    } finally {
      setBusy('');
    }
  }

  async function handlePredictText(text) {
    const pid = activeProject?.project_id;
    if (!pid) return;
    setBusy('predictText');
    setError('');
    try {
      const res = await api.predictText(pid, text);
      setTextPrediction(res);
    } catch (err) {
      setError(err.message || 'Text prediction failed.');
    } finally {
      setBusy('');
    }
  }

  async function handlePredictImage(imageFile) {
    const pid = activeProject?.project_id;
    if (!pid) return;
    setBusy('predictImage');
    setError('');
    try {
      const res = await api.predictImage(pid, imageFile);
      setImagePrediction(res);
    } catch (err) {
      setError(err.message || 'Image prediction failed.');
    } finally {
      setBusy('');
    }
  }

  async function handleRunAgent(inputPayload) {
    const pid = activeProject?.project_id;
    if (!pid) return;
    setBusy('agent');
    setError('');
    try {
      const res = await api.runAgent(pid, inputPayload);
      setAgentResult(res);
    } catch (err) {
      setError(err.message || 'Agent execution failed.');
    } finally {
      setBusy('');
    }
  }

  function handleProviderChanged(newProvider, newModel) {
    setActiveProvider(newProvider);
    if (newModel) setActiveModel(newModel);
  }

  const projectId = activeProject?.project_id;
  const selectedComps = activeProject?.selected_components || [];
  const hasVision = selectedComps.includes('opencv') || selectedComps.includes('ocr');

  return (
    <div className="app-shell">
      <Sidebar
        open={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        view={view}
        onViewChange={setView}
        activeProject={activeProject}
        health={health}
        capabilities={capabilities}
        activeProvider={activeProvider}
      />

      <div className={`app-main ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
        <Topbar
          view={view}
          onViewChange={setView}
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          health={health}
          capabilities={capabilities}
          activeProject={activeProject}
          activeProvider={activeProvider}
          activeModel={activeModel}
        />

        <main className="app-content">
          {error && (
            <div className="error-banner" role="alert">
              <IconAlertTriangle size={16} className="error-icon" />
              <span>{error}</span>
              <button className="error-dismiss" onClick={() => setError('')} aria-label="Dismiss error">
                <IconClose size={14} />
              </button>
            </div>
          )}

          {view === 'command' && (
            <CommandCenter
              onGenerate={handleGenerate}
              onCodegen={handleCodegen}
              busy={busy}
              messages={chatMessages}
              activeProject={activeProject}
              onViewWorkspace={() => setView('workspace')}
              activeProvider={activeProvider}
              activeModel={activeModel}
              onProviderChange={setActiveProvider}
              onViewProviders={() => setView('providers')}
            />
          )}

          {view === 'workspace' && projectId && (
            <Workspace
              projectId={projectId}
              codegenResult={codegenResult}
              activeProject={activeProject}
              metrics={metrics}
              busy={busy}
              onProjectUpdate={setActiveProject}
              activeProvider={activeProvider}
              activeModel={activeModel}
            />
          )}

          {view === 'workspace' && !projectId && (
            <div className="empty-state">
              <div className="empty-state-icon">
                <IconWorkspace size={32} />
              </div>
              <h2>No Project Selected</h2>
              <p>Generate a new project or select an existing one from the sidebar.</p>
              <button className="btn-primary" onClick={() => setView('command')}>
                <IconZap size={14} className="mr-1" />
                <span>Open Command Center</span>
              </button>
            </div>
          )}

          {view === 'providers' && (
            <ProviderControlCenter
              activeProvider={activeProvider}
              activeModel={activeModel}
              onProviderChanged={handleProviderChanged}
              onError={setError}
            />
          )}

          {view === 'dataset' && (
            <DatasetManager
              projectId={projectId}
              onTrain={handleTrain}
              busy={busy}
              metrics={metrics}
              inspection={inspection}
            />
          )}

          {view === 'playground' && (
            <Playground
              projectId={projectId}
              onPredictText={handlePredictText}
              onPredictImage={handlePredictImage}
              busy={busy}
              textPrediction={textPrediction}
              imagePrediction={imagePrediction}
              hasVision={hasVision}
            />
          )}

          {view === 'reports' && (
            <Reports
              activeProject={activeProject}
              projectId={projectId}
              codegenResult={codegenResult}
              metrics={metrics}
              activeProvider={activeProvider}
              activeModel={activeModel}
            />
          )}

          {view === 'system' && (
            <SystemStatus health={health} capabilities={capabilities} />
          )}
        </main>
      </div>
    </div>
  );
}
