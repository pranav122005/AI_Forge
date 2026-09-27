import React from 'react';
import { api } from '../../api/apiClient';
import {
  IconMenu,
  IconPackage,
  IconCpu,
  IconChevronRight,
  IconCheckCircle,
  IconXCircle,
} from '../common/Icons';

export function Topbar({
  view,
  onViewChange,
  onToggleSidebar,
  health,
  capabilities,
  activeProject,
  activeProvider,
  activeModel,
}) {
  const isBackendReady = Boolean(health?.status === 'ok' || health?.service_alive);
  const providerName = (activeProvider || 'Gemini').toUpperCase();
  const modelName = activeModel || 'gemini-3.8-flash';

  const viewNames = {
    command: 'Command Center',
    workspace: 'Workspace',
    providers: 'AI Providers',
    dataset: 'Dataset & ML',
    playground: 'Playground',
    reports: 'Reports & Intelligence',
    system: 'System Diagnostics',
  };

  const projectId = activeProject?.project_id;
  const currentSectionName = viewNames[view] || 'Workspace';

  return (
    <header className="topbar">
      <div className="topbar-left">
        <button className="hamburger-btn" onClick={onToggleSidebar} aria-label="Toggle Sidebar">
          <IconMenu size={16} />
        </button>
        <div className="breadcrumb">
          <span className="breadcrumb-root">AIForge</span>
          <span className="breadcrumb-separator"><IconChevronRight size={12} /></span>
          <span className="breadcrumb-section">{currentSectionName}</span>
          {activeProject && (
            <>
              <span className="breadcrumb-separator"><IconChevronRight size={12} /></span>
              <span className="breadcrumb-project" title={projectId}>
                {activeProject.project_name || (projectId ? projectId.slice(0, 16) : 'Active Project')}
              </span>
            </>
          )}
        </div>
      </div>

      <div className="topbar-right">
        {activeProject && projectId && (
          <div className="topbar-project">
            <a
              className="btn-ghost topbar-download"
              href={api.getCodegenDownloadUrl(projectId)}
              download
              title="Download Project ZIP Package"
            >
              <IconPackage size={14} />
              <span>EXPORT ZIP</span>
            </a>
          </div>
        )}
        <div className="status-indicators">
          <div
            className="status-pill status-pill-online status-pill-clickable"
            onClick={() => onViewChange('providers')}
            title={`Active Engine: ${providerName} (${modelName}). Click to manage.`}
          >
            <IconCpu size={13} className="engine-icon" />
            <span className="engine-text">[ {providerName} · {modelName} ]</span>
          </div>
          <div
            className={`status-pill ${isBackendReady ? 'status-pill-online' : 'status-pill-offline'}`}
            title={`Backend Service: ${isBackendReady ? 'Online' : 'Offline'}`}
          >
            {isBackendReady ? (
              <span className="status-dot status-dot-online" />
            ) : (
              <span className="status-dot status-dot-offline" />
            )}
            <span className="backend-text">API: {isBackendReady ? 'ONLINE' : 'OFFLINE'}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
