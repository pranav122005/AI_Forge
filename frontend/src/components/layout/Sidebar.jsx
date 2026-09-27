import React from 'react';
import {
  IconTerminal,
  IconWorkspace,
  IconCpu,
  IconDatabase,
  IconFlask,
  IconReport,
  IconSettings,
} from '../common/Icons';

export const Sidebar = ({
  open,
  onToggle,
  view,
  onViewChange,
  activeProject,
  health,
  capabilities,
  activeProvider,
}) => {
  const navItems = [
    { id: 'command', label: 'Command Center', icon: IconTerminal },
    { id: 'workspace', label: 'Workspace', icon: IconWorkspace },
    { id: 'providers', label: 'AI Providers', icon: IconCpu },
    { id: 'dataset', label: 'Dataset & ML', icon: IconDatabase },
    { id: 'playground', label: 'Playground', icon: IconFlask },
    { id: 'reports', label: 'Reports', icon: IconReport },
    { id: 'system', label: 'System', icon: IconSettings },
  ];

  const isBackendReady = Boolean(health?.status === 'ok' || health?.service_alive);
  const providerLabel = (activeProvider || 'gemini').toUpperCase();

  return (
    <>
      {open && (
        <div className="sidebar-overlay" onClick={onToggle} aria-label="Close Sidebar" />
      )}
      <aside className={`sidebar ${open ? '' : 'sidebar-collapsed'}`}>
        <div className="sidebar-brand">
          <div className="brand-icon">
            <span className="brand-dot"></span>
          </div>
          {open && (
            <div className="brand-text">
              <h1 className="brand-title">AIForge</h1>
              <p className="brand-subtitle">SCIENTIFIC AI ENGINEERING</p>
            </div>
          )}
        </div>

        {open && (
          <div className="sidebar-actions">
            <button className="new-project-btn" onClick={() => onViewChange('command')}>
              + New Project
            </button>
          </div>
        )}

        <nav className="sidebar-nav">
          {navItems.map((item) => {
            const IconComp = item.icon;
            const isActive = view === item.id;
            return (
              <button
                key={item.id}
                className={`nav-item ${isActive ? 'nav-item-active' : ''}`}
                onClick={() => onViewChange(item.id)}
                aria-label={item.label}
                title={!open ? item.label : undefined}
              >
                <span className="nav-icon">
                  <IconComp size={16} />
                </span>
                {open && <span className="nav-label">{item.label}</span>}
              </button>
            );
          })}
        </nav>

        {open && activeProject && (
          <div className="sidebar-project">
            <div className="project-id" title={activeProject.project_id}>
              PRJ: {(activeProject.project_id || '').slice(0, 14)}
            </div>
            <div className="project-status">
              <span className="status-indicator-dot"></span>
              <span className="status-badge">{activeProject.lifecycle_status || activeProject.status || 'Active'}</span>
            </div>
          </div>
        )}

        <div className="sidebar-footer">
          <div className="system-status">
            <div className="status-item" title={`Active AI Engine: ${providerLabel}`}>
              <span className="status-dot status-dot-online" />
              {open && <span>ENGINE: {providerLabel}</span>}
            </div>
            <div className="status-item" title={`Backend Server: ${isBackendReady ? 'READY' : 'OFFLINE'}`}>
              <span className={`status-dot ${isBackendReady ? 'status-dot-online' : 'status-dot-offline'}`} />
              {open && <span>API: {isBackendReady ? 'READY' : 'OFFLINE'}</span>}
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};
