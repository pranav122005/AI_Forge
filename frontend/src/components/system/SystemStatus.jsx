import React, { useState } from 'react';
import { api } from '../../api/apiClient';
import {
  IconSettings,
  IconCpu,
  IconTerminal,
  IconEye,
  IconDatabase,
  IconBot,
  IconCheckCircle,
  IconXCircle,
  IconChevronDown,
  IconChevronRight,
} from '../common/Icons';

export const SystemStatus = ({ health, capabilities }) => {
  const [showJson, setShowJson] = useState(false);

  const getBackendStatus = () => {
    return health?.status === 'healthy' || health?.status === 'ok' ? 'online' : 'offline';
  };

  const getLlmStatus = () => {
    const isGemini = Boolean(
      capabilities?.gemini_configured ||
      health?.gemini_configured ||
      capabilities?.gemini_reachable ||
      health?.gemini_reachable ||
      (capabilities?.llm?.available && capabilities?.llm?.provider === 'gemini') ||
      (health?.llm?.available && health?.llm?.provider === 'gemini')
    );
    return isGemini ? 'online' : 'offline';
  };

  const getOcrStatus = () => {
    return capabilities?.vision?.ocr ? 'online' : 'offline';
  };

  const getMlStatus = () => {
    return capabilities?.ml?.text_classification ? 'online' : 'offline';
  };

  const getAgentStatus = () => {
    return capabilities?.agent ? 'online' : 'offline';
  };

  const renderCard = (name, status, detail, IconComponent = IconTerminal) => (
    <div className={`status-card ${status === 'online' ? 'status-card-online' : 'status-card-offline'}`}>
      <div className="status-icon">
        <IconComponent size={20} />
      </div>
      <div className="status-info">
        <div className="status-name">{name}</div>
        <div className="status-detail">{detail}</div>
      </div>
      <div className={`status-dot ${status === 'online' ? 'status-dot-online' : 'status-dot-offline'}`}></div>
    </div>
  );

  const providers = capabilities?.providers || health?.providers || [];
  const activeProviderId = capabilities?.active_provider || health?.active_provider || 'gemini';

  return (
    <div className="system-view">
      <div className="system-header-group">
        <div className="system-kicker">
          <IconSettings size={14} />
          <span>RUNTIME DIAGNOSTICS</span>
        </div>
        <h1 className="system-header">System Status & Environment Health</h1>
      </div>

      <div className="status-grid">
        {renderCard(
          'Backend API Core',
          getBackendStatus(),
          health?.version ? `Version: ${health.version}` : 'Online (FastAPI / Uvicorn)',
          IconTerminal
        )}

        {providers.length > 0 ? (
          providers.map((p) => {
            const isAct = p.id === activeProviderId || p.active;
            const isConfig = p.configured;
            return renderCard(
              `${p.name} (${p.id})`,
              isConfig ? 'online' : 'offline',
              `${isAct ? '[ACTIVE] · ' : ''}${p.default_model || 'default'} · ${isConfig ? 'Configured' : 'Not configured'}`,
              IconCpu
            );
          })
        ) : (
          renderCard(
            'Gemini LLM Provider',
            getLlmStatus(),
            capabilities?.llm?.provider && capabilities?.llm?.model
              ? `${capabilities.llm.provider} - ${capabilities.llm.model}`
              : 'Configured & Online',
            IconCpu
          )
        )}

        {renderCard(
          'OCR Service',
          getOcrStatus(),
          getOcrStatus() === 'online' ? 'Engine Ready' : 'Standby / Optional',
          IconEye
        )}

        {renderCard(
          'ML Model Pipeline',
          getMlStatus(),
          getMlStatus() === 'online' ? 'Scikit-Learn Ready' : 'Standby',
          IconDatabase
        )}

        {renderCard(
          'Autonomous Agent',
          getAgentStatus(),
          getAgentStatus() === 'online' ? 'Autonomous Loop Engine Ready' : 'Standby',
          IconBot
        )}
      </div>

      <div className="json-toggle" onClick={() => setShowJson(!showJson)}>
        {showJson ? <IconChevronDown size={14} className="mr-1" /> : <IconChevronRight size={14} className="mr-1" />}
        <span>{showJson ? 'HIDE RAW TELEMETRY JSON' : 'INSPECT RAW TELEMETRY JSON'}</span>
      </div>

      {showJson && (
        <div className="json-viewer">
          <pre>{JSON.stringify({ health, capabilities }, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};
