import React from 'react';
import { Sparkles, Activity, AlertTriangle, CheckCircle2, Cpu } from 'lucide-react';

export function Header({ health, capabilities }) {
  const isOnline = health?.status === 'ok';
  
  const llmInfo = capabilities?.llm || health?.llm || {};
  const isLlmAvailable = llmInfo.available || false;
  const providerName = (llmInfo.provider || 'fallback').toUpperCase();
  const modelName = llmInfo.model || 'none';

  const hasOcr = capabilities?.vision?.ocr || health?.components?.ocr;

  return (
    <header className="topbar">
      <div className="brand">
        <div className="logo">
          <Sparkles size={20} />
        </div>
        <div>
          <div className="brand-name">AIForge</div>
          <div className="brand-sub">Agentic AI Engineering Factory</div>
        </div>
      </div>

      <div className="top-actions">
        <span className={`status-pill ${isOnline ? 'online' : 'offline'}`}>
          <Activity size={14} />
          {isOnline ? 'Backend API Ready' : 'Backend Unreachable'}
        </span>

        <span className={`badge ${isLlmAvailable ? 'tone-purple' : 'tone-amber'}`}>
          <Cpu size={13} />
          {isLlmAvailable ? `${providerName} LLM (${modelName})` : 'Deterministic Fallback'}
        </span>

        <span className={`badge ${hasOcr ? 'tone-green' : 'tone-amber'}`}>
          {hasOcr ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}
          {hasOcr ? 'Tesseract OCR Ready' : 'OCR Unconfigured'}
        </span>
      </div>
    </header>
  );
}
