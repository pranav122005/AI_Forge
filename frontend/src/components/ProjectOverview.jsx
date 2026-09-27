import React from 'react';
import { CheckCircle2, AlertTriangle, Cpu, Layers, FileCode, CheckSquare } from 'lucide-react';

export function ProjectOverview({ projectData }) {
  if (!projectData) return null;

  const {
    project_id,
    project_name,
    requirement,
    plan,
    execution_status,
    lifecycle_status,
    selected_components = [],
    implemented_components = [],
    catalog_components = [],
    generated_files = [],
    training,
  } = projectData;

  const isReady = execution_status === 'ready';
  const isPartial = execution_status === 'partial';

  return (
    <section className="panel project-panel">
      <div className="panel-head">
        <div>
          <div className="kicker">02 · GENERATED PROJECT METADATA</div>
          <h2>{project_name || project_id}</h2>
          <code className="project-id-tag">ID: {project_id}</code>
        </div>
        <div className="badge-group">
          <span className={`badge ${isReady ? 'tone-green' : isPartial ? 'tone-amber' : 'tone-red'}`}>
            <CheckCircle2 size={14} /> Execution Status: {execution_status.toUpperCase()}
          </span>
          <span className="badge tone-purple">
            Lifecycle: {lifecycle_status || 'BUILT'}
          </span>
        </div>
      </div>

      {requirement && (
        <div className="spec-breakdown-card">
          <div className="card-title"><Cpu size={16} /> LLM Requirement Analysis</div>
          <p className="spec-goal">"{requirement.goal}"</p>

          <div className="capability-flags">
            <span className={`flag ${requirement.requires_vision ? 'active' : ''}`}>
              Vision Processing: {requirement.requires_vision ? 'Yes' : 'No'}
            </span>
            <span className={`flag ${requirement.requires_ocr ? 'active' : ''}`}>
              OCR Extraction: {requirement.requires_ocr ? 'Yes' : 'No'}
            </span>
            <span className={`flag ${requirement.requires_classification ? 'active' : ''}`}>
              Text Classifier: {requirement.requires_classification ? 'Yes' : 'No'}
            </span>
            <span className={`flag ${requirement.requires_api ? 'active' : ''}`}>
              REST API: {requirement.requires_api ? 'Yes' : 'No'}
            </span>
          </div>
        </div>
      )}

      {plan?.warnings && plan.warnings.length > 0 && (
        <div className="warning-box">
          <AlertTriangle size={16} />
          <div>
            <strong>Architecture Warnings:</strong>
            <ul>
              {plan.warnings.map((w, idx) => (
                <li key={idx}>{w}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      <div className="meta-grid">
        <div className="meta-stat">
          <Layers size={18} />
          <div>
            <span>Selected Components</span>
            <strong>{selected_components.length} Total</strong>
          </div>
        </div>
        <div className="meta-stat">
          <CheckSquare size={18} />
          <div>
            <span>Executable in MVP</span>
            <strong>{implemented_components.length} Ready</strong>
          </div>
        </div>
        <div className="meta-stat">
          <FileCode size={18} />
          <div>
            <span>Generated Files</span>
            <strong>{generated_files.length} Files</strong>
          </div>
        </div>
      </div>
    </section>
  );
}
