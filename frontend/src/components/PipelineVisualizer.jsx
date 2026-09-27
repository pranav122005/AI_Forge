import React from 'react';
import {
  Camera, ScanText, BrainCircuit, Sparkles, Network, Database, Server,
  ChevronRight, AlertTriangle, Layers3, CheckCircle2
} from 'lucide-react';

const ICON_MAP = {
  camera: Camera,
  scan: ScanText,
  brain: BrainCircuit,
  sparkles: Sparkles,
  network: Network,
  database: Database,
  server: Server,
};

export function PipelineVisualizer({ plan }) {
  if (!plan) return null;

  const components = plan.components || [];
  const pipelineSteps = plan.pipeline || [];

  return (
    <section className="panel pipeline-panel">
      <div className="panel-head">
        <div>
          <div className="kicker">03 · AI ARCHITECTURE & PIPELINE</div>
          <h2>Pipeline Step Visualization</h2>
        </div>
        <Layers3 className="panel-icon" size={24} />
      </div>

      <div className="pipeline-flow-container">
        {pipelineSteps.map((stepName, idx) => (
          <React.Fragment key={stepName}>
            <div className="pipeline-step-node">
              <div className="step-idx">{idx + 1}</div>
              <span className="step-name">{stepName}</span>
            </div>
            {idx < pipelineSteps.length - 1 && (
              <ChevronRight className="pipeline-arrow" size={20} />
            )}
          </React.Fragment>
        ))}
      </div>

      <div className="component-cards-list">
        <h3>Selected Component Architecture ({components.length})</h3>
        <div className="comp-grid">
          {components.map((comp) => {
            const IconComponent = ICON_MAP[comp.icon] || BrainCircuit;
            const isExec = comp.executable === true;

            return (
              <div className={`comp-card ${isExec ? 'executable' : 'catalog'}`} key={comp.key || comp.id}>
                <div className="comp-card-header">
                  <div className="comp-icon-wrapper">
                    <IconComponent size={18} />
                  </div>
                  <div>
                    <h4 className="comp-title">{comp.name}</h4>
                    <span className="comp-category">{comp.category}</span>
                  </div>
                </div>

                <p className="comp-desc">{comp.purpose || comp.description}</p>

                <div className="comp-card-footer">
                  <span className={`badge ${isExec ? 'tone-green' : 'tone-amber'}`}>
                    {isExec ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}
                    {isExec ? 'MVP Ready' : 'Catalog'}
                  </span>
                  {!isExec && (
                    <span className="catalog-note">Architecture only</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
