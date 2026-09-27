import React, { useState } from 'react';
import { ArrowRight, WandSparkles, Loader2, Sparkles, FolderGit2 } from 'lucide-react';

const STARTER_PROMPTS = [
  {
    label: 'Legal Document Classifier + OCR',
    prompt: 'Build an AI system that extracts text from scanned legal images using OCR, classifies them into categories like Petition, Affidavit, and Notice, and exposes a REST API.',
    name: 'legal-ocr-classifier',
  },
  {
    label: 'Text Classification Service',
    prompt: 'Build a text classification service to categorize incoming customer support tickets into Billing, Technical Support, and Account Inquiries with a REST API.',
    name: 'support-ticket-classifier',
  },
  {
    label: 'Image Preprocessing Pipeline',
    prompt: 'Process image input using OpenCV denoising and binarization, extract text with Tesseract OCR, and serve predictions over a FastAPI endpoint.',
    name: 'vision-ocr-pipeline',
  },
];

export function GenerateForm({ onGenerate, busy, error }) {
  const [requirement, setRequirement] = useState(STARTER_PROMPTS[0].prompt);
  const [projectName, setProjectName] = useState(STARTER_PROMPTS[0].name);

  function handleSubmit(e) {
    e.preventDefault();
    if (!requirement.trim() || busy) return;
    onGenerate(requirement.trim(), projectName.trim());
  }

  function handleSelectPreset(preset) {
    setRequirement(preset.prompt);
    setProjectName(preset.name);
  }

  return (
    <section className="panel spec-panel">
      <div className="panel-head">
        <div>
          <div className="kicker">01 · NATURAL LANGUAGE REQUIREMENT</div>
          <h2>Describe your AI System</h2>
        </div>
        <WandSparkles className="panel-icon" size={24} />
      </div>

      <form onSubmit={handleSubmit}>
        <div className="preset-bar">
          <span className="preset-label"><Sparkles size={13} /> Sample Presets:</span>
          {STARTER_PROMPTS.map((preset) => (
            <button
              type="button"
              key={preset.name}
              className="preset-btn"
              onClick={() => handleSelectPreset(preset)}
            >
              {preset.label}
            </button>
          ))}
        </div>

        <div className="form-group">
          <textarea
            rows={4}
            value={requirement}
            onChange={(e) => setRequirement(e.target.value)}
            placeholder="e.g. Build an AI system that classifies legal documents into categories using OCR and exposes a REST API..."
            disabled={!!busy}
          />
        </div>

        <div className="field-row">
          <label>
            <FolderGit2 size={15} /> Project Name / Identifier
            <input
              type="text"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              placeholder="my-ai-project"
              disabled={!!busy}
            />
          </label>
        </div>

        {error && (
          <div className="error-banner">
            <span>{error}</span>
          </div>
        )}

        <button
          type="submit"
          className="primary-btn full-width"
          disabled={!!busy || requirement.trim().length < 5}
        >
          {busy === 'generate' ? (
            <>
              <Loader2 className="spin" size={18} /> Generating Project Scaffold…
            </>
          ) : (
            <>
              Generate AI Project <ArrowRight size={18} />
            </>
          )}
        </button>
      </form>
    </section>
  );
}
