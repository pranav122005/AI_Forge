import React, { useState } from 'react';
import {
  IconCopy,
  IconCheck,
  IconCode,
  IconFile,
} from '../common/Icons';

export const CodeViewer = ({ filePath, content, loading }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!content) return;
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text', err);
    }
  };

  const getLanguageLabel = (path) => {
    if (!path) return 'TEXT';
    const ext = path.split('.').pop().toLowerCase();
    switch (ext) {
      case 'py': return 'PYTHON';
      case 'js': return 'JAVASCRIPT';
      case 'jsx': return 'REACT JSX';
      case 'json': return 'JSON';
      case 'md': return 'MARKDOWN';
      case 'txt': return 'PLAIN TEXT';
      case 'sh': return 'SHELL';
      case 'yml':
      case 'yaml': return 'YAML';
      case 'toml': return 'TOML';
      case 'html': return 'HTML';
      case 'css': return 'CSS';
      default: return ext.toUpperCase();
    }
  };

  if (loading) {
    return (
      <div className="code-viewer">
        <div className="code-loading">
          <span className="icon-pulse">LOADING SOURCE CODE...</span>
        </div>
      </div>
    );
  }

  if (!filePath) {
    return (
      <div className="code-viewer">
        <div className="code-empty">
          <IconFile size={28} className="code-empty-icon" />
          <p>SELECT A FILE FROM THE EXPLORER TO INSPECT SOURCE</p>
        </div>
      </div>
    );
  }

  const lines = typeof content === 'string' ? content.split('\n') : [];
  const lang = getLanguageLabel(filePath);

  return (
    <div className="code-viewer">
      <div className="code-header">
        <div className="code-header-left">
          <IconCode size={14} className="code-file-icon" />
          <span className="code-path">{filePath}</span>
          <span className="code-lang-badge">{lang}</span>
          <span className="code-line-count">{lines.length} lines</span>
        </div>
        <div className="code-actions">
          <button
            className={`code-copy-btn ${copied ? 'code-copy-btn-success' : ''}`}
            onClick={handleCopy}
            aria-label="Copy code to clipboard"
          >
            {copied ? (
              <>
                <IconCheck size={12} />
                <span>COPIED</span>
              </>
            ) : (
              <>
                <IconCopy size={12} />
                <span>COPY</span>
              </>
            )}
          </button>
        </div>
      </div>
      <div className="code-body">
        <div className="code-lines" aria-hidden="true">
          {lines.map((_, i) => (
            <div key={i} className="code-line-number">{i + 1}</div>
          ))}
        </div>
        <pre className="code-content">
          <code>{content}</code>
        </pre>
      </div>
    </div>
  );
};
