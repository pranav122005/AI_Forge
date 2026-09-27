import React, { useState } from 'react';
import {
  IconFolder,
  IconFile,
  IconChevronRight,
  IconChevronDown,
} from '../common/Icons';

const FileTreeNode = ({ node, selectedPath, onSelect, depth = 0 }) => {
  const [isOpen, setIsOpen] = useState(true);

  const indents = Array.from({ length: depth }).map((_, i) => (
    <span key={i} className="tree-indent"></span>
  ));

  if (node.is_dir || node.children) {
    return (
      <div className="tree-node">
        <div
          className="tree-folder"
          onClick={() => setIsOpen(!isOpen)}
          aria-label={`Folder: ${node.name}`}
        >
          {indents}
          <span className="tree-chevron">
            {isOpen ? <IconChevronDown size={12} /> : <IconChevronRight size={12} />}
          </span>
          <span className="tree-icon">
            <IconFolder size={14} className="icon-folder-svg" />
          </span>
          <span className="tree-folder-name">{node.name}</span>
        </div>
        {isOpen && node.children && (
          <div className="tree-children">
            {node.children.map((child, idx) => (
              <FileTreeNode
                key={idx}
                node={child}
                selectedPath={selectedPath}
                onSelect={onSelect}
                depth={depth + 1}
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  const isSelected = selectedPath === node.path;

  return (
    <div
      className={`tree-file ${isSelected ? 'tree-file-selected' : ''}`}
      onClick={() => onSelect(node.path)}
      aria-label={`File: ${node.name}`}
    >
      {indents}
      <span className="tree-spacer"></span>
      <span className="tree-icon">
        <IconFile size={14} className="icon-file-svg" />
      </span>
      <span className="tree-file-name">{node.name}</span>
    </div>
  );
};

export const FileTree = ({ tree, selectedPath, onSelect }) => {
  if (!tree) {
    return <div className="file-tree-empty">No files available.</div>;
  }

  const nodes = Array.isArray(tree) ? tree : [tree];

  return (
    <div className="file-tree">
      {nodes.map((node, idx) => (
        <FileTreeNode
          key={idx}
          node={node}
          selectedPath={selectedPath}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
};
