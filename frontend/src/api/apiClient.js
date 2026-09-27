/**
 * AIForge Frontend API Client Service
 * ===================================
 * Centralized service layer for interacting with AIForge backend REST APIs.
 */

const API_BASE = (import.meta.env && import.meta.env.VITE_API_BASE_URL) || 'https://ai-forge-by-team-agni.onrender.com';

export async function extractErrorMessage(response, defaultMsg = 'An unexpected error occurred.') {
  if (!response) return defaultMsg;
  const status = response.status;
  try {
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      const data = await response.json();
      if (Array.isArray(data.detail)) {
        return data.detail.map((err) => err.msg || JSON.stringify(err)).join('; ');
      }
      if (typeof data.detail === 'string') {
        if (status === 404 && data.detail.toLowerCase() === 'not found') {
          return 'Generation endpoint unavailable (404 Not Found). Verify backend API service is active.';
        }
        return data.detail;
      }
      if (data.message) {
        return data.message;
      }
      if (data.error) {
        return typeof data.error === 'string' ? data.error : JSON.stringify(data.error);
      }
      return JSON.stringify(data);
    }
    const text = await response.text();
    if (status === 404) {
      return 'Generation endpoint unavailable (404 Not Found). Verify backend API service is active.';
    }
    return text || `HTTP ${response.status}: ${response.statusText}`;
  } catch {
    if (status === 404) {
      return 'Generation endpoint unavailable (404 Not Found). Verify backend API service is active.';
    }
    return `HTTP ${response.status}: ${response.statusText}`;
  }
}

export const api = {
  async getHealth() {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (!res.ok) throw new Error(await extractErrorMessage(res));
      return await res.json();
    } catch (err) {
      if (err instanceof TypeError && err.message.toLowerCase().includes('fetch')) {
        return { status: 'offline', error: 'Backend API unreachable' };
      }
      throw err;
    }
  },

  async getCapabilities() {
    try {
      const res = await fetch(`${API_BASE}/api/capabilities`);
      if (!res.ok) throw new Error(await extractErrorMessage(res));
      return await res.json();
    } catch {
      return null;
    }
  },

  async getCatalog() {
    const res = await fetch(`${API_BASE}/api/catalog`);
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async parseLLM(requirement) {
    const res = await fetch(`${API_BASE}/api/llm/parse`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ requirement }),
    });
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async generateProject(requirement, projectName) {
    const res = await fetch(`${API_BASE}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        requirement,
        project_name: projectName || 'aiforge-project',
      }),
    });
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async trainModel(projectId, fileObject = null) {
    let trainingFile = fileObject;
    if (!trainingFile) {
      const demoRes = await fetch(`${API_BASE}/demo/legal_demo.csv`);
      if (!demoRes.ok) {
        throw new Error(`Demo dataset not available (HTTP ${demoRes.status}). Upload a custom CSV dataset.`);
      }
      const blob = await demoRes.blob();
      trainingFile = new File([blob], 'legal_demo.csv', { type: 'text/csv' });
    }

    const formData = new FormData();
    formData.append('dataset', trainingFile);

    const res = await fetch(`${API_BASE}/api/train?project_id=${encodeURIComponent(projectId)}`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async predictText(projectId, text) {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id: projectId, text }),
    });
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async predictImage(projectId, imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);

    const res = await fetch(`${API_BASE}/api/predict-image?project_id=${encodeURIComponent(projectId)}`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async runAgent(projectId, inputPayload) {
    const res = await fetch(`${API_BASE}/api/agent/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_id: projectId, input: inputPayload }),
    });
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  // --- Phase 9 Real LLM Codegen APIs ---
  async generateCodegen(requirement, projectName, extraInstructions, provider, model) {
    const res = await fetch(`${API_BASE}/api/codegen/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        requirement_text: requirement,
        project_name: projectName || 'aiforge-project',
        extra_instructions: extraInstructions,
        provider: provider || undefined,
        model: model || undefined,
      }),
    });
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async getCodegenFiles(projectId) {
    const res = await fetch(`${API_BASE}/api/codegen/${encodeURIComponent(projectId)}/files`);
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async getCodegenFileContent(projectId, filePath) {
    const res = await fetch(`${API_BASE}/api/codegen/${encodeURIComponent(projectId)}/file?path=${encodeURIComponent(filePath)}`);
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async testCodegenProject(projectId) {
    const res = await fetch(`${API_BASE}/api/codegen/${encodeURIComponent(projectId)}/test`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async repairCodegenProject(projectId, testOutput, userFeedback) {
    const res = await fetch(`${API_BASE}/api/codegen/${encodeURIComponent(projectId)}/repair`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        project_id: projectId,
        test_output: testOutput,
        user_feedback: userFeedback,
      }),
    });
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  getCodegenDownloadUrl(projectId) {
    return `${API_BASE}/api/codegen/${encodeURIComponent(projectId)}/download`;
  },

  async modifyCodegenProject(projectId, instruction, provider, model) {
    const res = await fetch(`${API_BASE}/api/codegen/${encodeURIComponent(projectId)}/modify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        project_id: projectId,
        instruction: instruction,
        provider: provider || undefined,
        model: model || undefined,
      }),
    });
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async runCodegenAgent(projectId, instruction, provider, model) {
    const res = await fetch(`${API_BASE}/api/codegen/${encodeURIComponent(projectId)}/agent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instruction: instruction,
        provider: provider || undefined,
        model: model || undefined,
      }),
    });
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async getCodegenAgentState(projectId, runId) {
    const res = await fetch(`${API_BASE}/api/codegen/${encodeURIComponent(projectId)}/agent/${encodeURIComponent(runId)}`);
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async getCodegenReport(projectId) {
    const res = await fetch(`${API_BASE}/api/codegen/${encodeURIComponent(projectId)}/report`);
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  // --- Multi-LLM Provider APIs ---
  async getLLMProviders() {
    const res = await fetch(`${API_BASE}/api/llm/providers`);
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async getLLMActive() {
    const res = await fetch(`${API_BASE}/api/llm/active`);
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async setLLMActive(provider, model) {
    const res = await fetch(`${API_BASE}/api/llm/active`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider, model: model || undefined }),
    });
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async configureLLMProvider(provider, config = {}) {
    const res = await fetch(`${API_BASE}/api/llm/providers/${encodeURIComponent(provider)}/configure`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },

  async testLLMProvider(provider, config = {}) {
    const res = await fetch(`${API_BASE}/api/llm/providers/${encodeURIComponent(provider)}/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!res.ok) throw new Error(await extractErrorMessage(res));
    return await res.json();
  },
};

