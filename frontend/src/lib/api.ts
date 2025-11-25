/**
 * API client for backend communication
 */

import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Sessions API
export const sessionsApi = {
  list: (limit = 20, offset = 0) =>
    api.get(`/sessions`, { params: { limit, offset } }).then((r) => r.data),

  get: (id: string) =>
    api.get(`/sessions/${id}`).then((r) => r.data),

  create: (data: { name?: string; input_data: string; start_agent?: string }) =>
    api.post(`/sessions`, data).then((r) => r.data),

  update: (id: string, data: { name?: string }) =>
    api.patch(`/sessions/${id}`, data).then((r) => r.data),

  delete: (id: string) =>
    api.delete(`/sessions/${id}`).then((r) => r.data),

  getStats: () =>
    api.get(`/sessions/stats/overview`).then((r) => r.data),
};

// Pipeline API
export const pipelineApi = {
  start: (sessionId: string, startAgent?: string, priority?: number) =>
    api
      .post(`/pipeline/${sessionId}/start`, { start_agent: startAgent, priority })
      .then((r) => r.data),

  resume: (sessionId: string, fromAgent: string, priority?: number) =>
    api
      .post(`/pipeline/${sessionId}/resume`, null, {
        params: { from_agent: fromAgent, priority },
      })
      .then((r) => r.data),

  cancel: (sessionId: string) =>
    api.post(`/pipeline/${sessionId}/cancel`).then((r) => r.data),

  getStatus: (sessionId: string) =>
    api.get(`/pipeline/${sessionId}/status`).then((r) => r.data),
};

// Agents API
export const agentsApi = {
  getInfo: () =>
    api.get(`/agents/info`).then((r) => r.data),

  getOutput: (sessionId: string, agentName: string) =>
    api.get(`/agents/${sessionId}/${agentName}`).then((r) => r.data),

  getAllOutputs: (sessionId: string) =>
    api.get(`/agents/${sessionId}`).then((r) => r.data),
};

// Queue API
export const queueApi = {
  getStatus: () =>
    api.get(`/queue/status`).then((r) => r.data),

  getTasks: (status?: string, limit?: number) =>
    api.get(`/queue/tasks`, { params: { status, limit } }).then((r) => r.data),

  getPosition: (sessionId: string) =>
    api.get(`/queue/position/${sessionId}`).then((r) => r.data),

  getHistory: (limit?: number) =>
    api.get(`/queue/history`, { params: { limit } }).then((r) => r.data),
};

// Files API
export const filesApi = {
  getImageUrl: (sessionId: string, imagePath: string) =>
    `${API_BASE_URL}/files/${sessionId}/images/${imagePath}`,

  getVideoUrl: (sessionId: string, videoPath: string) =>
    `${API_BASE_URL}/files/${sessionId}/videos/${videoPath}`,

  listFiles: (sessionId: string) =>
    api.get(`/files/${sessionId}/list`).then((r) => r.data),

  downloadSession: (sessionId: string) =>
    `${API_BASE_URL}/files/${sessionId}/download`,
};
