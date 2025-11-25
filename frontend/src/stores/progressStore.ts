/**
 * Zustand store for real-time progress tracking
 */

import { create } from "zustand";
import { ProgressEvent, PipelineCompletedEvent } from "@/types/progress";

interface AgentProgress {
  progress: number;
  message?: string;
  current?: number;
  total?: number;
}

interface ProgressState {
  // Current state
  events: ProgressEvent[];
  currentAgent: string | null;
  overallProgress: number;
  isRunning: boolean;
  currentMessage: string | null;

  // Agent-level progress tracking
  agents: Record<string, AgentProgress>;

  // Pipeline status
  pipelineStartedAt: Date | null;
  completedAgents: string[];

  // Actions
  addEvent: (event: ProgressEvent) => void;
  setAgentCompleted: (agentName: string) => void;
  setPipelineCompleted: (data: PipelineCompletedEvent) => void;
  setPipelineFailed: (error: string) => void;
  reset: () => void;
}

export const useProgressStore = create<ProgressState>((set, get) => ({
  events: [],
  currentAgent: null,
  overallProgress: 0,
  isRunning: false,
  currentMessage: null,
  agents: {},
  pipelineStartedAt: null,
  completedAgents: [],

  addEvent: (event) =>
    set((state) => {
      const newState: Partial<ProgressState> = {
        events: [...state.events.slice(-100), event], // Keep last 100 events
      };

      // Update current agent
      if (event.agent_name) {
        newState.currentAgent = event.agent_name;
      }

      // Update overall progress
      if (event.overall_progress !== undefined) {
        newState.overallProgress = event.overall_progress;
      }

      // Update current message
      if (event.message) {
        newState.currentMessage = event.message;
      }

      // Update agent-level progress
      if (event.agent_name && event.progress !== undefined) {
        const updatedAgents = { ...state.agents };
        updatedAgents[event.agent_name] = {
          progress: event.progress,
          message: event.message,
          current: event.step_current,
          total: event.step_total,
        };
        newState.agents = updatedAgents;
      }

      // Handle pipeline started
      if (event.event_type === "pipeline_started") {
        newState.isRunning = true;
        newState.pipelineStartedAt = new Date();
        newState.completedAgents = [];
        newState.agents = {};
      }

      // Handle agent completed
      if (event.event_type === "agent_completed" && event.agent_name) {
        const updatedAgents = { ...state.agents };
        updatedAgents[event.agent_name] = {
          progress: 1,
          message: "Completed",
        };
        newState.agents = updatedAgents;
        newState.completedAgents = [...state.completedAgents, event.agent_name];
      }

      // Handle pipeline completed/failed
      if (
        event.event_type === "pipeline_completed" ||
        event.event_type === "pipeline_failed"
      ) {
        newState.isRunning = false;
        newState.currentAgent = null;
        newState.currentMessage = null;
      }

      return newState;
    }),

  setAgentCompleted: (agentName) =>
    set((state) => ({
      completedAgents: [...state.completedAgents, agentName],
      currentAgent: null,
    })),

  setPipelineCompleted: (data) =>
    set({
      isRunning: false,
      currentAgent: null,
      overallProgress: 1,
      completedAgents: data.agents_completed,
    }),

  setPipelineFailed: (error) =>
    set({
      isRunning: false,
    }),

  reset: () =>
    set({
      events: [],
      currentAgent: null,
      overallProgress: 0,
      isRunning: false,
      currentMessage: null,
      agents: {},
      pipelineStartedAt: null,
      completedAgents: [],
    }),
}));
