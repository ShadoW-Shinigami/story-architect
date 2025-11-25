/**
 * Utility functions
 */

import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge Tailwind CSS classes with clsx
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format duration in seconds to human-readable string
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  } else if (seconds < 3600) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
  }
}

/**
 * Format file size in bytes to human-readable string
 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  } else if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  } else if (bytes < 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  } else {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }
}

/**
 * Truncate string with ellipsis
 */
export function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength - 3) + "...";
}

/**
 * Get agent display name
 */
export function getAgentDisplayName(agentName: string): string {
  const names: Record<string, string> = {
    agent_1: "Screenplay Generator",
    agent_2: "Scene Breakdown",
    agent_3: "Shot Breakdown",
    agent_4: "Shot Grouping",
    agent_5: "Character Creator",
    agent_6: "Parent Image Generator",
    agent_7: "Parent Verification",
    agent_8: "Child Image Generator",
    agent_9: "Child Verification",
    agent_10: "Video Dialogue Generator",
    agent_11: "Intelligent Video Editor",
  };
  return names[agentName] || agentName;
}

/**
 * Get agent phase number
 */
export function getAgentPhase(agentName: string): number {
  const num = parseInt(agentName.replace("agent_", ""));
  if (num <= 4) return 1;
  if (num <= 9) return 2;
  return 3;
}
