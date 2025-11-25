"use client";

import { useState } from "react";
import Link from "next/link";
import { useSessions, useDeleteSession } from "@/hooks/useSessions";
import { Header } from "@/components/shared/Header";
import { Sidebar } from "@/components/shared/Sidebar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Plus,
  Loader2,
  FolderOpen,
  Calendar,
  Trash2,
  Play,
  Eye,
} from "lucide-react";
import { SessionListItem, SessionStatus } from "@/types/session";

const statusColors: Record<SessionStatus, string> = {
  pending: "bg-gray-500",
  queued: "bg-yellow-500",
  in_progress: "bg-blue-500",
  completed: "bg-green-500",
  failed: "bg-red-500",
  cancelled: "bg-gray-400",
};

const statusLabels: Record<SessionStatus, string> = {
  pending: "Pending",
  queued: "Queued",
  in_progress: "Processing",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

function ProjectCard({ session }: { session: SessionListItem & { input_data?: string } }) {
  const deleteSession = useDeleteSession();
  const [isDeleting, setIsDeleting] = useState(false);

  const handleDelete = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (!confirm("Are you sure you want to delete this project?")) return;

    setIsDeleting(true);
    try {
      await deleteSession.mutateAsync(session.id);
    } finally {
      setIsDeleting(false);
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getProgressPercent = () => {
    if (!session.current_agent) return 0;
    const agentNum = parseInt(session.current_agent.replace("agent_", ""));
    if (session.status === "completed") return 100;
    return Math.round((agentNum / 11) * 100);
  };

  return (
    <Card className="group hover:shadow-md transition-shadow">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <CardTitle className="text-lg truncate">
              {session.name || `Project ${session.id.slice(0, 8)}`}
            </CardTitle>
            <div className="flex items-center gap-2 mt-1 text-sm text-muted-foreground">
              <Calendar className="h-3 w-3" />
              <span>{formatDate(session.created_at)}</span>
            </div>
          </div>
          <Badge className={statusColors[session.status]}>
            {statusLabels[session.status]}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {/* Progress bar */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Progress</span>
              <span>{getProgressPercent()}%</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div
                className={`h-full transition-all ${
                  session.status === "failed"
                    ? "bg-red-500"
                    : session.status === "completed"
                    ? "bg-green-500"
                    : "bg-blue-500"
                }`}
                style={{ width: `${getProgressPercent()}%` }}
              />
            </div>
          </div>

          {/* Current agent */}
          {session.current_agent && session.status === "in_progress" && (
            <div className="text-sm text-muted-foreground">
              Current: {session.current_agent.replace("_", " ").toUpperCase()}
            </div>
          )}

          {/* Input preview */}
          {session.input_data && (
            <p className="text-sm text-muted-foreground line-clamp-2">
              {session.input_data.slice(0, 150)}...
            </p>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 pt-2">
            <Link href={`/projects/${session.id}`} className="flex-1">
              <Button variant="outline" size="sm" className="w-full">
                <Eye className="h-4 w-4 mr-2" />
                View
              </Button>
            </Link>
            {session.status === "pending" && (
              <Link href={`/projects/${session.id}`}>
                <Button size="sm">
                  <Play className="h-4 w-4 mr-2" />
                  Start
                </Button>
              </Link>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={handleDelete}
              disabled={isDeleting || session.status === "in_progress"}
              className="text-red-500 hover:text-red-600 hover:bg-red-50"
            >
              {isDeleting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Trash2 className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function ProjectsPage() {
  const { data: sessions, isLoading, error } = useSessions();

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header title="Projects" />
        <main className="flex-1 p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-2xl font-bold">Your Projects</h2>
              <p className="text-muted-foreground">
                Manage your story-to-video pipeline projects
              </p>
            </div>
            <Link href="/projects/new">
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                New Project
              </Button>
            </Link>
          </div>

          {/* Content */}
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : error ? (
            <Card className="p-8 text-center">
              <p className="text-red-500">Failed to load projects</p>
              <p className="text-sm text-muted-foreground mt-1">
                {error.message}
              </p>
            </Card>
          ) : sessions?.length === 0 ? (
            <Card className="p-12 text-center">
              <FolderOpen className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold">No projects yet</h3>
              <p className="text-muted-foreground mt-1 mb-4">
                Create your first project to get started
              </p>
              <Link href="/projects/new">
                <Button>
                  <Plus className="h-4 w-4 mr-2" />
                  Create Project
                </Button>
              </Link>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {sessions?.map((session) => (
                <ProjectCard key={session.id} session={session} />
              ))}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
