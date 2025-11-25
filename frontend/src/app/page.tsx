"use client";

import Link from "next/link";
import { useSessions, useDashboardStats } from "@/hooks/useSessions";
import { useQueueStatus } from "@/hooks/useQueue";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Sidebar } from "@/components/shared/Sidebar";
import { Header } from "@/components/shared/Header";
import { QueueStatus } from "@/components/queue/QueueStatus";
import { formatDistanceToNow } from "date-fns";
import { Plus, FolderOpen, Clock, CheckCircle, XCircle, Loader2, TrendingUp, Activity, Film } from "lucide-react";

const statusColors: Record<string, string> = {
  pending: "bg-gray-500",
  queued: "bg-yellow-500",
  in_progress: "bg-blue-500",
  completed: "bg-green-500",
  failed: "bg-red-500",
  cancelled: "bg-gray-400",
};

const statusIcons: Record<string, React.ReactNode> = {
  pending: <Clock className="h-4 w-4" />,
  queued: <Loader2 className="h-4 w-4 animate-spin" />,
  in_progress: <Loader2 className="h-4 w-4 animate-spin" />,
  completed: <CheckCircle className="h-4 w-4" />,
  failed: <XCircle className="h-4 w-4" />,
};

export default function Dashboard() {
  const { data: sessions, isLoading } = useSessions(5);
  const { data: queueStatus } = useQueueStatus();
  const { data: stats, isLoading: statsLoading } = useDashboardStats();

  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header title="Dashboard" />
        <main className="flex-1 overflow-y-auto p-6">
          <div className="max-w-6xl mx-auto space-y-6">
            {/* Stats Overview */}
            <div className="grid gap-4 md:grid-cols-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Projects</CardTitle>
                  <Film className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {statsLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : stats?.total_projects ?? 0}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {stats?.recent_projects ?? 0} created this week
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-blue-600">
                    {statsLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : stats?.active_count ?? 0}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Currently processing
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Completed</CardTitle>
                  <CheckCircle className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-green-600">
                    {statsLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : stats?.completed_count ?? 0}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Successfully finished
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Failed</CardTitle>
                  <XCircle className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-red-600">
                    {statsLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : stats?.failed_count ?? 0}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Need attention
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Quick Actions */}
            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">New Project</CardTitle>
                  <CardDescription>
                    Start a new script-to-video pipeline
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Link href="/projects/new">
                    <Button className="w-full">
                      <Plus className="mr-2 h-4 w-4" />
                      Create Project
                    </Button>
                  </Link>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Queue Status</CardTitle>
                  <CardDescription>Current pipeline execution</CardDescription>
                </CardHeader>
                <CardContent>
                  <QueueStatus compact />
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">All Projects</CardTitle>
                  <CardDescription>Browse and manage sessions</CardDescription>
                </CardHeader>
                <CardContent>
                  <Link href="/projects">
                    <Button variant="outline" className="w-full">
                      <FolderOpen className="mr-2 h-4 w-4" />
                      View All
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            </div>

            {/* Recent Sessions */}
            <Card>
              <CardHeader>
                <CardTitle>Recent Projects</CardTitle>
                <CardDescription>
                  Your most recently updated projects
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                  </div>
                ) : sessions && sessions.length > 0 ? (
                  <div className="space-y-4">
                    {sessions.map((session) => (
                      <Link
                        key={session.id}
                        href={`/projects/${session.id}`}
                        className="block"
                      >
                        <div className="flex items-center justify-between p-4 rounded-lg border hover:bg-muted/50 transition-colors">
                          <div className="flex items-center gap-4">
                            <div
                              className={`w-2 h-2 rounded-full ${statusColors[session.status]}`}
                            />
                            <div>
                              <p className="font-medium">
                                {session.name || `Project ${session.id.slice(0, 8)}`}
                              </p>
                              <p className="text-sm text-muted-foreground">
                                {session.current_agent
                                  ? `Current: ${session.current_agent}`
                                  : session.status}
                              </p>
                            </div>
                          </div>
                          <div className="flex items-center gap-4">
                            <Badge variant="outline">
                              {statusIcons[session.status]}
                              <span className="ml-1 capitalize">
                                {session.status.replace("_", " ")}
                              </span>
                            </Badge>
                            <span className="text-sm text-muted-foreground">
                              {formatDistanceToNow(new Date(session.updated_at), {
                                addSuffix: true,
                              })}
                            </span>
                          </div>
                        </div>
                      </Link>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <p>No projects yet</p>
                    <Link href="/projects/new">
                      <Button variant="link">Create your first project</Button>
                    </Link>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    </div>
  );
}
