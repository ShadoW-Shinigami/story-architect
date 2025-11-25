"use client";

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useQueueStatus, useQueueTasks, useQueueHistory } from "@/hooks/useQueue";
import { Sidebar } from "@/components/shared/Sidebar";
import { Header } from "@/components/shared/Header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { pipelineApi } from "@/lib/api";
import { formatDistanceToNow, format } from "date-fns";
import Link from "next/link";
import {
  Loader2,
  Play,
  Pause,
  XCircle,
  CheckCircle,
  Clock,
  RefreshCcw,
  AlertCircle,
  ListOrdered,
} from "lucide-react";

const statusConfig: Record<string, { color: string; icon: React.ReactNode; label: string }> = {
  pending: { color: "bg-yellow-500", icon: <Clock className="h-4 w-4" />, label: "Pending" },
  running: { color: "bg-blue-500", icon: <Loader2 className="h-4 w-4 animate-spin" />, label: "Running" },
  completed: { color: "bg-green-500", icon: <CheckCircle className="h-4 w-4" />, label: "Completed" },
  failed: { color: "bg-red-500", icon: <XCircle className="h-4 w-4" />, label: "Failed" },
  cancelled: { color: "bg-gray-500", icon: <AlertCircle className="h-4 w-4" />, label: "Cancelled" },
};

export default function QueuePage() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("pending");

  const { data: queueStatus, isLoading: statusLoading, refetch: refetchStatus } = useQueueStatus();
  const { data: pendingTasks, isLoading: tasksLoading } = useQueueTasks("pending", 50);
  const { data: historyTasks, isLoading: historyLoading } = useQueueHistory(50);

  const cancelMutation = useMutation({
    mutationFn: (sessionId: string) => pipelineApi.cancel(sessionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["queue"] });
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });

  const handleRefresh = () => {
    refetchStatus();
    queryClient.invalidateQueries({ queryKey: ["queue"] });
  };

  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header title="Queue Management" />
        <main className="flex-1 overflow-y-auto p-6">
          <div className="max-w-6xl mx-auto space-y-6">
            {/* Queue Status Overview */}
            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center gap-2">
                    {queueStatus?.is_processing ? (
                      <div className="h-3 w-3 rounded-full bg-green-500 animate-pulse" />
                    ) : (
                      <div className="h-3 w-3 rounded-full bg-gray-400" />
                    )}
                    Pipeline Status
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {statusLoading ? (
                      <Loader2 className="h-6 w-6 animate-spin" />
                    ) : queueStatus?.is_processing ? (
                      "Processing"
                    ) : (
                      "Idle"
                    )}
                  </div>
                  {queueStatus?.current_task && (
                    <p className="text-sm text-muted-foreground mt-1">
                      Session: {queueStatus.current_task.session_id.slice(0, 8)}...
                    </p>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <ListOrdered className="h-4 w-4" />
                    Pending Tasks
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {statusLoading ? (
                      <Loader2 className="h-6 w-6 animate-spin" />
                    ) : (
                      queueStatus?.pending_count ?? 0
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground mt-1">
                    In queue
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Actions</CardTitle>
                </CardHeader>
                <CardContent className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleRefresh}
                    className="flex-1"
                  >
                    <RefreshCcw className="mr-2 h-4 w-4" />
                    Refresh
                  </Button>
                </CardContent>
              </Card>
            </div>

            {/* Current Task */}
            {queueStatus?.current_task && (
              <Card className="border-blue-500">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span className="flex items-center gap-2">
                      <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
                      Currently Processing
                    </span>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={() => cancelMutation.mutate(queueStatus.current_task!.session_id)}
                      disabled={cancelMutation.isPending}
                    >
                      {cancelMutation.isPending ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <XCircle className="mr-2 h-4 w-4" />
                      )}
                      Cancel
                    </Button>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <div>
                      <Link
                        href={`/projects/${queueStatus.current_task.session_id}`}
                        className="font-medium text-blue-600 hover:underline"
                      >
                        Session: {queueStatus.current_task.session_id.slice(0, 8)}...
                      </Link>
                      {queueStatus.current_task.started_at && (
                        <p className="text-sm text-muted-foreground">
                          Started {formatDistanceToNow(new Date(queueStatus.current_task.started_at), { addSuffix: true })}
                        </p>
                      )}
                    </div>
                    <Link href={`/projects/${queueStatus.current_task.session_id}`}>
                      <Button variant="outline" size="sm">
                        View Progress
                      </Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Task Lists */}
            <Card>
              <CardHeader>
                <CardTitle>Task Queue</CardTitle>
                <CardDescription>
                  View pending and completed pipeline tasks
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Tabs value={activeTab} onValueChange={setActiveTab}>
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="pending">
                      Pending ({pendingTasks?.length ?? 0})
                    </TabsTrigger>
                    <TabsTrigger value="history">
                      History
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="pending" className="mt-4">
                    {tasksLoading ? (
                      <div className="flex items-center justify-center py-8">
                        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                      </div>
                    ) : pendingTasks && pendingTasks.length > 0 ? (
                      <div className="space-y-2">
                        {pendingTasks.map((task, index) => (
                          <div
                            key={task.id}
                            className="flex items-center justify-between p-4 rounded-lg border"
                          >
                            <div className="flex items-center gap-4">
                              <div className="flex items-center justify-center w-8 h-8 rounded-full bg-muted text-sm font-medium">
                                {index + 1}
                              </div>
                              <div>
                                <Link
                                  href={`/projects/${task.session_id}`}
                                  className="font-medium hover:underline"
                                >
                                  Session: {task.session_id.slice(0, 8)}...
                                </Link>
                                <p className="text-sm text-muted-foreground">
                                  Starting from: {task.start_agent}
                                  {task.resume_from && ` (resume from ${task.resume_from})`}
                                </p>
                              </div>
                            </div>
                            <div className="flex items-center gap-4">
                              <Badge variant="outline">
                                Priority: {task.priority}
                              </Badge>
                              <span className="text-sm text-muted-foreground">
                                {formatDistanceToNow(new Date(task.created_at), { addSuffix: true })}
                              </span>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => cancelMutation.mutate(task.session_id)}
                                disabled={cancelMutation.isPending}
                              >
                                <XCircle className="h-4 w-4 text-destructive" />
                              </Button>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-muted-foreground">
                        <Clock className="h-12 w-12 mx-auto mb-4 opacity-50" />
                        <p>No pending tasks</p>
                        <p className="text-sm mt-1">Start a new project to add tasks to the queue</p>
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="history" className="mt-4">
                    {historyLoading ? (
                      <div className="flex items-center justify-center py-8">
                        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                      </div>
                    ) : historyTasks && historyTasks.length > 0 ? (
                      <div className="space-y-2">
                        {historyTasks.map((task) => {
                          const config = statusConfig[task.status] || statusConfig.pending;
                          return (
                            <div
                              key={task.id}
                              className="flex items-center justify-between p-4 rounded-lg border"
                            >
                              <div className="flex items-center gap-4">
                                <div className={`w-2 h-2 rounded-full ${config.color}`} />
                                <div>
                                  <Link
                                    href={`/projects/${task.session_id}`}
                                    className="font-medium hover:underline"
                                  >
                                    Session: {task.session_id.slice(0, 8)}...
                                  </Link>
                                  {task.error_message && (
                                    <p className="text-sm text-red-500 truncate max-w-md">
                                      {task.error_message}
                                    </p>
                                  )}
                                </div>
                              </div>
                              <div className="flex items-center gap-4">
                                <Badge variant={task.status === "completed" ? "default" : task.status === "failed" ? "destructive" : "secondary"}>
                                  {config.icon}
                                  <span className="ml-1">{config.label}</span>
                                </Badge>
                                <span className="text-sm text-muted-foreground">
                                  {task.completed_at
                                    ? format(new Date(task.completed_at), "MMM d, HH:mm")
                                    : format(new Date(task.created_at), "MMM d, HH:mm")}
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-muted-foreground">
                        <CheckCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                        <p>No history yet</p>
                        <p className="text-sm mt-1">Completed tasks will appear here</p>
                      </div>
                    )}
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    </div>
  );
}
