"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Header } from "@/components/shared/Header";
import { Sidebar } from "@/components/shared/Sidebar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useCreateSession } from "@/hooks/useSessions";
import {
  ArrowLeft,
  ArrowRight,
  Loader2,
  FileText,
  Settings,
  Rocket,
} from "lucide-react";

type Step = "input" | "settings" | "review";
type StartAgent = "agent_1" | "agent_2";

interface FormData {
  name: string;
  inputData: string;
  startAgent: StartAgent;
}

export default function NewProjectPage() {
  const router = useRouter();
  const createSession = useCreateSession();
  const [step, setStep] = useState<Step>("input");
  const [formData, setFormData] = useState<FormData>({
    name: "",
    inputData: "",
    startAgent: "agent_1",
  });

  const steps: { id: Step; label: string; icon: React.ReactNode }[] = [
    { id: "input", label: "Input", icon: <FileText className="h-4 w-4" /> },
    { id: "settings", label: "Settings", icon: <Settings className="h-4 w-4" /> },
    { id: "review", label: "Review", icon: <Rocket className="h-4 w-4" /> },
  ];

  const currentStepIndex = steps.findIndex((s) => s.id === step);

  const canProceed = () => {
    if (step === "input") {
      return formData.inputData.trim().length >= 10;
    }
    return true;
  };

  const handleNext = () => {
    if (step === "input") setStep("settings");
    else if (step === "settings") setStep("review");
  };

  const handleBack = () => {
    if (step === "settings") setStep("input");
    else if (step === "review") setStep("settings");
  };

  const handleSubmit = async () => {
    try {
      const session = await createSession.mutateAsync({
        name: formData.name || undefined,
        input_data: formData.inputData,
        start_agent: formData.startAgent,
      });
      router.push(`/projects/${session.id}`);
    } catch (error) {
      console.error("Failed to create project:", error);
    }
  };

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header title="New Project" />
        <main className="flex-1 p-6">
          <div className="max-w-3xl mx-auto">
            {/* Back button */}
            <Button
              variant="ghost"
              className="mb-6"
              onClick={() => router.push("/projects")}
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Projects
            </Button>

            {/* Step indicator */}
            <div className="flex items-center justify-center mb-8">
              {steps.map((s, i) => (
                <div key={s.id} className="flex items-center">
                  <div
                    className={`flex items-center gap-2 px-4 py-2 rounded-full ${
                      s.id === step
                        ? "bg-primary text-primary-foreground"
                        : i < currentStepIndex
                        ? "bg-primary/20 text-primary"
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {s.icon}
                    <span className="hidden sm:inline">{s.label}</span>
                  </div>
                  {i < steps.length - 1 && (
                    <div
                      className={`w-12 h-0.5 mx-2 ${
                        i < currentStepIndex ? "bg-primary" : "bg-muted"
                      }`}
                    />
                  )}
                </div>
              ))}
            </div>

            {/* Step content */}
            <Card>
              {step === "input" && (
                <>
                  <CardHeader>
                    <CardTitle>Enter Your Story</CardTitle>
                    <CardDescription>
                      Provide a logline, story outline, or complete script to
                      transform into a video
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="name">Project Name (optional)</Label>
                      <Input
                        id="name"
                        placeholder="My Awesome Story"
                        value={formData.name}
                        onChange={(e) =>
                          setFormData({ ...formData, name: e.target.value })
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="input">Story Content *</Label>
                      <Textarea
                        id="input"
                        placeholder="Enter your story, logline, or script here...&#10;&#10;Example: A young chef discovers a secret family recipe that transports her back in time to 1920s Paris, where she must compete in a cooking competition to save her family's restaurant."
                        className="min-h-[300px] font-mono"
                        value={formData.inputData}
                        onChange={(e) =>
                          setFormData({ ...formData, inputData: e.target.value })
                        }
                      />
                      <p className="text-xs text-muted-foreground">
                        Minimum 10 characters. The more detail you provide, the
                        better the output.
                      </p>
                    </div>
                  </CardContent>
                </>
              )}

              {step === "settings" && (
                <>
                  <CardHeader>
                    <CardTitle>Pipeline Settings</CardTitle>
                    <CardDescription>
                      Configure how your story will be processed
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="space-y-4">
                      <Label>Starting Point</Label>
                      <RadioGroup
                        value={formData.startAgent}
                        onValueChange={(v) =>
                          setFormData({ ...formData, startAgent: v as StartAgent })
                        }
                      >
                        <div className="flex items-start space-x-3 p-4 border rounded-lg hover:bg-muted/50 cursor-pointer">
                          <RadioGroupItem value="agent_1" id="agent_1" />
                          <div className="flex-1">
                            <Label htmlFor="agent_1" className="cursor-pointer">
                              <span className="font-medium">
                                Start from Screenplay Generation
                              </span>
                            </Label>
                            <p className="text-sm text-muted-foreground mt-1">
                              Best for loglines, story outlines, or rough ideas.
                              The AI will generate a complete screenplay first.
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start space-x-3 p-4 border rounded-lg hover:bg-muted/50 cursor-pointer">
                          <RadioGroupItem value="agent_2" id="agent_2" />
                          <div className="flex-1">
                            <Label htmlFor="agent_2" className="cursor-pointer">
                              <span className="font-medium">
                                Start from Scene Breakdown
                              </span>
                            </Label>
                            <p className="text-sm text-muted-foreground mt-1">
                              Best if you already have a properly formatted
                              screenplay with INT./EXT. scene headings.
                            </p>
                          </div>
                        </div>
                      </RadioGroup>
                    </div>
                  </CardContent>
                </>
              )}

              {step === "review" && (
                <>
                  <CardHeader>
                    <CardTitle>Review & Create</CardTitle>
                    <CardDescription>
                      Verify your project settings before starting
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="space-y-4">
                      <div className="p-4 bg-muted rounded-lg">
                        <h4 className="font-medium mb-2">Project Name</h4>
                        <p className="text-sm text-muted-foreground">
                          {formData.name || "Untitled Project"}
                        </p>
                      </div>
                      <div className="p-4 bg-muted rounded-lg">
                        <h4 className="font-medium mb-2">Input Preview</h4>
                        <p className="text-sm text-muted-foreground whitespace-pre-wrap line-clamp-6">
                          {formData.inputData}
                        </p>
                        <p className="text-xs text-muted-foreground mt-2">
                          {formData.inputData.length} characters
                        </p>
                      </div>
                      <div className="p-4 bg-muted rounded-lg">
                        <h4 className="font-medium mb-2">Starting Agent</h4>
                        <p className="text-sm text-muted-foreground">
                          {formData.startAgent === "agent_1"
                            ? "Screenplay Generation (from scratch)"
                            : "Scene Breakdown (existing screenplay)"}
                        </p>
                      </div>
                    </div>

                    <div className="p-4 border border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950 rounded-lg">
                      <h4 className="font-medium text-blue-700 dark:text-blue-300 mb-1">
                        Pipeline Overview
                      </h4>
                      <p className="text-sm text-blue-600 dark:text-blue-400">
                        Your story will go through 11 AI agents: Screenplay →
                        Scene Breakdown → Shot Breakdown → Grouping → Character
                        Creation → Image Generation → Verification → Video
                        Generation → Final Edit
                      </p>
                    </div>
                  </CardContent>
                </>
              )}

              {/* Navigation */}
              <div className="flex items-center justify-between p-6 pt-0">
                <Button
                  variant="outline"
                  onClick={handleBack}
                  disabled={step === "input"}
                >
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Back
                </Button>
                {step === "review" ? (
                  <Button
                    onClick={handleSubmit}
                    disabled={createSession.isPending}
                  >
                    {createSession.isPending ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Creating...
                      </>
                    ) : (
                      <>
                        <Rocket className="h-4 w-4 mr-2" />
                        Create Project
                      </>
                    )}
                  </Button>
                ) : (
                  <Button onClick={handleNext} disabled={!canProceed()}>
                    Next
                    <ArrowRight className="h-4 w-4 ml-2" />
                  </Button>
                )}
              </div>
            </Card>
          </div>
        </main>
      </div>
    </div>
  );
}
