"use client";

import { TextOutput } from "./TextOutput";
import { JsonViewer } from "./JsonViewer";
import { ImageGallery } from "./ImageGallery";
import { VideoPlayer } from "./VideoPlayer";

interface OutputViewerProps {
  agentName: string;
  output: any;
  sessionId: string;
}

export function OutputViewer({ agentName, output, sessionId }: OutputViewerProps) {
  if (!output) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No output data available
      </div>
    );
  }

  // Route to appropriate viewer based on agent
  switch (agentName) {
    case "agent_1":
      // Screenplay - text output
      const screenplayText = typeof output === "string" ? output : output?.text;
      return <TextOutput content={screenplayText} title="Screenplay" />;

    case "agent_2":
      // Scene breakdown - JSON
      return (
        <JsonViewer
          data={output}
          title="Scene Breakdown"
          summary={`${output?.scenes?.length || 0} scenes`}
        />
      );

    case "agent_3":
      // Shot breakdown - JSON
      return (
        <JsonViewer
          data={output}
          title="Shot Breakdown"
          summary={`${output?.shots?.length || 0} shots`}
        />
      );

    case "agent_4":
      // Shot grouping - JSON
      return (
        <JsonViewer
          data={output}
          title="Shot Grouping"
          summary={`${output?.total_parent_shots || 0} parent shots, ${
            output?.total_child_shots || 0
          } child shots`}
        />
      );

    case "agent_5":
      // Character creation - images + JSON
      return (
        <div className="space-y-6">
          <ImageGallery
            images={output?.characters?.map((c: any) => ({
              src: `/api/files/${sessionId}/images/${c.image_path}`,
              alt: c.name,
              caption: c.name,
            }))}
            title="Characters"
          />
          <JsonViewer
            data={output}
            title="Character Data"
            collapsed
          />
        </div>
      );

    case "agent_6":
    case "agent_7":
      // Parent shots - images
      return (
        <ImageGallery
          images={output?.parent_shots?.map((s: any) => ({
            src: `/api/files/${sessionId}/images/${s.image_path}`,
            alt: s.shot_id,
            caption: `${s.shot_id} - ${s.verification_status || "pending"}`,
          }))}
          title="Parent Shot Images"
        />
      );

    case "agent_8":
    case "agent_9":
      // Child shots - images
      return (
        <ImageGallery
          images={output?.child_shots?.map((s: any) => ({
            src: `/api/files/${sessionId}/images/${s.image_path}`,
            alt: s.shot_id,
            caption: `${s.shot_id} - ${s.verification_status || "pending"}`,
          }))}
          title="Child Shot Images"
        />
      );

    case "agent_10":
      // Video generation - video grid
      return (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {output?.videos?.map((v: any) => (
              <VideoPlayer
                key={v.shot_id}
                src={v.video_path ? `/api/files/${sessionId}/videos/${v.video_path}` : v.video_url}
                title={v.shot_id}
              />
            ))}
          </div>
          <JsonViewer
            data={output}
            title="Video Generation Data"
            collapsed
          />
        </div>
      );

    case "agent_11":
      // Final video
      return (
        <div className="space-y-6">
          {output?.master_video_path && (
            <div>
              <h4 className="font-medium mb-2">Master Video</h4>
              <VideoPlayer
                src={`/api/files/${sessionId}/videos/${output.master_video_path}`}
                title="Final Video"
              />
            </div>
          )}
          {output?.scene_videos?.length > 0 && (
            <div>
              <h4 className="font-medium mb-2">Scene Videos</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {output.scene_videos.map((s: any) => (
                  <VideoPlayer
                    key={s.scene_id}
                    src={`/api/files/${sessionId}/videos/${s.video_path}`}
                    title={s.scene_id}
                  />
                ))}
              </div>
            </div>
          )}
          <JsonViewer
            data={output?.edit_timeline}
            title="Edit Timeline"
            collapsed
          />
        </div>
      );

    default:
      // Fallback to JSON viewer
      return <JsonViewer data={output} title={`${agentName} Output`} />;
  }
}
