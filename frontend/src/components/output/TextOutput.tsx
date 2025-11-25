"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Copy, Check, ChevronDown, ChevronUp } from "lucide-react";

interface TextOutputProps {
  content: string;
  title?: string;
  maxHeight?: number;
}

export function TextOutput({
  content,
  title,
  maxHeight = 500,
}: TextOutputProps) {
  const [copied, setCopied] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!content) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No content available
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {title && (
        <div className="flex items-center justify-between">
          <h4 className="font-medium">{title}</h4>
          <Button variant="ghost" size="sm" onClick={handleCopy}>
            {copied ? (
              <Check className="h-4 w-4 text-green-500" />
            ) : (
              <Copy className="h-4 w-4" />
            )}
          </Button>
        </div>
      )}
      <div className="relative">
        <pre
          className={`whitespace-pre-wrap font-mono text-sm bg-muted p-4 rounded-lg overflow-auto ${
            !expanded ? "max-h-[500px]" : ""
          }`}
          style={!expanded ? { maxHeight: `${maxHeight}px` } : undefined}
        >
          {content}
        </pre>
        {content.length > 2000 && (
          <Button
            variant="ghost"
            size="sm"
            className="absolute bottom-2 right-2"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? (
              <>
                <ChevronUp className="h-4 w-4 mr-1" />
                Collapse
              </>
            ) : (
              <>
                <ChevronDown className="h-4 w-4 mr-1" />
                Expand
              </>
            )}
          </Button>
        )}
      </div>
    </div>
  );
}
