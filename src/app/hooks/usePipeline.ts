"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import type { PipelineEvent, Session } from "../types";

interface UsePipelineOptions {
  session: Session;
  apiBase: string;
  onEvent: (event: PipelineEvent) => void;
  onComplete: (event: PipelineEvent) => void;
}

export function usePipeline({ session, apiBase, onEvent, onComplete }: UsePipelineOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const start = useCallback(() => {
    if (wsRef.current || isRunning) return;

    const wsUrl = apiBase.replace("http", "ws") + `/ws/${session.id}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsRunning(true);
      ws.send(JSON.stringify({ action: "start" }));
    };

    ws.onmessage = (msg) => {
      try {
        const event: PipelineEvent = JSON.parse(msg.data);
        onEvent(event);
        if (event.type === "complete" || event.type === "error") {
          setIsRunning(false);
          onComplete(event);
        }
      } catch {
        console.warn("Failed to parse event:", msg.data);
      }
    };

    ws.onclose = () => {
      setIsRunning(false);
      wsRef.current = null;
    };

    ws.onerror = () => {
      setIsRunning(false);
    };
  }, [session.id, apiBase, onEvent, onComplete, isRunning]);

  // NEW: Add a cancel function
  const cancel = useCallback(() => {
    if (wsRef.current) {
      // Send the cancel action if the socket is still open
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: "cancel" }));
      }
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsRunning(false);
  }, []);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  // NEW: Expose cancel
  return { start, cancel, isRunning };
}