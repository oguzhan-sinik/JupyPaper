"use client";

interface Props {
  error: string | null;
}

export default function SplashScreen({ error }: Props) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-8 animate-scale-in">
      {/* Spinning logo */}
      <div className="relative h-20 w-20">
        <div
          className="h-20 w-20 rounded-full animate-spin-ring"
          style={{
            border: "3px solid var(--color-surface-4)",
            borderTopColor: "var(--color-accent)",
          }}
        />
        <div className="absolute inset-0 flex items-center justify-center font-mono text-2xl font-bold"
             >
          J<span style={{ color: "var(--color-accent)" }}>P</span>
        </div>
      </div>

      {/* Title */}
      <h1 className="text-3xl font-semibold tracking-tight"
          style={{
            background: "linear-gradient(135deg, var(--color-foreground) 0%, var(--color-accent) 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>
        JupyPaper
      </h1>

      {error ? (
        <div className="rounded-md px-4 py-2 font-mono text-xs"
             style={{
               color: "var(--color-danger)",
               background: "rgba(248,113,113,0.08)",
               border: "1px solid rgba(248,113,113,0.2)",
             }}>
          {error}
        </div>
      ) : (
        <div className="flex gap-1.5">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="h-1.5 w-1.5 rounded-full animate-dot-pulse"
              style={{
                background: "var(--color-accent)",
                animationDelay: `${i * 0.2}s`,
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}