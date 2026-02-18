import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "JupyPaper",
  description: "Transform research papers into executable Jupyter notebooks — powered by NVIDIA NIM",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}