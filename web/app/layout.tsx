import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Hackstack",
  description: "Permission-aware chat over your team's knowledge, with live sources.",
};

// System fonts only: no font download at build time, so the UI builds and runs offline.
export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full">
        {children}
      </body>
    </html>
  );
}
