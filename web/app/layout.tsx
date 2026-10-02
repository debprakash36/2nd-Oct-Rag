import type { Metadata } from "next";
import Link from "next/link";
import AuthGate from "@/components/AuthGate";
import SessionControls from "@/components/SessionControls";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Chatbot",
  description: "Ask questions about the indexed corpus, with citations you can check.",
};

/**
 * Root layout.
 *
 * The nav is rendered here rather than per page so every route can reach the chat or
 * the admin console without duplicating the header, and so there is exactly one place
 * that decides what the product is called.
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="app-shell">
          <header className="app-header">
            <h1>RAG Chatbot</h1>
            <nav className="app-nav">
              <Link href="/chat">Chat</Link>
              <Link href="/admin">Admin</Link>
              <Link href="/admin/pilot">Pilot</Link>
              <SessionControls />
            </nav>
          </header>
          <AuthGate>{children}</AuthGate>
        </div>
      </body>
    </html>
  );
}