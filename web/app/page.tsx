import Link from "next/link";

/**
 * Landing page.
 *
 * Exists so the root URL is not a 404, which is what a user opening the deployed app
 * first sees. Both links go to real routes rather than being placeholders.
 */
export default function HomePage() {
  return (
    <main style={{ padding: "3rem 1.5rem", maxWidth: "42rem", margin: "0 auto" }}>
      <h2>Ask your corpus</h2>
      <p className="muted">
        Every answer is grounded in retrieved passages, and every citation links to the exact
        text it came from.
      </p>
      <p>
        <Link href="/chat">Start chatting</Link> or open the <Link href="/admin">admin console</Link>.
      </p>
    </main>
  );
}