"use client";

import { useEffect, useState } from "react";
import { AUTH_CHANGED_EVENT, clearToken, readToken } from "@/lib/auth";

/** Sign-out control. Hidden when no session token is stored. */
export default function SessionControls() {
  const [signedIn, setSignedIn] = useState(false);

  useEffect(() => {
    const sync = () => setSignedIn(Boolean(readToken()));
    sync();
    window.addEventListener(AUTH_CHANGED_EVENT, sync);
    return () => window.removeEventListener(AUTH_CHANGED_EVENT, sync);
  }, []);

  if (!signedIn) return null;

  return (
    <button type="button" className="ghost" onClick={() => clearToken()}>
      Sign out
    </button>
  );
}
