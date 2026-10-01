import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // Standalone output so the container image carries only the runtime dependencies.
  output: "standalone",
};

export default config;