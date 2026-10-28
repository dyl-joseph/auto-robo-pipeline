import { existsSync } from "node:fs";
import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "tests/browser",
  workers: 1,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:8123",
    viewport: { width: 1512, height: 1100 },
    launchOptions: {
      executablePath:
        process.env.CHROMIUM_PATH ||
        (existsSync("/usr/bin/chromium") ? "/usr/bin/chromium" : undefined),
      args: ["--no-sandbox"],
    },
    video: "on",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "uv run --locked python -m scripts.browser_server",
    url: "http://127.0.0.1:8123/api/queue",
    reuseExistingServer: false,
    timeout: 60000,
  },
  outputDir: "artifacts/browser-tests",
});
