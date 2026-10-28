// Record automated edits on generated episodes.
import { chromium } from "@playwright/test";
import { spawn } from "node:child_process";
import { mkdir, copyFile } from "node:fs/promises";

process.env.PLAYWRIGHT_BROWSERS_PATH = ".cache/playwright";
const server = spawn(
  "uv",
  ["run", "--locked", "python", "-m", "scripts.browser_server"],
  { stdio: "ignore" },
);
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let browser;
try {
  for (let i = 0; i < 100; i++) {
    try {
      if ((await fetch("http://127.0.0.1:8123/api/queue")).ok) break;
    } catch {}
    await delay(200);
  }
  await mkdir("artifacts/walkthrough", { recursive: true });
  browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
    args: ["--no-sandbox"],
  });
  const context = await browser.newContext({
    viewport: { width: 1512, height: 1100 },
    recordVideo: {
      dir: "artifacts/walkthrough",
      size: { width: 1512, height: 1100 },
    },
  });
  const page = await context.newPage();
  const video = page.video();
  await page.goto("http://127.0.0.1:8123");
  await page.locator("#workspace").waitFor();
  const items = (
    await (await page.request.get("http://127.0.0.1:8123/api/queue")).json()
  ).items;
  let selected;
  for (const item of items) {
    const data = await (
      await page.request.get(`http://127.0.0.1:8123/api/episodes/${item.id}`)
    ).json();
    if (data.episode.source_id === "illustrated-tabletop-0") {
      selected = item;
      break;
    }
  }
  await page
    .locator(".queue-card")
    .filter({ hasText: selected.id.slice(0, 8) })
    .click();
  async function caption(value) {
    await page.locator("#breadcrumb").evaluate((node, text) => {
      node.textContent = text;
    }, value);
    await delay(1700);
  }
  await caption(
    "AUTOMATED WALKTHROUGH / SYNTHETIC DATA / INSPECT SOURCE + STREAMS",
  );
  await page.locator("#play").click();
  await delay(3500);
  await page.locator("#play").click();
  await caption(
    "01 / MOTION CANDIDATES ARE EVIDENCE — HUMAN LABELS STAY SEPARATE",
  );
  await page.locator("#load-suggestions").click();
  await delay(1300);
  await page
    .getByLabel("Label segment 1", { exact: true })
    .selectOption("reach");
  await page.getByLabel("end_s segment 1", { exact: true }).fill("2.5");
  await caption("02 / CORRECT THE LABEL AND TEMPORAL BOUNDARY");
  await page.locator("#actor").fill("walkthrough-automation");
  await page
    .locator("#rationale")
    .fill(
      "Automated demo: illustrative reach ends at 2.5s. This decision is not human-study evidence.",
    );
  await page.locator("#save").click();
  await page
    .locator("#message")
    .filter({ hasText: "Revision 1 draft" })
    .waitFor();
  await caption("03 / SAVED DRAFT — ACCEPTANCE IS A SEPARATE DECISION");
  await page
    .locator("#rationale")
    .fill("Automated walkthrough acceptance of synthetic fixture only.");
  await page.locator("#accept").click();
  await page.locator("#status").filter({ hasText: "accepted" }).waitFor();
  await caption("04 / IMMUTABLE REVISIONS KEEP CORRECTIONS AND REVIEW EFFORT");
  await page.locator("#video").evaluate((v) => {
    v.currentTime = 3.5;
  });
  await page.waitForFunction(
    () => document.getElementById("video").readyState >= 2,
  );
  await page.screenshot({
    path: "docs/evidence/synchronized-review.png",
    fullPage: true,
  });
  await page.locator("#export-open").click();
  await delay(1200);
  await page.locator("#export-confirm").click();
  await page
    .locator("#export-result")
    .filter({ hasText: "Validated version" })
    .waitFor();
  await delay(2500);
  await context.close();
  await video.saveAs("docs/evidence/review-walkthrough.webm");
} finally {
  if (browser) await browser.close();
  server.kill("SIGTERM");
}
