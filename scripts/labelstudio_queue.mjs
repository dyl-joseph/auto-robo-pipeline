import { chromium, expect } from "@playwright/test";

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
  args: ["--no-sandbox"],
});
try {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
  });
  await context.addCookies(
    Object.entries(JSON.parse(process.env.LS_FIXTURE_COOKIES)).map(
      ([name, value]) => ({
        name,
        value,
        domain: "127.0.0.1",
        path: "/",
      }),
    ),
  );
  const page = await context.newPage();
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    return url.hostname === "127.0.0.1" ||
      ["data:", "blob:"].includes(url.protocol)
      ? route.continue()
      : route.abort();
  });
  const base = `http://127.0.0.1:8134/projects/${process.env.LS_FIXTURE_PROJECT}/data`;
  await page.goto(`${base}?tab=${process.env.LS_FIXTURE_VIEW}`, {
    waitUntil: "domcontentloaded",
  });
  await expect(page.getByText("Review priority", { exact: true })).toBeVisible({
    timeout: 30000,
  });
  await expect(
    page.getByText("review_priority", { exact: true }).first(),
  ).toBeVisible({ timeout: 15000 });
  await page.screenshot({ path: "artifacts/batch-queue.png", fullPage: true });
  await page.goto(`${base}?task=${process.env.LS_FIXTURE_TASK}`, {
    waitUntil: "domcontentloaded",
  });
  await page.locator("video").waitFor({ state: "attached", timeout: 30000 });
  await expect
    .poll(() => page.locator("video").evaluate((video) => video.readyState), {
      timeout: 20000,
    })
    .toBe(4);
  await page.screenshot({ path: "artifacts/batch-review.png", fullPage: true });
} finally {
  await browser.close();
}
