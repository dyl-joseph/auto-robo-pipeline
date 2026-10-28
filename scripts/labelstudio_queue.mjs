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
  const taskRow = page.getByTestId("table-row").first();
  await expect(
    taskRow.getByRole("checkbox", {
      name: `Select Task ${process.env.LS_FIXTURE_TASK}`,
      exact: true,
    }),
  ).toBeVisible({ timeout: 20000 });
  await page.screenshot({ path: "artifacts/batch-queue.png", fullPage: true });
  await taskRow
    .getByText(process.env.LS_FIXTURE_TITLE, { exact: true })
    .click();
  await page.locator("video").waitFor({ state: "attached", timeout: 30000 });
  await expect
    .poll(() => page.locator("video").evaluate((video) => video.readyState), {
      timeout: 20000,
    })
    .toBe(4);
  const prediction = page.locator(".lsf-annotation-button").filter({
    has: page.locator(".lsf-annotation-button__userpic_prediction"),
  });
  await expect(prediction).toHaveCount(1);
  await prediction.click();
  await page.getByTestId("playback-button:play").click({ timeout: 20000 });
  await expect
    .poll(() => page.locator("video").evaluate((video) => video.currentTime), {
      timeout: 10000,
    })
    .toBeGreaterThan(0.5);
  await page.getByTestId("playback-button:pause").click();
  await expect(page.locator(".episode-media canvas").first()).toBeVisible();
  await page.screenshot({ path: "artifacts/batch-review.png", fullPage: true });
} finally {
  await browser.close();
}
