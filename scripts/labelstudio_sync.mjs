import { chromium, expect } from "@playwright/test";
import { writeFile } from "node:fs/promises";
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
  args: ["--no-sandbox"],
});
const evidence = [];
let currentPage;
async function checkSync() {
  const context = await browser.newContext({
    viewport: { width: 1512, height: 1050 },
  });
  await context.addCookies(
    Object.entries(JSON.parse(process.env.LS_FIXTURE_COOKIES)).map(
      ([name, value]) => ({ name, value, domain: "127.0.0.1", path: "/" }),
    ),
  );
  for (const test of JSON.parse(process.env.LS_SYNC_CASES)) {
    const page = await context.newPage();
    currentPage = page;
    page.setDefaultTimeout(5000);
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.route("**/*", (r) =>
      new URL(r.request().url()).hostname === "127.0.0.1"
        ? r.continue()
        : r.abort(),
    );
    const read = () =>
      page.evaluate(() => {
        const sensor = window.Htx.annotationStore.names.get("sensors");
        return {
          canvasWidth: sensor.canvasWidth,
          notReady: sensor.isNotReady,
          range: [...sensor.brushRange],
          keys: [...sensor.keysRange],
          cursor: sensor.cursorTime,
          videoTime: document.querySelector("video").currentTime,
          domWidth: document
            .querySelector(".htx-timeseries")
            ?.getBoundingClientRect().width,
        };
      });
    try {
      await page.goto(
        `http://127.0.0.1:8134/projects/${test.project}/data?task=${test.task}`,
      );
      await expect
        .poll(() => page.locator("video").evaluate((v) => v.readyState), {
          timeout: 20000,
        })
        .toBe(4);
      await page.locator(".htx-timeseries").waitFor({ state: "attached" });
      const beforeOpen = await read();
      if (test.name === "open-resized") {
        const panel = page.getByText("Signals", { exact: true });
        await panel.click();
        await expect(page.locator(".htx-timeseries")).toBeHidden();
        await page.setViewportSize({ width: 1400, height: 1050 });
        await page.waitForTimeout(250);
        await panel.click();
      }
      if (test.name === "collapsed")
        await page.getByText("Signals", { exact: true }).click();
      if (test.name === "themed") {
        await page.setViewportSize({ width: 1400, height: 1050 });
      }
      const before = await read();
      const plot = page
        .locator(".htx-timeseries-channel .new_brush .overlay")
        .first();
      await plot.scrollIntoViewIfNeeded();
      const box = await plot.boundingBox();
      await plot.click({
        position: { x: box.width * 0.7, y: box.height * 0.5 },
      });
      await page.waitForTimeout(250);
      const after = await read();
      evidence.push({ variant: test.name, beforeOpen, before, after, errors });
      if (["collapsed", "open-resized"].includes(test.name)) {
        expect(before.canvasWidth).toBe(0);
        expect(before.domWidth).toBeGreaterThan(0);
        expect(after.notReady).toBe(true);
        expect(after.videoTime).toBe(0);
      } else {
        expect(before.canvasWidth).toBeGreaterThan(0);
        expect(before.notReady).toBe(false);
        expect(after.videoTime).toBeCloseTo(5.54, 1);
        expect(errors).toEqual([]);
      }
      await page.screenshot({
        path: `artifacts/labelstudio-sync-${test.name}.png`,
        fullPage: true,
      });
    } catch (e) {
      evidence.push({ variant: test.name, error: String(e), errors });
      await page.screenshot({
        path: `artifacts/labelstudio-sync-${test.name}-failure.png`,
        fullPage: true,
      });
    }
    await page.close();
  }
  expect(evidence.filter((row) => row.error)).toEqual([]);
}
let deadline;
try {
  await Promise.race([
    checkSync(),
    new Promise((_, reject) => {
      deadline = setTimeout(
        () => reject(new Error("Sync contract exceeded 75 seconds")),
        75000,
      );
    }),
  ]);
} catch (error) {
  evidence.push({ error: String(error) });
  if (currentPage && !currentPage.isClosed())
    await currentPage.screenshot({
      path: "artifacts/labelstudio-sync-timeout.png",
      fullPage: true,
    });
  throw error;
} finally {
  clearTimeout(deadline);
  await writeFile(
    "artifacts/labelstudio-sync.json",
    JSON.stringify(evidence, null, 2),
  );
  console.log(JSON.stringify(evidence));
  await browser.close();
}
