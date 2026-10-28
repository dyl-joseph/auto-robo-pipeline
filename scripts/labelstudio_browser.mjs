import { chromium, expect } from "@playwright/test";
import { copyFile, writeFile } from "node:fs/promises";
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
  args: ["--no-sandbox"],
});
try {
  const context = await browser.newContext({
    viewport: { width: 1512, height: 1600 },
    recordVideo: {
      dir: "artifacts/labelstudio-video",
      size: { width: 1200, height: 1000 },
    },
  });
  await context.addCookies(
    Object.entries(JSON.parse(process.env.LS_FIXTURE_COOKIES)).map(
      ([name, value]) => ({ name, value, domain: "127.0.0.1", path: "/" }),
    ),
  );
  const page = await context.newPage();
  page.setDefaultTimeout(7000);
  const errors = [];
  const mutations = [];
  page.on("response", async (response) => {
    const request = response.request();
    if (
      request.url().includes("/api/") &&
      !["GET", "HEAD"].includes(request.method())
    )
      mutations.push({
        path: new URL(request.url()).pathname,
        method: request.method(),
        status: response.status(),
      });
  });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/*", (route) =>
    new URL(route.request().url()).hostname === "127.0.0.1"
      ? route.continue()
      : route.abort(),
  );
  await page.goto(
    `http://127.0.0.1:8134/projects/${process.env.LS_FIXTURE_PROJECT}/data?task=${process.env.LS_FIXTURE_TASK}`,
  );
  const video = page.locator("video");
  await video.waitFor({ state: "attached" });
  await expect
    .poll(() => video.evaluate((v) => v.readyState), { timeout: 20000 })
    .toBe(4);
  await expect(page.locator(".htx-timeseries-channel")).toHaveCount(2);
  await video.evaluate(async (v) => {
    v.currentTime = 1;
    await v.play();
  });
  await expect
    .poll(() => video.evaluate((v) => v.currentTime))
    .toBeGreaterThan(1.2);
  await video.evaluate((v) => v.pause());
  await page.waitForTimeout(150);
  const before = await video.evaluate((v) => v.currentTime);
  const chart = page.locator(".htx-timeseries-channel svg").first();
  await chart.scrollIntoViewIfNeeded();
  const box = await chart.boundingBox();
  await chart.click({ position: { x: box.width * 0.7, y: box.height * 0.5 } });
  await expect
    .poll(async () =>
      Math.abs((await video.evaluate((v) => v.currentTime)) - before),
    )
    .toBeGreaterThan(1 / 12);
  const note =
    "Automated native Label Studio GUI correction; synthetic fixture, not human study evidence.";
  await page
    .getByPlaceholder("Evidence, corrections, and uncertainty")
    .fill(note);
  await page
    .getByPlaceholder("Evidence, corrections, and uncertainty")
    .press("Enter");
  await page.locator(".lsf-tree__node_type_timelineregion").first().click();
  await page
    .getByRole("spinbutton", { name: "End frame", exact: true })
    .fill("6");
  await page
    .getByRole("spinbutton", { name: "End frame", exact: true })
    .press("Enter");
  const submit = page.getByRole("button", {
    name: "Submit current annotation",
    exact: true,
  });
  await expect(submit).toBeEnabled();
  await submit.click();
  const taskURL = `http://127.0.0.1:8134/api/tasks/${process.env.LS_FIXTURE_TASK}/`;
  await expect
    .poll(
      async () =>
        JSON.stringify(
          (await (await page.request.get(taskURL)).json()).annotations,
        ),
      { timeout: 5000 },
    )
    .toContain(note);
  const persisted = await (await page.request.get(taskURL)).json();
  const first = persisted.annotations[0].result.find(
    (r) => r.id === "segment-0",
  );
  expect(first.value.ranges).toEqual([{ start: 1, end: 6 }]);
  await page.reload();
  await expect(page.getByText(note, { exact: true })).toBeVisible();
  await page.locator(".lsf-tree__node_type_timelineregion").first().click();
  await expect(
    page.getByRole("spinbutton", { name: "End frame", exact: true }),
  ).toHaveValue("6");
  await page.locator(".episode-title").scrollIntoViewIfNeeded();
  await page.screenshot({
    path: "docs/evidence/labelstudio-native.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
  await context.close();
  await copyFile(
    await page.video().path(),
    "docs/evidence/labelstudio-walkthrough.webm",
  );
  await writeFile(
    "artifacts/labelstudio-browser.json",
    JSON.stringify({
      evidence_kind: "synthetic_fixture",
      native_video_playback: true,
      sensor_seek_sync: true,
      native_annotation_update: true,
      native_temporal_edit: { start: 1, end: 6 },
      mutation_trace: mutations,
      script_errors: errors,
    }),
  );
  console.log(
    "Native Label Studio playback, sensor seek, annotation update and browser checks passed",
  );
} finally {
  await browser.close();
}
