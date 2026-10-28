import { chromium, expect } from "@playwright/test";
import { writeFile } from "node:fs/promises";
import { observeNativePage } from "./labelstudio_trace.mjs";

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
  args: ["--no-sandbox"],
});
const context = await browser.newContext({
  viewport: { width: 1512, height: 1100 },
});
await context.addCookies(
  Object.entries(JSON.parse(process.env.LS_FIXTURE_COOKIES)).map(
    ([name, value]) => ({ name, value, domain: "127.0.0.1", path: "/" }),
  ),
);
const page = await context.newPage();
page.setDefaultTimeout(10000);
const trace = { responses: [], mutations: [], errors: [] };
const cpuRate = Number(process.env.LS_CPU_RATE || 1);
if (cpuRate > 1) {
  await (
    await context.newCDPSession(page)
  ).send("Emulation.setCPUThrottlingRate", { rate: cpuRate });
}
let holdReloadMedia = false;
let releaseReloadMedia;
const reloadMedia = new Promise((resolve) => {
  releaseReloadMedia = resolve;
});
let heldMediaRequests = 0;
observeNativePage(page, trace);
await page.route("**/*", async (route) => {
  const url = new URL(route.request().url());
  if (url.hostname !== "127.0.0.1") return route.abort();
  if (
    holdReloadMedia &&
    url.pathname === "/data/local-files/" &&
    url.searchParams.get("d")?.endsWith("video.mp4")
  ) {
    heldMediaRequests += 1;
    await reloadMedia;
  }
  return route.continue();
});
const taskId = Number(process.env.LS_FIXTURE_TASK);
const taskURL = `http://127.0.0.1:8134/api/tasks/${taskId}/`;
const corrected =
  "Automated test correction: initial state has no earlier sample.";
const object = "Automated fixture object; unvalidated";
const predictionContent = (task) =>
  task.predictions.map(
    ({ id, model_version, result, score, created_at, updated_at }) => ({
      id,
      model_version,
      result,
      score,
      created_at,
      updated_at,
    }),
  );
async function waitForVideo() {
  await page.locator("video").waitFor({ state: "attached", timeout: 30000 });
  await expect
    .poll(() => page.locator("video").evaluate((v) => v.readyState), {
      timeout: 20000,
    })
    .toBe(4);
}
const endField = page.getByRole("spinbutton", {
  name: "End frame",
  exact: true,
});
try {
  await page.goto(
    `http://127.0.0.1:8134/projects/${process.env.LS_FIXTURE_PROJECT}/data`,
  );
  const row = page.getByTestId("table-row").filter({
    has: page.getByRole("checkbox", {
      name: `Select Task ${taskId}`,
      exact: true,
    }),
  });
  await row.waitFor({ state: "visible", timeout: 30000 });
  await row.getByText(process.env.LS_FIXTURE_TITLE, { exact: true }).click();
  await waitForVideo();
  const initial = await (await page.request.get(taskURL)).json();
  await page
    .locator(".lsf-annotation-button")
    .filter({ has: page.locator(".lsf-annotation-button__userpic_prediction") })
    .click();
  await page
    .getByRole("button", { name: "Create an annotation", exact: true })
    .click();
  const regions = page.locator(".lsf-tree__node_type_timelineregion");
  await expect(regions).toHaveCount(Number(process.env.LS_FIXTURE_SEGMENTS));
  await regions.first().click();
  await page.locator('textarea[name^="subtask_text:"]').fill(corrected);
  await page.getByPlaceholder("Object, if observable").fill(object);
  await page.getByPlaceholder("Object, if observable").press("Enter");
  const end = endField;
  await expect(end).toHaveAttribute("max", process.env.LS_FIXTURE_FRAMES);
  await end.fill("2");
  await end.press("Enter");
  await expect(end).toHaveValue("2");
  await regions.nth(1).click();
  const start = page.getByRole("spinbutton", {
    name: "Start frame",
    exact: true,
  });
  await start.fill("3");
  await start.press("Enter");
  await expect(start).toHaveValue("3");
  await page
    .getByPlaceholder("Evidence, corrections, and uncertainty")
    .fill(
      "Automated semantic review on real DROID; labels unvalidated, timing inferred. Integration check only.",
    );
  await page
    .getByPlaceholder("Evidence, corrections, and uncertainty")
    .press("Enter");
  await page
    .getByRole("button", { name: "Submit current annotation", exact: true })
    .click();
  await expect
    .poll(
      async () =>
        JSON.stringify(
          (await (await page.request.get(taskURL)).json()).annotations,
        ),
      { timeout: 10000 },
    )
    .toContain(corrected);
  const saved = await (await page.request.get(taskURL)).json();
  expect(predictionContent(saved)).toEqual(predictionContent(initial));
  expect(saved.annotations).toHaveLength(1);
  const savedRanges = saved.annotations[0].result
    .filter((r) => r.type === "timelinelabels")
    .map((r) => ({ id: r.id, ranges: r.value.ranges }));
  console.log("Saved semantic ranges", JSON.stringify(savedRanges));
  expect(savedRanges.find((r) => r.id === "segment-0").ranges).toEqual([
    { start: 1, end: 2 },
  ]);
  expect(savedRanges.find((r) => r.id === "segment-1").ranges[0].start).toBe(3);
  holdReloadMedia = true;
  await page.reload({ waitUntil: "domcontentloaded" });
  const tab = page
    .locator(
      `.lsf-annotation-button[data-annotation-id="${saved.annotations[0].id}"]`,
    )
    .filter({
      hasNot: page.locator(".lsf-annotation-button__userpic_prediction"),
    });
  await tab.waitFor({ state: "visible", timeout: 30000 });
  await expect
    .poll(() => heldMediaRequests, { timeout: 30000 })
    .toBeGreaterThan(0);
  const unloadedVideoStates = await page
    .locator("video")
    .evaluateAll((elements) => elements.map((element) => element.readyState));
  expect([[], [0]]).toContainEqual(unloadedVideoStates);
  releaseReloadMedia();
  await waitForVideo();
  await tab.click();
  await expect(tab).toHaveClass(/lsf-annotation-button_selected/);
  await expect(regions).toHaveCount(Number(process.env.LS_FIXTURE_SEGMENTS));
  await regions.first().click();
  await expect(endField).toHaveAttribute("max", process.env.LS_FIXTURE_FRAMES);
  await expect(page.locator('textarea[name^="subtask_text:"]')).toHaveValue(
    corrected,
  );
  await expect(page.locator('input[name^="object_text:"]')).toHaveValue(object);
  await expect(
    page.getByRole("spinbutton", { name: "End frame", exact: true }),
  ).toHaveValue("2");
  const reloaded = await (await page.request.get(taskURL)).json();
  expect(predictionContent(reloaded)).toEqual(predictionContent(initial));
  expect(reloaded.annotations[0].result).toEqual(saved.annotations[0].result);
  await page.locator(".episode-title").scrollIntoViewIfNeeded();
  await page.screenshot({
    path: "artifacts/semantic-droid-native.png",
    fullPage: true,
  });
  expect(trace.errors).toEqual([]);
  await writeFile(
    "artifacts/semantic-native-browser.json",
    JSON.stringify(
      {
        evidence_kind: "real_droid_automated_integration",
        saved_annotation: saved.annotations[0].id,
        predictions_unchanged: true,
        saved_ranges_verified: true,
        held_reload_media_requests: heldMediaRequests,
        unloaded_video_states: unloadedVideoStates,
        trace,
      },
      null,
      2,
    ),
  );
} catch (error) {
  await writeFile(
    "artifacts/semantic-native-failure-aria.txt",
    await page.locator("body").ariaSnapshot(),
  );
  await page.screenshot({
    path: "artifacts/semantic-native-failure.png",
    fullPage: true,
  });
  console.error(JSON.stringify(trace));
  throw error;
} finally {
  releaseReloadMedia();
  await browser.close();
}
