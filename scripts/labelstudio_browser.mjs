import { chromium, expect } from "@playwright/test";
import { copyFile, writeFile } from "node:fs/promises";
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
  args: ["--no-sandbox"],
});
let page;
const mutations = [];
const errors = [];
const realDroid = process.env.LS_FIXTURE_REAL_DROID === "1";
const evidencePrefix = realDroid ? "droid" : "labelstudio";
// Real dataset images stay in ignored local evidence, never the Git history.
const screenshotPrefix = realDroid
  ? "artifacts/droid"
  : "docs/evidence/labelstudio";
const fps = Number(process.env.LS_FIXTURE_FPS || 12);
async function waitForVideo() {
  const video = page.locator("video");
  // A cold editor may mount after the default seven-second locator timeout.
  // Keep attachment separate so readiness polling cannot fail on that timeout.
  await video.waitFor({ state: "attached", timeout: 30000 });
  await expect
    .poll(
      () =>
        page.evaluate(() => document.querySelector("video")?.readyState ?? -1),
      { timeout: 20000 },
    )
    .toBe(4);
  return video;
}
async function review() {
  const context = await browser.newContext({
    viewport: { width: 1512, height: 1050 },
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
  page = await context.newPage();
  page.setDefaultTimeout(7000);
  if (process.env.LS_CPU_RATE) {
    const cdp = await context.newCDPSession(page);
    await cdp.send("Emulation.setCPUThrottlingRate", {
      rate: Number(process.env.LS_CPU_RATE),
    });
  }
  page.on("response", async (response) => {
    const request = response.request();
    const url = new URL(request.url());
    if (
      /^\/api\/tasks\/\d+\/?$/.test(url.pathname) &&
      request.method() === "GET"
    ) {
      const task = await response.json();
      console.log(
        "Native task response",
        JSON.stringify({
          path: url.pathname + url.search,
          id: task.id,
          predictions: (task.predictions || []).map((p) => ({
            id: p.id,
            model_version: p.model_version,
            result_count: p.result?.length,
          })),
        }),
      );
    }
    if (url.pathname === `/api/projects/${process.env.LS_FIXTURE_PROJECT}`) {
      const project = await response.json();
      console.log(
        "Native project configuration",
        JSON.stringify({
          id: project.id,
          model_version: project.model_version,
          show_collab_predictions: project.show_collab_predictions,
        }),
      );
    }
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
  const loadingContract = process.env.LS_TASK_LOADING_CONTRACT === "1";
  let detailsDelivered = 0;
  let orderedListDelivered = false;
  let releaseList;
  const detailsReady = new Promise((resolve) => {
    releaseList = resolve;
  });
  await page.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (url.hostname !== "127.0.0.1") return route.abort();
    // Reorder real responses only. No response payload or application state is modified.
    if (loadingContract && !orderedListDelivered) {
      if (/^\/api\/tasks\/\d+$/.test(url.pathname) && detailsDelivered < 2) {
        const response = await route.fetch();
        await route.fulfill({ response });
        detailsDelivered += 1;
        if (detailsDelivered === 2) releaseList();
        return;
      }
      if (url.pathname === "/api/tasks" && url.searchParams.has("page")) {
        const response = await route.fetch();
        let timer;
        try {
          await Promise.race([
            detailsReady,
            new Promise((_, reject) => {
              timer = setTimeout(
                () => reject(new Error("task detail delivery timeout")),
                7000,
              );
            }),
          ]);
          await route.fulfill({ response });
          orderedListDelivered = true;
        } finally {
          clearTimeout(timer);
        }
        return;
      }
    }
    return route.continue();
  });
  const projectURL = `http://127.0.0.1:8134/projects/${process.env.LS_FIXTURE_PROJECT}/data`;
  if (loadingContract) {
    await page.goto(`${projectURL}?task=${process.env.LS_FIXTURE_TASK}`);
    await waitForVideo();
    const cold = await page.evaluate(() => ({
      task_predictions: window.DM.taskStore.selected.predictions.length,
      editor_predictions: window.Htx.annotationStore.predictions.length,
      human_annotations: window.Htx.annotationStore.annotations.length,
    }));
    expect(cold).toEqual({
      task_predictions: 0,
      editor_predictions: 0,
      human_annotations: 1,
    });
    await writeFile(
      "artifacts/labelstudio-task-loading.json",
      JSON.stringify({
        upstream_version: "1.23.2",
        response_order: "two_task_details_then_list",
        cold,
      }),
    );
    console.log(
      "Reproduced upstream cold-task loading race",
      JSON.stringify(cold),
    );
    await page
      .getByRole("link", { name: "Robot episode review", exact: true })
      .click();
  } else {
    await page.goto(projectURL);
  }
  const taskRow = page.getByTestId("table-row").filter({
    has: page.getByRole("checkbox", {
      name: `Select Task ${process.env.LS_FIXTURE_TASK}`,
      exact: true,
    }),
  });
  await taskRow.waitFor({ state: "visible", timeout: 20000 });
  await taskRow
    .getByText(process.env.LS_FIXTURE_TITLE || "Transfer cube", { exact: true })
    .click();
  const video = await waitForVideo();
  const initialTask = await (
    await page.request.get(
      `http://127.0.0.1:8134/api/tasks/${process.env.LS_FIXTURE_TASK}/`,
    )
  ).json();
  console.log(
    "Native task readiness",
    JSON.stringify({
      id: initialTask.id,
      predictions: initialTask.predictions.map((p) => ({
        id: p.id,
        model_version: p.model_version,
        result_count: p.result.length,
      })),
      annotations: initialTask.annotations.length,
    }),
  );
  console.log(
    "Native UI readiness",
    JSON.stringify(
      await page.evaluate(() => ({
        flags: Object.fromEntries(
          Object.entries(window.APP_SETTINGS?.feature_flags || {}).filter(
            ([key]) =>
              /prediction|model_version|interactive_view_all|annotation/.test(
                key,
              ),
          ),
        ),
        taskStore: {
          loading: window.DM.taskStore.loading,
          selectedId: window.DM.taskStore.selectedId,
          predictions: window.DM.taskStore.selected?.predictions?.length,
        },
        predictionCount: window.Htx.annotationStore.predictions.length,
        annotationCount: window.Htx.annotationStore.annotations.length,
        annotationTabs: Array.from(
          document.querySelectorAll(".lsf-annotation-button"),
        ).map((el) => ({
          id: el.getAttribute("data-annotation-id"),
          predictionIcon: !!el.querySelector(
            ".lsf-annotation-button__userpic_prediction",
          ),
        })),
      })),
    ),
  );
  expect(initialTask.predictions).toHaveLength(1);
  await page
    .locator(".lsf-annotation-button")
    .filter({ has: page.locator(".lsf-annotation-button__userpic_prediction") })
    .click();
  await expect(
    page.getByPlaceholder("Evidence, corrections, and uncertainty"),
  ).toBeDisabled();
  await page
    .getByRole("button", { name: "Create an annotation", exact: true })
    .click();
  await expect(page.locator(".lsf-tree__node_type_timelineregion")).toHaveCount(
    Number(process.env.LS_FIXTURE_SEGMENTS),
  );
  await expect(
    page.getByPlaceholder("Evidence, corrections, and uncertainty"),
  ).toBeEnabled();
  await waitForVideo();
  await page.locator(".episode-title").scrollIntoViewIfNeeded();
  await page.screenshot({
    path: `${screenshotPrefix}-native-desktop.png`,
    fullPage: false,
  });
  await writeFile(
    "artifacts/labelstudio-initial-aria.txt",
    await page.locator("body").ariaSnapshot(),
  );
  await expect(
    page.getByText("Robot signals · seconds", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".htx-timeseries-channel")).toHaveCount(
    Number(process.env.LS_FIXTURE_CHANNELS || 2),
  );
  if (realDroid) {
    await expect(
      page.locator(".lsf-richtext__line").filter({
        hasText: "inferred nominal timing; no capture timestamps",
      }),
    ).toBeVisible();
  }
  expect(
    (await page.locator(".htx-timeseries-channel").allTextContents()).join(" "),
  ).not.toContain("1900");
  await page.getByTestId("playback-button:play").click();
  await expect
    .poll(() => video.evaluate((v) => v.currentTime))
    .toBeGreaterThan(1.2);
  await page.getByTestId("playback-button:pause").click();
  await page.waitForTimeout(150);
  const before = await video.evaluate((v) => v.currentTime);
  const chart = page
    .locator(".htx-timeseries-channel .new_brush .overlay")
    .first();
  await chart.click({ trial: true });
  const box = await chart.boundingBox();
  await chart.click({ position: { x: box.width * 0.7, y: box.height * 0.5 } });
  await expect
    .poll(async () =>
      Math.abs((await video.evaluate((v) => v.currentTime)) - before),
    )
    .toBeGreaterThan(1 / fps);
  const note = realDroid
    ? "Automated native GUI correction on real DROID; integration check only, labels unvalidated, timing inferred."
    : "Automated native Label Studio GUI correction; synthetic fixture, not human study evidence.";
  await page
    .getByRole("radio", { name: realDroid ? /^unknown\[/ : /^failure\[/ })
    .check();
  await page
    .getByRole("checkbox", {
      name: realDroid ? /^incomplete\[/ : /^recovery\[/,
    })
    .check();
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
  expect(
    persisted.annotations[0].result.find((r) => r.from_name === "decision_note")
      .value.text,
  ).toEqual([note]);
  const first = persisted.annotations[0].result.find(
    (r) => r.id === "segment-0",
  );
  expect(first.value.ranges).toEqual([{ start: 1, end: 6 }]);
  await page.reload();
  const savedTab = page
    .locator(
      `.lsf-annotation-button[data-annotation-id="${persisted.annotations[0].id}"]`,
    )
    .filter({
      hasNot: page.locator(".lsf-annotation-button__userpic_prediction"),
    });
  await savedTab.waitFor({ state: "visible", timeout: 30000 });
  await savedTab.click();
  await expect(savedTab).toHaveClass(/lsf-annotation-button_selected/);
  await waitForVideo();
  await expect(
    page.getByPlaceholder("Evidence, corrections, and uncertainty"),
  ).toBeEnabled();
  // TextArea with Enter stores submitted text as a result; its input is blank.
  await expect(
    page.locator("#label-studio-dm").getByText(note, { exact: true }),
  ).toBeVisible({ timeout: 20000 });
  await page.locator(".lsf-tree__node_type_timelineregion").first().click();
  await expect(
    page.getByRole("spinbutton", { name: "End frame", exact: true }),
  ).toHaveValue("6");
  await expect(page.getByTestId("bottombar-update-button")).toBeInViewport();
  await page.locator(".episode-title").scrollIntoViewIfNeeded();
  const design = await page.evaluate(() => {
    const body = getComputedStyle(document.body);
    const heading = getComputedStyle(document.querySelector(".episode-title"));
    return {
      background: body.backgroundColor,
      font: body.fontFamily,
      headingFont: heading.fontFamily,
      headingSize: heading.fontSize,
    };
  });
  expect(design.background).toBe("rgb(245, 244, 239)");
  expect(design.font).toContain("monospace");
  expect(design.headingFont).toContain("Georgia");
  expect(design.headingSize).toBe("30px");
  await expect(page.locator(".lsf-menu-header__trigger svg")).toBeVisible();
  await writeFile("artifacts/labelstudio-design.json", JSON.stringify(design));
  await page.screenshot({
    path: `${screenshotPrefix}-native.png`,
    fullPage: true,
  });
  expect(errors).toEqual([]);
  await context.close();
  await copyFile(
    await page.video().path(),
    `${screenshotPrefix}-walkthrough.webm`,
  );
  await writeFile(
    `artifacts/${evidencePrefix}-browser.json`,
    JSON.stringify({
      evidence_kind: realDroid
        ? "real_droid_automated_integration"
        : "synthetic_fixture",
      entry_path: "native_task_table",
      cold_task_loading_control: loadingContract,
      cpu_throttle_rate: Number(process.env.LS_CPU_RATE || 1),
      native_video_playback: true,
      sensor_seek_sync: true,
      native_annotation_update: true,
      native_temporal_edit: { start: 1, end: 6 },
      native_outcome: realDroid ? "unknown" : "failure",
      native_quality: realDroid ? ["incomplete"] : ["recovery"],
      mutation_trace: mutations,
      script_errors: errors,
    }),
  );
  console.log(
    "Native Label Studio playback, sensor seek, annotation update and browser checks passed",
  );
}
let deadline;
try {
  await Promise.race([
    review(),
    new Promise((_, reject) => {
      deadline = setTimeout(
        () => reject(new Error("Native review exceeded 120 seconds")),
        120000,
      );
    }),
  ]);
} catch (error) {
  console.error(
    "Native review failure",
    JSON.stringify({
      error: String(error),
      mutation_trace: mutations,
      script_errors: errors,
    }),
  );
  await writeFile(
    "artifacts/labelstudio-failure.json",
    JSON.stringify({
      error: String(error),
      mutation_trace: mutations,
      script_errors: errors,
    }),
  );
  if (page) {
    await page.screenshot({
      path: "artifacts/labelstudio-failure.png",
      fullPage: true,
    });
    await writeFile(
      "artifacts/labelstudio-failure-aria.txt",
      await page.locator("body").ariaSnapshot(),
    );
    console.error(
      "Native failure UI",
      await page.locator("body").ariaSnapshot(),
    );
  }
  throw error;
} finally {
  clearTimeout(deadline);
  await browser.close();
}
