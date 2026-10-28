import { chromium, expect } from "@playwright/test";
import { copyFile, writeFile } from "node:fs/promises";
import { observeNativePage } from "./labelstudio_trace.mjs";
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
  args: ["--no-sandbox"],
});
let page;
const mutations = [];
const responses = [];
const errors = [];
const controlErrors = [];
const realDroid = process.env.LS_FIXTURE_REAL_DROID === "1";
const evidencePrefix = realDroid ? "droid" : "labelstudio";
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
  observeNativePage(page, { responses, mutations, errors });
  const loadingContract = process.env.LS_TASK_LOADING_CONTRACT === "1";
  let coldOrder = null;
  let detailsDelivered = 0;
  let listDelivered = false;
  let bootstrapHeld = false;
  const state = () =>
    page.evaluate(() => ({
      task_predictions: window.DM?.taskStore.selected?.predictions.length ?? -1,
      editor_predictions: window.Htx?.annotationStore.predictions.length ?? -1,
      human_annotations: window.Htx?.annotationStore.annotations.length ?? -1,
    }));
  await page.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (url.hostname !== "127.0.0.1") return route.abort();
    try {
      // Reorder genuine responses without changing payloads or upstream stores.
      if (coldOrder && !listDelivered) {
        if (/^\/api\/tasks\/\d+$/.test(url.pathname) && detailsDelivered < 2) {
          const response = await route.fetch();
          await route.fulfill({ response });
          detailsDelivered += 1;
          return;
        }
        if (url.pathname === "/api/tasks" && url.searchParams.has("page")) {
          const response = await route.fetch();
          await expect.poll(() => detailsDelivered, { timeout: 20000 }).toBe(2);
          if (coldOrder === "editor_before_list") {
            await expect
              .poll(async () => (await state()).editor_predictions, {
                timeout: 20000,
              })
              .toBe(1);
          }
          await route.fulfill({ response });
          listDelivered = true;
          return;
        }
      }
      // This is the pinned 1.23.2 editor environment bootstrap (production.js).
      // Holding it keeps editor initialization after the list was *consumed*.
      if (
        coldOrder === "list_before_editor" &&
        url.pathname === "/react-app/99.js"
      ) {
        bootstrapHeld = true;
        await expect.poll(() => listDelivered, { timeout: 20000 }).toBe(true);
        await expect
          .poll(async () => (await state()).task_predictions, {
            timeout: 20000,
          })
          .toBe(0);
      }
      await route.continue();
    } catch (error) {
      controlErrors.push(String(error));
      await route.abort();
    }
  });
  const projectURL = `http://127.0.0.1:8134/projects/${process.env.LS_FIXTURE_PROJECT}/data`;
  async function openTaskRow(taskId) {
    const row = page.getByTestId("table-row").filter({
      has: page.getByRole("checkbox", {
        name: `Select Task ${taskId}`,
        exact: true,
      }),
    });
    await row.waitFor({ state: "visible", timeout: 20000 });
    await row
      .getByText(process.env.LS_FIXTURE_TITLE || "Transfer cube", {
        exact: true,
      })
      .click();
    await waitForVideo();
  }
  async function checkPrediction(taskId) {
    const task = await (
      await page.request.get(`http://127.0.0.1:8134/api/tasks/${taskId}/`)
    ).json();
    const editor = await page.evaluate(() =>
      window.Htx.annotationStore.predictions.map((prediction) => ({
        id: Number(prediction.pk),
        result: prediction.serializeAnnotation(),
      })),
    );
    expect(task.predictions).toHaveLength(1);
    const results = (items) =>
      items.map(({ id, result }) => ({
        id,
        result: result.map(({ id, from_name, to_name, type, value }) => ({
          id,
          from_name,
          to_name,
          type,
          value,
        })),
      }));
    expect(results(editor)).toEqual(results(task.predictions));
    return {
      task_id: taskId,
      prediction_ids: editor.map(({ id }) => id),
      contents_match: true,
    };
  }
  if (loadingContract) {
    const cases = [];
    const fixtures = JSON.parse(process.env.LS_LOADING_CASES);
    expect(fixtures).toHaveLength(2);
    for (const [index, order] of [
      "list_before_editor",
      "editor_before_list",
    ].entries()) {
      const fixture = fixtures[index];
      coldOrder = order;
      detailsDelivered = 0;
      listDelivered = false;
      bootstrapHeld = false;
      await page.goto(
        `http://127.0.0.1:8134/projects/${fixture.project_id}/data?task=${fixture.task_id}`,
      );
      await waitForVideo();
      await expect.poll(() => listDelivered, { timeout: 20000 }).toBe(true);
      await expect
        .poll(async () => (await state()).task_predictions, { timeout: 20000 })
        .toBe(0);
      const cold = await state();
      const expected = {
        task_predictions: 0,
        editor_predictions: order === "list_before_editor" ? 0 : 1,
        human_annotations: 1,
      };
      expect(cold).toEqual(expected);
      expect(bootstrapHeld).toBe(order === "list_before_editor");
      const samples = [cold];
      // Observe persistence after both responses and media loading have finished.
      for (let i = 0; i < 5; i += 1) {
        await page.waitForTimeout(400);
        const sample = await state();
        expect(sample).toEqual(expected);
        samples.push(sample);
      }
      expect(controlErrors).toEqual([]);
      const server = await (
        await page.request.get(
          `http://127.0.0.1:8134/api/tasks/${fixture.task_id}/`,
        )
      ).json();
      expect(server.predictions).toHaveLength(1);
      expect(server.annotations).toHaveLength(0);
      coldOrder = null;
      await page
        .getByRole("link", { name: "Robot episode review", exact: true })
        .click();
      await openTaskRow(fixture.task_id);
      expect(await state()).toEqual({
        task_predictions: 1,
        editor_predictions: 1,
        human_annotations: 1,
      });
      const recovered = await checkPrediction(fixture.task_id);
      cases.push({
        order,
        cold,
        persistence_ms: 2000,
        samples,
        server_annotations: 0,
        recovered,
      });
      console.log(
        "Reproduced upstream cold-task loading race",
        JSON.stringify(cases.at(-1)),
      );
    }
    await writeFile(
      "artifacts/labelstudio-task-loading.json",
      JSON.stringify({
        upstream_version: "1.23.2",
        response_order: "two_task_details_then_list",
        cases,
      }),
    );
  }
  await page.goto(projectURL);
  await openTaskRow(Number(process.env.LS_FIXTURE_TASK));
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
  await checkPrediction(Number(process.env.LS_FIXTURE_TASK));
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
  const predictionAfterReload = await checkPrediction(
    Number(process.env.LS_FIXTURE_TASK),
  );
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
  expect(controlErrors).toEqual([]);
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
      prediction_after_reload: predictionAfterReload,
      native_temporal_edit: { start: 1, end: 6 },
      native_outcome: realDroid ? "unknown" : "failure",
      native_quality: realDroid ? ["incomplete"] : ["recovery"],
      mutation_trace: mutations,
      response_trace: responses,
      script_errors: errors,
      loading_control_errors: controlErrors,
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
      response_trace: responses,
      script_errors: errors,
      loading_control_errors: controlErrors,
    }),
  );
  await writeFile(
    "artifacts/labelstudio-failure.json",
    JSON.stringify({
      error: String(error),
      mutation_trace: mutations,
      response_trace: responses,
      script_errors: errors,
      loading_control_errors: controlErrors,
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
