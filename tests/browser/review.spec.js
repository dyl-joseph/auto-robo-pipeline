import { test, expect } from "@playwright/test";

test("review, correction, stale tab, audited acceptance and export", async ({
  page,
  context,
}) => {
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page.locator("#workspace")).toBeVisible();
  await expect(page.locator("#data-kind")).toHaveText("Synthetic fixture");
  await page.waitForFunction(() => {
    const video = document.getElementById("video");
    if (video.error)
      throw new Error(`Media decode failed: ${video.error.message}`);
    return video.readyState >= 2;
  });
  await page.locator("#video").evaluate((v) => (v.currentTime = 3.5));
  await expect(page.locator("#frame-label")).toContainText("PTS");
  await page.locator("#play").click();
  await expect
    .poll(() => page.locator("#video").evaluate((v) => v.currentTime))
    .toBeGreaterThan(3.7);
  await page.locator("#play").click();
  await page.locator("#load-suggestions").click();
  await expect(page.locator(".segment-row").first()).toBeVisible();

  await page
    .getByLabel("Label segment 1", { exact: true })
    .selectOption("reach");
  await page
    .locator("#rationale")
    .fill(
      "Browser automation: corrected heuristic motion to illustrative reach; not human study data.",
    );
  await page.locator("#save").click();
  await expect(page.locator("#message")).toContainText("Revision 1 draft");
  expect(
    await page.locator("#video").evaluate((v) => v.currentTime),
  ).toBeGreaterThan(3.5);
  const stale = await context.newPage();
  await stale.goto("/");
  await expect(stale.locator("#revision-label")).toHaveText("r1");
  await page
    .locator("#rationale")
    .fill(
      "Browser automation: inspected synthetic fixture and accepted revision.",
    );
  await page.locator("#accept").click();
  await expect(page.locator("#status")).toHaveText("accepted");
  await stale.locator("#rationale").fill("stale save attempt");
  await stale.locator("#save").click();
  await expect(stale.locator("#message")).toContainText("Stale revision");
  await stale.close();
  await page.locator("#video").evaluate((v) => (v.currentTime = 3.5));
  await page.waitForFunction(
    () => document.getElementById("video").readyState >= 2,
  );
  await page.screenshot({
    path: "docs/evidence/review-workspace.png",
    fullPage: true,
  });
  await page.locator("#export-open").click();
  await page.locator("#export-confirm").click();
  await expect(page.locator("#export-result")).toContainText(
    "Validated version",
  );
  const manifestURL = await page
    .getByRole("link", { name: "Download manifest.json" })
    .getAttribute("href");
  const manifest = await (await page.request.get(manifestURL)).json();
  expect(manifest.episode_count).toBe(1);
  await page.screenshot({ path: "docs/evidence/dataset-export.png" });
  await page.locator("#export-close").click();
  await page
    .getByLabel("Label segment 1", { exact: true })
    .selectOption("unknown");
  await page
    .locator("#rationale")
    .fill("Automated re-review invalidates prior acceptance.");
  await page.locator("#save").click();
  await expect(page.locator("#status")).toHaveText("draft");
  await page.reload();
  await expect(page.getByLabel("Label segment 1", { exact: true })).toHaveValue(
    "unknown",
  );
  expect(errors).toEqual([]);
});

test("mobile editor and manual workflow", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.locator("#task")).toBeVisible();
  page.on("dialog", (d) => d.accept());
  await page.locator("#manual-mode").check();
  await expect(page.locator("#load-suggestions")).toBeDisabled();
  await page.locator("#add-segment").click();
  await expect(page.locator(".segment-row")).toHaveCount(1);
  await page.screenshot({
    path: "docs/evidence/mobile-review.png",
    fullPage: true,
  });
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  );
  expect(overflow).toBe(false);
});

test("out-of-order loads and saving transitions preserve the latest intent", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator("#workspace")).toBeVisible();
  const items = (await (await page.request.get("/api/queue")).json()).items;
  const a = items[1],
    b = items[2];
  let release;
  const held = new Promise((resolve) => {
    release = resolve;
  });
  await page.route(`**/api/episodes/${a.id}`, async (route) => {
    const response = await route.fetch();
    await held;
    await route.fulfill({ response });
  });
  await page
    .locator(".queue-card")
    .filter({ hasText: a.id.slice(0, 8) })
    .click();
  await page
    .locator(".queue-card")
    .filter({ hasText: b.id.slice(0, 8) })
    .click();
  await expect(page.locator("#episode-id")).toHaveText(b.id.slice(0, 12));
  await Promise.all([
    page.waitForResponse((r) => r.url().endsWith(`/api/episodes/${a.id}`)),
    Promise.resolve().then(() => release()),
  ]);
  await expect(page.locator("#episode-id")).toHaveText(b.id.slice(0, 12));
  await page.unroute(`**/api/episodes/${a.id}`);
  let releaseEdited;
  const editedLoad = new Promise((resolve) => {
    releaseEdited = resolve;
  });
  await page.route(`**/api/episodes/${a.id}`, async (route) => {
    const response = await route.fetch();
    await editedLoad;
    await route.fulfill({ response });
  });
  await page
    .locator(".queue-card")
    .filter({ hasText: a.id.slice(0, 8) })
    .click();
  await page
    .locator("#rationale")
    .fill("Preserve this note while another episode loads");
  await Promise.all([
    page.waitForResponse((r) => r.url().endsWith(`/api/episodes/${a.id}`)),
    Promise.resolve().then(() => releaseEdited()),
  ]);
  await expect(page.locator("#rationale")).toHaveValue(
    "Preserve this note while another episode loads",
  );
  await expect(page.locator("#episode-id")).toHaveText(b.id.slice(0, 12));
  await page.locator("#add-segment").click();
  await page.locator("#rationale").fill("Automation: delayed save regression");
  let finish;
  const pending = new Promise((resolve) => {
    finish = resolve;
  });
  let calls = 0;
  await page.route(`**/api/reviews/${b.id}/save`, async (route) => {
    calls++;
    await pending;
    await route.continue();
  });
  await page.locator("#save").click();
  await expect(page.locator("#save")).toBeDisabled();
  await expect(page.locator(".queue-card").first()).toBeDisabled();
  await expect(
    page.getByLabel("Label segment 1", { exact: true }),
  ).toBeDisabled();
  await page.locator("#save").dispatchEvent("click");
  await page.locator(".queue-card").first().dispatchEvent("click");
  expect(calls).toBe(1);
  expect(await page.locator("#episode-id").textContent()).toBe(
    b.id.slice(0, 12),
  );
  finish();
  await expect(page.locator("#save")).toBeEnabled();
  await expect(page.locator("#message")).toContainText("draft");
});

test("visible playback counts after 30s idle, with pause and visibility exclusions", async ({
  page,
}) => {
  await page.clock.install();
  await page.goto("/");
  await expect(page.locator("#workspace")).toBeVisible();
  await page.locator("#video").evaluate(async (v) => {
    v.loop = true;
    await v.play();
  });
  await page.clock.runFor(31000);
  const active = () =>
    page
      .locator("#active-time")
      .evaluate((n) => Number.parseFloat(n.textContent));
  expect(await active()).toBeGreaterThanOrEqual(30);
  await page.locator("#pause-timer").click();
  const paused = await active();
  await page.clock.runFor(5000);
  expect(await active()).toBe(paused);
  await page.locator("#pause-timer").click();
  await page.evaluate(() =>
    Object.defineProperty(document, "hidden", {
      configurable: true,
      get: () => true,
    }),
  );
  const hidden = await active();
  await page.clock.runFor(5000);
  expect(await active()).toBe(hidden);
  await page.evaluate(() =>
    Object.defineProperty(document, "hidden", {
      configurable: true,
      get: () => false,
    }),
  );
  await page.clock.runFor(1000);
  expect(await active()).toBeGreaterThan(hidden);
  await page.locator("#video").evaluate((v) => v.pause());
  await page.clock.runFor(31000);
  const idle = await active();
  await page.clock.runFor(5000);
  expect(await active()).toBe(idle);
});

test("saved suggestion origin and restored condition survive a newer active run", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator("#workspace")).toBeVisible();
  const items = (await (await page.request.get("/api/queue")).json()).items;
  const item = items.find((i) => i.status === "unreviewed");
  await page
    .locator(".queue-card")
    .filter({ hasText: item.id.slice(0, 8) })
    .click();
  await expect(page.locator("#episode-id")).toHaveText(item.id.slice(0, 12));
  const original = await (
    await page.request.get(`/api/episodes/${item.id}`)
  ).json();
  await page.locator("#load-suggestions").click();
  await page.locator("#rationale").fill("Automation: provenance regression");
  await page.locator("#save").click();
  await expect(page.locator("#message")).toContainText("draft");
  await page.route(`**/api/episodes/${item.id}`, async (route) => {
    const response = await route.fetch();
    const data = await response.json();
    data.analysis.id = "f".repeat(24);
    await route.fulfill({ json: data });
  });
  await page.reload();
  await expect(page.locator("#revision-label")).toHaveText("r1");
  page.on("dialog", (d) => d.accept());
  await page.locator("#manual-mode").check();
  await page.getByRole("button", { name: "Restore as draft" }).first().click();
  await expect(page.locator("#manual-mode")).not.toBeChecked();
  await page.locator("#save").click();
  await expect(page.locator("#message")).toContainText("Revision 2 draft");
  const saved = await (
    await page.request.get(`/api/episodes/${item.id}`)
  ).json();
  expect(saved.review.suggestion_run).toBe(original.analysis.id);
  expect(saved.review.condition).toBe("assisted");
});
