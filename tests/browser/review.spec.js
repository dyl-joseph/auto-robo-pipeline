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
  await page.locator("#video").evaluate((v) => (v.currentTime = 3.5));
  await expect(page.locator("#frame-label")).toContainText("PTS");
  await page.locator("#play").click();
  await page.waitForTimeout(500);
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
