import { expect, test } from "@playwright/test";
import { once } from "node:events";
import { writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { observeNativePage } from "../../scripts/labelstudio_trace.mjs";

let server;
let origin;

test.beforeAll(async () => {
  server = createServer((request, response) => {
    if (request.url.startsWith("/api/")) {
      response.writeHead(request.method === "POST" ? 201 : 200, {
        "Content-Type": "application/json",
      });
      response.end(request.url.endsWith("invalid") ? '{"id":' : '{"id":17}');
    } else {
      response.end("<!doctype html><title>Observer navigation fixture</title>");
    }
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  origin = `http://127.0.0.1:${server.address().port}`;
});

test.afterAll(async () => {
  server.closeAllConnections();
  await new Promise((resolve) => server.close(resolve));
});

test("response diagnostics survive navigation after an annotation POST", async ({
  page,
}, testInfo) => {
  const trace = { responses: [], mutations: [], errors: [] };
  observeNativePage(page, trace);
  await page.goto(origin);
  expect(
    await page.evaluate(async () => {
      const response = await fetch("/api/tasks/17/annotations", {
        method: "POST",
      });
      return { status: response.status, body: await response.json() };
    }),
  ).toEqual({ status: 201, body: { id: 17 } });

  const received = page.waitForResponse(`${origin}/api/tasks/17`);
  await page.evaluate(async () => (await fetch("/api/tasks/17")).status);
  const response = await received;
  await response.finished();
  await page.goto(`${origin}/next`);

  // Delaying the old observer's body read until navigation reproduces the CI failure.
  const unavailableBody = await response.json().catch((error) => String(error));
  expect(unavailableBody).toContain("Protocol error (Network.getResponseBody)");
  expect(unavailableBody).toContain("No resource with given identifier found");
  expect(trace.responses).toEqual([
    { path: "/api/tasks/17/annotations", method: "POST", status: 201 },
    { path: "/api/tasks/17", method: "GET", status: 200 },
  ]);
  expect(trace.mutations).toEqual([trace.responses[0]]);
  expect(trace.errors).toEqual([]);
  const evidencePath = testInfo.outputPath("navigation-response-lifetime.json");
  await writeFile(
    evidencePath,
    JSON.stringify({ unavailableBody, trace }, null, 2),
  );
  await testInfo.attach("navigation-response-lifetime", {
    path: evidencePath,
    contentType: "application/json",
  });
});

test("application errors and awaited malformed JSON remain failures", async ({
  page,
}) => {
  const trace = { responses: [], mutations: [], errors: [] };
  observeNativePage(page, trace);
  await page.goto(origin);
  const thrown = page.waitForEvent("pageerror");
  await page.evaluate(() => {
    setTimeout(() => {
      throw new Error("application error regression sentinel");
    }, 0);
  });
  await thrown;
  expect(trace.errors).toEqual(["application error regression sentinel"]);
  const response = await page.request.get(`${origin}/api/tasks/invalid`);
  await expect(response.json()).rejects.toThrow(/JSON/);
});
