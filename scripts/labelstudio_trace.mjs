export function observeNativePage(page, { responses, mutations, errors }) {
  // Chromium can release response bodies on navigation. Diagnostics use only cached metadata.
  page.on("response", (response) => {
    const request = response.request();
    const path = new URL(request.url()).pathname;
    if (!path.startsWith("/api/")) return;
    const entry = { path, method: request.method(), status: response.status() };
    responses.push(entry);
    if (!["GET", "HEAD"].includes(entry.method)) mutations.push(entry);
  });
  page.on("pageerror", (error) => errors.push(error.message));
}
