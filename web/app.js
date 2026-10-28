const $ = (id) => document.getElementById(id);
const state = {
  items: [],
  labels: [],
  current: null,
  segments: [],
  selected: -1,
  dirty: false,
  filter: "all",
  condition: "assisted",
  active: 0,
  wall: 0,
  last: performance.now(),
  input: performance.now(),
  paused: false,
  run: null,
  loadToken: 0,
  editVersion: 0,
  saving: false,
  suggestionOrigin: null,
  perceptionOrigin: null,
};
const colors = ["#759850", "#74929a", "#b59d61", "#a78999", "#c78c62"];
const text = (tag, value, cls) => {
  const node = document.createElement(tag);
  node.textContent = value;
  if (cls) node.className = cls;
  return node;
};
async function api(url, body) {
  const response = await fetch(
    url,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const value = await response.json();
  if (!response.ok) throw new Error(value.detail || "Request failed");
  return value;
}
function message(value, error = false) {
  $("message").textContent = value;
  $("message").classList.toggle("error", error);
}
function dirty() {
  state.editVersion++;
  state.dirty = true;
  $("save-state").textContent = "Unsaved changes";
}
function fmt(t) {
  return `${Math.floor(t / 60)
    .toString()
    .padStart(2, "0")}:${(t % 60).toFixed(2).padStart(5, "0")}`;
}
async function refreshQueue() {
  const data = await api("/api/queue");
  state.items = data.items;
  state.labels = data.labels;
  drawQueue();
  drawCoverage();
}
function drawQueue() {
  const q = $("search").value.toLowerCase();
  $("queue").replaceChildren();
  $("queue-count").textContent = state.items.length;
  for (const [index, item] of state.items.entries()) {
    if (
      (state.filter === "accepted" && item.status !== "accepted") ||
      (state.filter === "pending" &&
        ["accepted", "excluded"].includes(item.status)) ||
      !`${item.task} ${item.id}`.toLowerCase().includes(q)
    )
      continue;
    const button = text(
      "button",
      "",
      `queue-card ${state.current?.episode.id === item.id ? "selected" : ""}`,
    );
    button.disabled = state.saving;
    const title = text("div", "", "queue-title");
    title.append(
      text("span", `${String(index + 1).padStart(2, "0")}  ${item.task}`),
      text("span", item.status === "accepted" ? "✓" : "↗", "queue-status"),
    );
    button.append(
      title,
      text(
        "span",
        `${item.id.slice(0, 8)} · ${item.duration_s.toFixed(1)}s · ${item.data_kind}`,
        "tiny",
      ),
      text("span", item.status.replaceAll("_", " "), "tiny"),
      text("span", item.reasons[0].replaceAll("_", " "), "tiny reason"),
    );
    button.onclick = () =>
      openEpisode(item.id).catch((e) => message(e.message, true));
    $("queue").append(button);
  }
}
function drawCoverage() {
  $("coverage").replaceChildren();
  const tasks = [...new Set(state.items.map((i) => i.task))];
  for (const task of tasks) {
    const total = state.items.filter((i) => i.task === task);
    const accepted = total.filter((i) => i.status === "accepted").length;
    const row = text("div", "", "coverage-row");
    row.append(
      text("span", task),
      text("span", `${accepted} / ${total.length}`),
    );
    const bar = text("div", "", "coverage-bar");
    const fill = text("i", "");
    fill.style.width = `${(100 * accepted) / total.length}%`;
    bar.append(fill);
    $("coverage").append(row, bar);
  }
}
async function openEpisode(id, force = false) {
  if (state.saving && !force) return;
  const token = ++state.loadToken;
  const editVersion = state.editVersion;
  if (
    state.dirty &&
    !force &&
    !confirm("Leave this episode and discard unsaved edits?")
  )
    return;
  let data;
  try {
    data = await api(`/api/episodes/${id}`);
  } catch (error) {
    if (token === state.loadToken) throw error;
    else return;
  }
  if (token !== state.loadToken) return;
  if (state.editVersion !== editVersion) {
    message(
      "New edits preserved. Save or discard them before switching episodes.",
    );
    return;
  }
  history.replaceState(null, "", `#${id}`);
  state.current = data;
  state.suggestionOrigin = data.review.suggestion_run || null;
  state.perceptionOrigin =
    data.review.revision > 0
      ? data.review.perception_run || null
      : data.perception?.id || null;
  state.run = data.analysis;
  state.segments = structuredClone(data.review.segments);
  state.selected = -1;
  state.dirty = false;
  state.active = 0;
  state.wall = 0;
  state.last = performance.now();
  state.input = performance.now();
  $("workspace").hidden = false;
  $("empty").hidden = true;
  $("task").textContent = data.episode.task;
  $("breadcrumb").textContent =
    `COLLECTION / ${data.episode.release} / EPISODE`;
  $("data-kind").textContent =
    data.episode.data_kind === "synthetic"
      ? "Synthetic fixture"
      : "Real recording";
  $("episode-id").textContent = data.episode.id.slice(0, 12);
  $("duration").textContent =
    `${data.episode.duration_s.toFixed(2)}s · ${data.episode.timestamps_s.length} measured PTS`;
  $("status").textContent = data.review.status.replaceAll("_", " ");
  $("save-state").textContent = `Saved revision ${data.review.revision}`;
  $("revision-label").textContent = `r${data.review.revision}`;
  if ($("video").dataset.episodeId !== id) {
    $("video").src = `/api/media/${id}`;
    $("video").dataset.episodeId = id;
  }
  $("seek").max = data.episode.duration_s;
  $("axis-mid").textContent = `${(data.episode.duration_s / 2).toFixed(1)}s`;
  $("axis-end").textContent = `${data.episode.duration_s.toFixed(1)}s`;
  $("outcome").value = data.review.outcome || "unknown";
  $("rationale").value = "";
  state.condition =
    data.review.condition || ($("manual-mode").checked ? "manual" : "assisted");
  $("manual-mode").checked = state.condition === "manual";
  $("flags").replaceChildren();
  for (const name of [
    "recovery",
    "occluded",
    "corrupt",
    "uncertain",
    "incomplete",
  ]) {
    const label = text("label", "");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = name;
    input.checked = (data.review.quality_flags || []).includes(name);
    input.onchange = dirty;
    label.append(input, document.createTextNode(name));
    $("flags").append(label);
  }
  drawSegments();
  drawSignals();
  drawHistory();
  drawQueue();
  drawStreams();
  message("");
  updateClock();
}
function drawSegments() {
  $("segments").replaceChildren();
  $("segment-count").textContent = `${state.segments.length} segments`;
  state.segments.forEach((segment, index) => {
    const row = text(
      "div",
      "",
      `segment-row ${index === state.selected ? "active" : ""}`,
    );
    const head = text("div", "", "segment-head");
    head.append(text("span", `SEGMENT ${String(index + 1).padStart(2, "0")}`));
    const del = text("button", "×");
    del.setAttribute("aria-label", `Delete segment ${index + 1}`);
    del.onclick = () => {
      state.segments.splice(index, 1);
      dirty();
      drawSegments();
    };
    head.append(del);
    row.append(head);
    const select = document.createElement("select");
    select.setAttribute("aria-label", `Label segment ${index + 1}`);
    for (const label of state.labels) {
      const option = text("option", label);
      option.value = label;
      select.append(option);
    }
    select.value = segment.label;
    select.onchange = () => {
      segment.label = select.value;
      dirty();
      drawTracks();
    };
    row.append(select);
    const bounds = text("div", "", "boundaries");
    for (const key of ["start_s", "end_s"]) {
      const label = text("label", key === "start_s" ? "START (s)" : "END (s)");
      const input = document.createElement("input");
      input.type = "number";
      input.min = 0;
      input.max = state.current.episode.duration_s;
      input.step = "0.01";
      input.value = segment[key].toFixed(3);
      input.setAttribute("aria-label", `${key} segment ${index + 1}`);
      input.onchange = () => {
        segment[key] = Number(input.value);
        state.selected = index;
        dirty();
        drawTracks();
      };
      input.onfocus = () => {
        state.selected = index;
      };
      label.append(input);
      bounds.append(label);
    }
    row.append(bounds);
    $("segments").append(row);
  });
  drawTracks();
}
function drawTracks() {
  for (const [id, segments, machine] of [
    ["human-track", state.segments, false],
    [
      "machine-track",
      state.condition === "manual" ? [] : state.run?.segments || [],
      true,
    ],
  ]) {
    $(id).replaceChildren();
    for (const [index, s] of segments.entries()) {
      const button = text(
        "button",
        s.label,
        `segment ${!machine && index === state.selected ? "selected" : ""}`,
      );
      button.style.left = `${(100 * s.start_s) / state.current.episode.duration_s}%`;
      button.style.width = `${(100 * (s.end_s - s.start_s)) / state.current.episode.duration_s}%`;
      button.title = `${s.label} [${s.start_s.toFixed(2)}, ${s.end_s.toFixed(2)})${machine ? ` · heuristic ${s.score.toFixed(2)}` : ""}`;
      button.onclick = () => {
        if (!machine) {
          state.selected = index;
          drawSegments();
        }
        $("video").currentTime = s.start_s;
      };
      $(id).append(button);
    }
  }
  $("load-suggestions").disabled = state.condition === "manual" || !state.run;
}
function drawSignals() {
  $("signals").replaceChildren();
  const warnings =
    state.run?.warnings || state.current.episode.validation.warnings;
  if (!warnings.length)
    $("signals").append(
      text(
        "div",
        "No import warnings. Human inspection still required.",
        "signal",
      ),
    );
  for (const warning of warnings) {
    const card = text("div", warning.replaceAll("_", " "), "signal");
    card.append(
      text(
        "small",
        warning.startsWith("missing")
          ? "Unavailable modality is explicit in the export."
          : "Inspect the source evidence before deciding.",
      ),
    );
    $("signals").append(card);
  }
  const note = text("div", "OpenCV motion baseline", "signal");
  note.append(
    text(
      "small",
      "Motion boxes + sparse optical flow. No object identity, physical contact, or semantic action inference. Scores are uncalibrated.",
    ),
  );
  $("signals").append(note);
  const perception = state.current.perception;
  if (perception && state.condition !== "manual") {
    const card = text(
      "div",
      "Pretrained MobileNet-SSD · VOC objects",
      "signal",
    );
    const detections = perception.samples.filter((s) => s.detections.length);
    card.append(
      text(
        "small",
        `${perception.sampled_frames} sampled frames · ${detections.length} with detections. Object candidates only; no contact or action recognition.`,
      ),
    );
    for (const sample of detections.slice(0, 8)) {
      const button = text(
        "button",
        `${sample.time_s.toFixed(1)}s: ${sample.detections.map((d) => `${d.label} ${d.score.toFixed(2)}`).join(", ")}`,
        "text-button",
      );
      button.style.display = "block";
      button.onclick = () => {
        $("video").currentTime = sample.time_s;
      };
      card.append(button);
    }
    $("signals").append(card);
  }
}
function drawHistory() {
  $("history").replaceChildren();
  const history = state.current.history;
  $("history-count").textContent = history.length;
  if (!history.length)
    $("history").append(text("p", "No review decisions yet.", "tiny"));
  for (const revision of [...history].reverse()) {
    const entry = text("div", "", "history-item");
    entry.append(
      text(
        "div",
        `r${revision.revision} · ${revision.status.replaceAll("_", " ")}`,
      ),
      text(
        "p",
        `${revision.actor} · ${new Date(revision.recorded_at).toLocaleTimeString()}`,
      ),
      text("p", revision.rationale),
      text(
        "p",
        `${revision.segments.length} segments · ${revision.active_s.toFixed(1)}s active · ${revision.condition}`,
      ),
    );
    const restore = text("button", "Restore as draft", "text-button");
    restore.onclick = async () => {
      if (state.saving) return;
      const id = state.current.episode.id,
        token = state.loadToken,
        version = state.editVersion;
      try {
        const perception = revision.perception_run
          ? await api(`/api/perception/${id}/${revision.perception_run}`)
          : null;
        if (
          token !== state.loadToken ||
          version !== state.editVersion ||
          state.saving
        )
          return;
        state.segments = structuredClone(revision.segments);
        state.suggestionOrigin = revision.suggestion_run || null;
        state.perceptionOrigin = revision.perception_run || null;
        state.condition = revision.condition;
        state.current.perception = perception;
        $("manual-mode").checked = state.condition === "manual";
        $("outcome").value = revision.outcome;
        for (const input of $("flags").querySelectorAll("input"))
          input.checked = revision.quality_flags.includes(input.value);
        $("rationale").value = `Restore revision ${revision.revision}`;
        dirty();
        drawSegments();
        drawSignals();
        updateClock();
      } catch (error) {
        message(error.message, true);
      }
    };
    entry.append(restore);
    $("history").append(entry);
  }
}
function drawStreams() {
  if (!state.current) return;
  const canvas = $("streams");
  canvas.width = Math.max(320, canvas.clientWidth * devicePixelRatio);
  canvas.height = 110 * devicePixelRatio;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const e = state.current.episode;
  $("stream-count").textContent = `${e.streams.length} available`;
  $("stream-legend").textContent =
    e.streams.map((s) => `${s.name}[0] (${s.units})`).join("   /   ") ||
    "State, actions, and sensors unavailable for this recording.";
  $("clock-note").textContent = e.streams.length
    ? "First value component shown. Original times + declared offsets; unmatched frame mappings remain null."
    : "Video presentation timestamps preserved. No robot state inferred from video.";
  e.streams.forEach((stream, index) => {
    const values = stream.values.map((r) => r[0]);
    const min = Math.min(...values),
      max = Math.max(...values);
    ctx.strokeStyle = colors[index % colors.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    stream.timestamps_s.forEach((t, i) => {
      const x = ((t + stream.offset_s) / e.duration_s) * canvas.width,
        y =
          canvas.height -
          15 -
          ((values[i] - min) / (max - min || 1)) * (canvas.height - 30);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  const x = ($("video").currentTime / e.duration_s) * canvas.width;
  ctx.strokeStyle = "#e3ebc8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, canvas.height);
  ctx.stroke();
}
function updateClock() {
  if (!state.current) return;
  const video = $("video"),
    e = state.current.episode,
    t = video.currentTime;
  $("timecode").textContent = fmt(t);
  $("play-time").textContent = `${fmt(t)} / ${fmt(e.duration_s)}`;
  $("seek").value = t;
  let frame = 0;
  while (frame + 1 < e.timestamps_s.length && e.timestamps_s[frame + 1] <= t)
    frame++;
  $("frame-label").textContent =
    `frame ${frame} · PTS ${e.original_pts[frame]}`;
  const canvas = $("overlay");
  canvas.width = 320;
  canvas.height = 180;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, 320, 180);
  if ($("show-tracks").checked && state.condition !== "manual") {
    const evidence = state.run?.evidence[frame];
    if (evidence) {
      ctx.strokeStyle = "#d0ec91";
      ctx.lineWidth = 0.7;
      if (evidence.box_xywh) ctx.strokeRect(...evidence.box_xywh);
      ctx.strokeStyle = "#72c4ae";
      for (const [x, y, u, v] of evidence.flow_vectors) {
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(u, v);
        ctx.stroke();
      }
    }
  }
  if (
    state.condition !== "manual" &&
    state.current.perception &&
    $("show-tracks").checked
  ) {
    const sample = state.current.perception.samples.find(
      (s) => Math.abs(s.time_s - t) < 0.25,
    );
    if (sample) {
      ctx.strokeStyle = "#e9bd72";
      ctx.fillStyle = "#e9bd72";
      ctx.font = "7px monospace";
      for (const d of sample.detections) {
        const [x, y, u, v] = d.xyxy_normalized;
        ctx.strokeRect(x * 320, y * 180, (u - x) * 320, (v - y) * 180);
        ctx.fillText(
          `${d.label} ${d.score.toFixed(2)}`,
          x * 320,
          Math.max(8, y * 180 - 3),
        );
      }
    }
  }
  drawStreams();
}
function payload() {
  return {
    expected_revision: state.current.review.revision,
    actor: $("actor").value,
    rationale: $("rationale").value,
    segments: state.segments,
    outcome: $("outcome").value,
    quality_flags: [...$("flags").querySelectorAll("input:checked")].map(
      (i) => i.value,
    ),
    active_s: state.active,
    wall_s: state.wall,
    condition: state.condition,
    suggestion_run:
      state.condition === "assisted" ? state.suggestionOrigin : null,
    perception_run:
      state.condition === "assisted" ? state.perceptionOrigin : null,
  };
}
async function commit(action) {
  if (!state.current || state.saving) return;
  state.saving = true;
  state.loadToken++;
  const controls = [
    ...document.querySelectorAll("button,input,select,textarea"),
  ].map((node) => [node, node.disabled]);
  for (const [node] of controls) node.disabled = true;
  const id = state.current.episode.id;
  try {
    const revision = await api(`/api/reviews/${id}/${action}`, payload());
    state.dirty = false;
    await refreshQueue();
    await openEpisode(id, true);
    message(
      `Revision ${revision.revision} ${revision.status.replaceAll("_", " ")}. History preserved.`,
    );
  } catch (error) {
    message(error.message, true);
  } finally {
    state.saving = false;
    for (const [node, disabled] of controls) node.disabled = disabled;
    drawQueue();
    drawTracks();
  }
}
$("play").onclick = () => {
  $("video").paused ? $("video").play() : $("video").pause();
};
$("video").onplay = () => {
  $("play").textContent = "Ⅱ";
};
$("video").onpause = () => {
  $("play").textContent = "▶";
};
$("video").ontimeupdate = updateClock;
$("video").onloadedmetadata = updateClock;
$("seek").oninput = () => {
  $("video").currentTime = Number($("seek").value);
  updateClock();
};
$("speed").onchange = () => {
  $("video").playbackRate = Number($("speed").value);
};
$("show-tracks").onchange = updateClock;
$("load-suggestions").onclick = () => {
  if (
    state.segments.length &&
    !confirm("Replace unsaved segments with machine candidates?")
  )
    return;
  state.suggestionOrigin = state.run.id;
  state.segments = state.run.segments.map(({ start_s, end_s, label }) => ({
    start_s,
    end_s,
    label,
  }));
  dirty();
  drawSegments();
};
$("add-segment").onclick = () => {
  const start = $("video").currentTime;
  state.segments.push({
    start_s: start,
    end_s: Math.min(start + 1, state.current.episode.duration_s),
    label: "unknown",
  });
  state.segments.sort((a, b) => a.start_s - b.start_s);
  dirty();
  drawSegments();
};
$("manual-mode").onchange = () => {
  if (state.dirty && !confirm("Change condition and clear unsaved segments?")) {
    $("manual-mode").checked = state.condition === "manual";
    return;
  }
  state.condition = $("manual-mode").checked ? "manual" : "assisted";
  state.suggestionOrigin = null;
  state.perceptionOrigin =
    state.condition === "assisted"
      ? state.current.active_perception?.id || null
      : null;
  state.current.perception =
    state.condition === "assisted" ? state.current.active_perception : null;
  state.segments = [];
  dirty();
  drawSegments();
  drawSignals();
  updateClock();
};
for (const action of ["save", "accept", "uncertain", "exclude"])
  $(action).onclick = () => commit(action);
$("outcome").onchange = dirty;
$("rationale").oninput = dirty;
$("actor").oninput = dirty;
$("search").oninput = drawQueue;
for (const button of document.querySelectorAll("[data-filter]"))
  button.onclick = () => {
    state.filter = button.dataset.filter;
    for (const b of document.querySelectorAll("[data-filter]"))
      b.classList.toggle("active", b === button);
    drawQueue();
  };
$("pause-timer").onclick = () => {
  state.paused = !state.paused;
  $("pause-timer").textContent = state.paused ? "Resume" : "Pause";
};
setInterval(() => {
  const now = performance.now(),
    delta = (now - state.last) / 1000;
  state.last = now;
  if (state.current) {
    state.wall += delta;
    if (
      !state.paused &&
      !document.hidden &&
      (now - state.input < 30000 || (!$("video").paused && !$("video").ended))
    )
      state.active += delta;
    $("active-time").textContent = `${state.active.toFixed(0)}s`;
    $("wall-time").textContent = `${state.wall.toFixed(0)}s`;
  }
}, 250);
for (const event of ["pointerdown", "pointermove", "keydown"])
  document.addEventListener(event, () => {
    state.input = performance.now();
  });
document.addEventListener("keydown", (event) => {
  if (
    !state.current ||
    state.saving ||
    ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(event.target.tagName)
  )
    return;
  if (event.code === "Space") {
    event.preventDefault();
    $("play").click();
  }
  if (event.key === "ArrowRight")
    $("video").currentTime = Math.min(
      state.current.episode.duration_s,
      $("video").currentTime + 0.1,
    );
  if (event.key === "ArrowLeft")
    $("video").currentTime = Math.max(0, $("video").currentTime - 0.1);
  if (["[", "]"].includes(event.key) && state.selected >= 0) {
    state.segments[state.selected][event.key === "[" ? "start_s" : "end_s"] =
      $("video").currentTime;
    dirty();
    drawSegments();
  }
});
window.addEventListener("beforeunload", (e) => {
  if (state.dirty) {
    e.preventDefault();
    e.returnValue = "";
  }
});
window.addEventListener("resize", drawStreams);
$("export-open").onclick = async () => {
  await refreshQueue();
  $("export-list").replaceChildren();
  $("export-result").replaceChildren();
  for (const item of state.items.filter((i) => i.status === "accepted")) {
    const row = text("div", "", "export-row");
    const label = text("label", "");
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = true;
    check.dataset.id = item.id;
    check.dataset.revision = item.revision;
    label.append(
      check,
      document.createTextNode(
        `${item.task} · ${item.id.slice(0, 8)} · r${item.revision}`,
      ),
    );
    const split = document.createElement("select");
    split.setAttribute("aria-label", `Split ${item.id}`);
    for (const name of ["train", "development", "test"]) {
      const option = text("option", name);
      option.value = name;
      split.append(option);
    }
    row.append(label, split);
    $("export-list").append(row);
  }
  if (!$("export-list").children.length)
    $("export-list").append(
      text(
        "p",
        "Accept a saved revision to make it eligible for export.",
        "subtle",
      ),
    );
  $("export-dialog").showModal();
};
$("export-close").onclick = () => $("export-dialog").close();
$("export-confirm").onclick = async () => {
  try {
    const selection = [...$("export-list").querySelectorAll(".export-row")]
      .filter((row) => row.querySelector("input").checked)
      .map((row) => ({
        episode_id: row.querySelector("input").dataset.id,
        revision: Number(row.querySelector("input").dataset.revision),
        split: row.querySelector("select").value,
      }));
    const result = await api("/api/export", { selection });
    $("export-result").replaceChildren(
      text(
        "p",
        `Validated version ${result.manifest.version} · ${result.manifest.episode_count} episode(s)`,
      ),
    );
    for (const name of [
      "manifest.json",
      "episodes.jsonl",
      "episodes.parquet",
    ]) {
      const link = text("a", `Download ${name}`);
      link.href = result.download.replace("manifest.json", name);
      link.style.display = "block";
      $("export-result").append(link);
    }
  } catch (error) {
    $("export-result").textContent = error.message;
  }
};
refreshQueue()
  .then(() =>
    state.items.length
      ? openEpisode(
          state.items.find((item) => item.id === location.hash.slice(1))?.id ||
            state.items[0].id,
        )
      : ($("empty").hidden = false),
  )
  .catch((error) => {
    $("empty").hidden = false;
    $("empty").textContent = error.message;
  });
