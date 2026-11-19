// Project view: title/artist editing, music bar, decoder check.

import { api, fmtTime } from "./api.js";
import { Player } from "./player.js";
import { Timeline } from "./timeline.js";
import { measureDecoderOffset } from "./decodercheck.js";
import { TimingControls } from "./timing.js";
import { AnalysisView } from "./analysis.js";
import { GeneratePanel } from "./generate.js";

const player = new Player();
let timeline = null;
let timing = null;
let analysis = null;
let current = null;   // project
let gen = null;
let raf = 0;

export function initProject() {
  timeline = new Timeline(document.getElementById("wave"), document.getElementById("overview"), {
    getTime: () => player.time,
    onSeek: (t) => { player.seek(t); timeline.draw(); },
  });
  timing = new TimingControls(player, timeline);
  analysis = new AnalysisView(timeline, timing);
  gen = new GeneratePanel(player, timeline, timing, analysis);
  timing.onConfirmed = (p) => { current = p; analysis.load(p); gen.project = p; gen.sync(); };
  const prevChange = timing.onChange;
  timing.onChange = () => { prevChange(); gen.sync(); };
  player.onstate = () => {
    document.getElementById("btn-play").textContent = player.playing ? "❚❚" : "▶";
    timeline.playing = player.playing;
    if (player.playing && !raf) loop();
    updateTime();
    timeline.draw();
  };

  document.getElementById("btn-play").onclick = () => player.toggle();
  document.getElementById("vol").oninput = (e) => player.setVolume(+e.target.value);
  document.getElementById("follow").onchange = (e) => { timeline.follow = e.target.checked; };
  document.getElementById("btn-zoom-in").onclick = () => timeline.zoom(1.5);
  document.getElementById("btn-zoom-out").onclick = () => timeline.zoom(1 / 1.5);

  for (const [id, key] of [["p-title", "title"], ["p-artist", "artist"]]) {
    const el = document.getElementById(id);
    el.addEventListener("change", async () => {
      try { current = await api.patchProject(current.id, { [key]: el.value }); }
      catch (e) { alert(e.message); el.value = current[key]; }
    });
  }

  document.addEventListener("keydown", (e) => {
    if (document.getElementById("view-project").hidden) return;
    const tag = e.target.tagName;
    if (tag === "SELECT" || (tag === "INPUT" && !["checkbox", "range", "button"].includes(e.target.type))) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const step = e.shiftKey ? 0.1 : 1;
    if (e.code === "Space") { e.preventDefault(); player.toggle(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); player.seek(player.time - step); timeline.draw(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); player.seek(player.time + step); timeline.draw(); }
    else if (e.key === "Home") { player.seek(0); timeline.centerOn(0); timeline.draw(); }
    else if (e.key === "+" || e.key === "=") timeline.zoom(1.5);
    else if (e.key === "-") timeline.zoom(1 / 1.5);
  });
}

function loop() {
  updateTime();
  timeline.draw();
  raf = player.playing ? requestAnimationFrame(loop) : 0;
}

function updateTime() {
  document.getElementById("time").textContent =
    `${fmtTime(player.time)} / ${fmtTime(current?.duration ?? 0)}`;
}

export async function showProject(id) {
  player.stop();
  const check = document.getElementById("decoder-check");
  check.textContent = "Decoder check: loading audio…";
  current = await api.getProject(id);
  document.getElementById("crumb").textContent = "› " + current.title;
  document.getElementById("p-title").value = current.title;
  document.getElementById("p-artist").value = current.artist;
  const jk = document.getElementById("p-jacket");
  jk.hidden = !current.jacket;
  if (current.jacket) jk.src = api.jacketUrl(id) + "?t=" + Date.now();
  document.getElementById("p-info").textContent =
    `id ${current.id} · source ${current.source_name} · ${current.duration.toFixed(3)} s`;

  const [meta, peaks, bytes] = await Promise.all([api.peaksMeta(id), api.peaks(id), api.trackBytes(id)]);
  timeline.markers = [{ t: meta.first_sound, color: "#ffab40", label: "first sound" }];
  timeline.setAudio(peaks, meta, current.duration);
  timing.load(current, meta);
  analysis.load(current);
  gen.load(current);

  const buf = await player.load(bytes);
  player.setVolume(+document.getElementById("vol").value);
  const r = measureDecoderOffset(buf, peaks, meta);
  if (r.ok) {
    player.decoderOffset = r.offset;
    const ms = r.offset * 1000;
    const match = `match ${(r.confidence * 100).toFixed(0)}%`;
    check.textContent = Math.abs(ms) < 0.5
      ? `Decoder check: browser playback lines up with the analysis audio (${match}). No correction needed.`
      : `Decoder check: browser audio is ${Math.abs(ms).toFixed(1)} ms ${ms > 0 ? "later" : "earlier"} ` +
        `than the analysis audio (${match}). Corrected automatically.`;
  } else {
    player.decoderOffset = 0;
    check.textContent = `Decoder check: inconclusive (${r.reason ?? "low match " + (r.confidence * 100).toFixed(0) + "%"}), no correction applied.`;
  }
  updateTime();
  timeline.draw();
}

export function leaveProject() {
  player.pause();
}

/** True when the user has timing edits that aren't confirmed yet. */
export function hasUnsavedTiming() {
  return !!timing?.project && timing.dirty && !document.getElementById("view-project").hidden;
}