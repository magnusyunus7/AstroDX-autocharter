// Home view: drop a song (or a whole level folder), create a project, list projects.

import { api, fmtTime } from "./api.js";

const AUDIO = ["mp3", "flac", "wav", "ogg", "m4a", "aac", "opus"];
const IMAGE = ["jpg", "jpeg", "png", "webp", "bmp"];
const ext = (f) => f.name.split(".").pop().toLowerCase();

let pending = { audio: null, jacket: null };

export function initHome() {
  const dz = document.getElementById("dropzone");
  const input = document.getElementById("file-input");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter") input.click(); });
  input.addEventListener("change", () => { takeFiles([...input.files]); input.value = ""; });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", async (e) => {
    e.preventDefault(); dz.classList.remove("over");
    takeFiles(await filesFromDrop(e.dataTransfer));
  });

  document.getElementById("upload-form").addEventListener("submit", (e) => { e.preventDefault(); submit(); });
  document.getElementById("up-cancel").addEventListener("click", resetForm);
}

export async function showHome() {
  resetForm();
  await refreshList();
}

// Folders dropped from Explorer arrive as directory entries; read one level deep.
async function filesFromDrop(dt) {
  const out = [];
  const entries = [...dt.items].map((it) => it.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...dt.files];
  const fileOf = (entry) => new Promise((res, rej) => entry.file(res, rej));
  for (const en of entries) {
    if (en.isFile) out.push(await fileOf(en));
    else if (en.isDirectory) {
      const kids = await new Promise((res, rej) => en.createReader().readEntries(res, rej));
      for (const k of kids) if (k.isFile) out.push(await fileOf(k));
    }
  }
  return out;
}

async function takeFiles(files) {
  const audio = files.find((f) => AUDIO.includes(ext(f)));
  if (!audio) { alert("No audio file found (mp3, flac, wav, ogg, m4a…)."); return; }
  const jacket = files.find((f) => IMAGE.includes(ext(f)) && /^bg\./i.test(f.name))
    ?? files.find((f) => IMAGE.includes(ext(f))) ?? null;
  const maidata = files.find((f) => f.name.toLowerCase() === "maidata.txt");

  pending = { audio, jacket };
  document.getElementById("up-audio").textContent = audio.name;
  document.getElementById("up-jacket").textContent = jacket ? jacket.name : "none (embedded cover if any)";
  const title = document.getElementById("up-title"), artist = document.getElementById("up-artist");
  title.value = ""; artist.value = "";
  if (maidata) {
    const text = await maidata.text();
    title.value = text.match(/^&title=(.*)$/m)?.[1]?.trim() ?? "";
    artist.value = text.match(/^&artist=(.*)$/m)?.[1]?.trim() ?? "";
  }
  document.getElementById("upload-form").hidden = false;
  document.getElementById("dropzone").hidden = true;
  title.focus();
}

async function submit() {
  const status = document.getElementById("up-status");
  const btn = document.querySelector("#upload-form button[type=submit]");
  const fd = new FormData();
  fd.append("audio", pending.audio);
  if (pending.jacket) fd.append("jacket", pending.jacket);
  const title = document.getElementById("up-title").value.trim();
  const artist = document.getElementById("up-artist").value.trim();
  if (title) fd.append("title", title);
  if (artist) fd.append("artist", artist);
  btn.disabled = true;
  status.textContent = "Uploading and processing…";
  try {
    const p = await api.createProject(fd);
    location.hash = `#/p/${p.id}`;
  } catch (e) {
    status.textContent = "Error: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

function resetForm() {
  pending = { audio: null, jacket: null };
  document.getElementById("upload-form").hidden = true;
  document.getElementById("dropzone").hidden = false;
  document.getElementById("up-status").textContent = "";
}

async function refreshList() {
  const ul = document.getElementById("project-list");
  const projects = await api.listProjects();
  document.getElementById("project-empty").hidden = projects.length > 0;
  ul.replaceChildren(...projects.map((p) => {
    const li = document.createElement("li");
    const img = p.jacket ? Object.assign(document.createElement("img"), { src: api.jacketUrl(p.id) })
                         : Object.assign(document.createElement("div"), { className: "nojacket" });
    const meta = document.createElement("div");
    meta.className = "meta";
    const b = document.createElement("b"); b.textContent = p.title;
    const s = document.createElement("span"); s.className = "hint";
    s.textContent = `${p.artist || "unknown artist"} · ${fmtTime(p.duration).slice(0, 5)}`;
    meta.append(b, s);
    const open = Object.assign(document.createElement("button"), { textContent: "Open" });
    open.onclick = () => { location.hash = `#/p/${p.id}`; };
    const del = Object.assign(document.createElement("button"), { textContent: "Delete", className: "danger" });
    del.onclick = async () => {
      if (!confirm(`Delete "${p.title}"? This removes its project folder.`)) return;
      await api.deleteProject(p.id);
      refreshList();
    };
    li.append(img, meta, open, del);
    return li;
  }));
}