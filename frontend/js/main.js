// Tiny hash router:  #/  -> home,  #/p/<id>  -> project

import { initHome, showHome } from "./home.js";
import { initProject, showProject, leaveProject, hasUnsavedTiming } from "./project.js";

const views = {
  home: document.getElementById("view-home"),
  project: document.getElementById("view-project"),
};
const musicbar = document.getElementById("musicbar");

function show(name) {
  for (const [k, el] of Object.entries(views)) el.hidden = k !== name;
  musicbar.hidden = name !== "project";
}

async function route() {
  const m = location.hash.match(/^#\/p\/([a-f0-9]+)$/);
  try {
    if (m) {
      show("project");
      await showProject(m[1]);
    } else {
      leaveProject();
      document.getElementById("crumb").textContent = "";
      show("home");
      await showHome();
    }
  } catch (e) {
    console.error(e);
    alert("Error: " + e.message);
    location.hash = "#/";
  }
}

window.addEventListener("beforeunload", (e) => {
  if (hasUnsavedTiming()) { e.preventDefault(); e.returnValue = ""; }
});

initHome();
initProject();
window.addEventListener("hashchange", route);
route();