#!/usr/bin/env python3
"""
Generates an interactive, zero-dependency HTML5/Canvas 3D robot visualizer
for the DOGU quadruped, embedding trajectory playback and live telemetry.
"""

import json
import os
import sys

def main():
    traj_locomotion_path = "/home/chakradhar/robotics/dogu_locomotion/evaluations/locomotion_trajectory.json"
    traj_standing_path = "/home/chakradhar/robotics/dogu_locomotion/evaluations/standing_trajectory.json"

    with open(traj_locomotion_path, "r") as f:
        data_locomotion = json.load(f)

    with open(traj_standing_path, "r") as f:
        data_standing = json.load(f)

    # Minify trajectory data for web embedding (every 2nd frame is plenty for 25fps smooth web rendering)
    loco_frames = data_locomotion["trajectory"][::2]
    stand_frames = data_standing["trajectory"][::2]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>DOGU Quadruped 3D Trajectory Viewer</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    body {{
      background-color: var(--background, #0f1117);
      color: var(--foreground, #e2e8f0);
      font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
      margin: 0;
      padding: 0;
      overflow: hidden;
      user-select: none;
    }}
    canvas {{
      display: block;
      cursor: grab;
    }}
    canvas:active {{
      cursor: grabbing;
    }}
  </style>
</head>
<body class="flex flex-col h-screen w-screen">

  <!-- Top Navigation & Controls Header -->
  <header class="bg-[var(--card,#1a1d26)] border-b border-[var(--border,#2e3444)] px-4 py-2.5 flex items-center justify-between shrink-0">
    <div class="flex items-center gap-3">
      <div class="w-3 h-3 rounded-full bg-emerald-400 animate-pulse"></div>
      <h1 class="font-bold text-base tracking-wide flex items-center gap-2">
        <span class="text-[var(--primary,#38bdf8)]">DOGU Quadruped</span>
        <span class="text-xs px-2 py-0.5 rounded bg-[var(--border,#2e3444)] text-[var(--muted-foreground,#94a3b8)]">Headless RL Playback</span>
      </h1>
    </div>

    <!-- Mode Selector Buttons -->
    <div class="flex items-center gap-2 bg-[var(--background,#0f1117)] p-1 rounded-lg border border-[var(--border,#2e3444)]">
      <button id="btn-locomotion" class="px-3 py-1 text-xs font-semibold rounded-md transition-colors bg-blue-600 text-white shadow">
        Forward Locomotion
      </button>
      <button id="btn-standing" class="px-3 py-1 text-xs font-semibold rounded-md transition-colors text-[var(--muted-foreground,#94a3b8)] hover:text-white">
        Standing Balance
      </button>
    </div>

    <!-- Camera View Presets -->
    <div class="flex items-center gap-1.5 text-xs">
      <span class="text-[var(--muted-foreground,#94a3b8)] mr-1">Camera:</span>
      <button onclick="setCameraView('iso')" class="px-2 py-1 bg-[var(--border,#2e3444)] hover:bg-slate-700 rounded text-xs">3D Iso</button>
      <button onclick="setCameraView('side')" class="px-2 py-1 bg-[var(--border,#2e3444)] hover:bg-slate-700 rounded text-xs">Side</button>
      <button onclick="setCameraView('front')" class="px-2 py-1 bg-[var(--border,#2e3444)] hover:bg-slate-700 rounded text-xs">Front</button>
      <button onclick="setCameraView('top')" class="px-2 py-1 bg-[var(--border,#2e3444)] hover:bg-slate-700 rounded text-xs">Top</button>
    </div>
  </header>

  <!-- Main Viewport + Overlay HUD -->
  <div class="relative flex-1 w-full overflow-hidden bg-gradient-to-b from-[#11131a] to-[#0a0c10]">
    <canvas id="glcanvas" class="w-full h-full"></canvas>

    <!-- Floating Telemetry HUD -->
    <div class="absolute top-4 left-4 bg-[var(--card,#1a1d26)]/90 backdrop-blur-md border border-[var(--border,#2e3444)] rounded-xl p-3.5 shadow-xl w-64 pointer-events-none">
      <div class="text-xs font-semibold text-[var(--muted-foreground,#94a3b8)] uppercase tracking-wider mb-2 flex justify-between">
        <span>Kinematic Telemetry</span>
        <span id="hud-task-tag" class="text-blue-400 font-mono">LOCOMOTION</span>
      </div>
      <div class="space-y-1.5 text-xs font-mono">
        <div class="flex justify-between">
          <span class="text-[var(--muted-foreground,#94a3b8)]">Height (Z):</span>
          <span id="hud-height" class="font-bold text-white">0.0 cm</span>
        </div>
        <div class="flex justify-between">
          <span class="text-[var(--muted-foreground,#94a3b8)]">Forward (X):</span>
          <span id="hud-pos-x" class="font-bold text-emerald-400">+0.0 cm</span>
        </div>
        <div class="flex justify-between">
          <span class="text-[var(--muted-foreground,#94a3b8)]">Lateral (Y):</span>
          <span id="hud-pos-y" class="font-bold text-slate-300">0.0 cm</span>
        </div>
        <div class="border-t border-[var(--border,#2e3444)] pt-2 mt-2">
          <div class="text-[11px] text-[var(--muted-foreground,#94a3b8)] mb-1">Foot Ground Contacts:</div>
          <div class="grid grid-cols-4 gap-1 text-center font-bold text-[10px]">
            <div id="contact-fl" class="py-1 rounded bg-slate-800 text-slate-400">FL</div>
            <div id="contact-fr" class="py-1 rounded bg-slate-800 text-slate-400">FR</div>
            <div id="contact-bl" class="py-1 rounded bg-slate-800 text-slate-400">BL</div>
            <div id="contact-br" class="py-1 rounded bg-slate-800 text-slate-400">BR</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Mouse Interaction Help Badge -->
    <div class="absolute bottom-4 left-4 bg-black/40 backdrop-blur-sm border border-white/5 rounded-lg px-2.5 py-1 text-[11px] text-slate-400 pointer-events-none">
      Left Drag: Orbit | Right Drag: Pan | Scroll: Zoom
    </div>
  </div>

  <!-- Bottom Playback Timeline Bar -->
  <footer class="bg-[var(--card,#1a1d26)] border-t border-[var(--border,#2e3444)] px-4 py-3 flex items-center gap-4 shrink-0">
    <!-- Play/Pause Button -->
    <button id="btn-play" class="w-9 h-9 flex items-center justify-center rounded-full bg-blue-600 hover:bg-blue-500 text-white shadow transition-transform active:scale-95">
      <svg id="icon-play" class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
      <svg id="icon-pause" class="w-4 h-4 hidden" fill="currentColor" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
    </button>

    <!-- Timestamp display -->
    <div class="text-xs font-mono text-[var(--muted-foreground,#94a3b8)] min-w-[70px]">
      <span id="time-current">0.00</span>s / <span id="time-total">5.00</span>s
    </div>

    <!-- Scrubber Slider -->
    <input id="timeline-slider" type="range" min="0" max="124" value="0"
           class="flex-1 accent-blue-500 h-1.5 bg-slate-700 rounded-lg cursor-pointer">

    <!-- Speed Selector -->
    <div class="flex items-center gap-1 text-xs">
      <span class="text-[var(--muted-foreground,#94a3b8)]">Speed:</span>
      <select id="speed-select" class="bg-[var(--background,#0f1117)] border border-[var(--border,#2e3444)] rounded px-1.5 py-0.5 text-xs text-white">
        <option value="0.25">0.25x</option>
        <option value="0.5">0.5x</option>
        <option value="1.0" selected>1.0x</option>
        <option value="2.0">2.0x</option>
      </select>
    </div>
  </footer>

  <script>
    const locoData = {json.dumps(loco_frames)};
    const standData = {json.dumps(stand_frames)};

    let currentFrames = locoData;
    let currentTaskName = "LOCOMOTION";
    let frameIdx = 0;
    let isPlaying = true;
    let playbackSpeed = 1.0;
    let lastTime = performance.now();

    // Canvas & 3D Camera State
    const canvas = document.getElementById("glcanvas");
    const ctx = canvas.getContext("2d");

    let camera = {{
      azimuth: 0.75,
      elevation: 0.35,
      distance: 0.85,
      target: [0, 0, 0.08]
    }};

    let isDragging = false;
    let isPanning = false;
    let prevMouseX = 0;
    let prevMouseY = 0;

    function resizeCanvas() {{
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.clientWidth * dpr;
      canvas.height = canvas.clientHeight * dpr;
    }}
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();

    // Mouse Controls
    canvas.addEventListener("mousedown", (e) => {{
      if (e.button === 0) isDragging = true;
      else if (e.button === 2) isPanning = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    }});
    window.addEventListener("mouseup", () => {{
      isDragging = false;
      isPanning = false;
    }});
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    canvas.addEventListener("mousemove", (e) => {{
      const dx = e.clientX - prevMouseX;
      const dy = e.clientY - prevMouseY;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;

      if (isDragging) {{
        camera.azimuth -= dx * 0.008;
        camera.elevation = Math.max(0.05, Math.min(Math.PI / 2 - 0.05, camera.elevation + dy * 0.008));
      }} else if (isPanning) {{
        const panSpeed = 0.001 * camera.distance;
        camera.target[0] -= dy * panSpeed * Math.cos(camera.azimuth);
        camera.target[1] += dy * panSpeed * Math.sin(camera.azimuth);
      }}
    }});

    canvas.addEventListener("wheel", (e) => {{
      e.preventDefault();
      camera.distance = Math.max(0.3, Math.min(2.5, camera.distance + e.deltaY * 0.0015));
    }}, {{ passive: false }});

    function setCameraView(preset) {{
      if (preset === 'iso') {{
        camera.azimuth = 0.75; camera.elevation = 0.40; camera.distance = 0.85;
      }} else if (preset === 'side') {{
        camera.azimuth = 0.0; camera.elevation = 0.08; camera.distance = 0.75;
      }} else if (preset === 'front') {{
        camera.azimuth = -Math.PI / 2; camera.elevation = 0.08; camera.distance = 0.75;
      }} else if (preset === 'top') {{
        camera.azimuth = 0.0; camera.elevation = Math.PI / 2 - 0.05; camera.distance = 0.85;
      }}
    }}

    // Project 3D Point to 2D Screen
    function project3D(point3D, camPos, camTarget) {{
      const [px, py, pz] = point3D;
      const [cx, cy, cz] = camPos;
      const [tx, ty, tz] = camTarget;

      let fx = tx - cx, fy = ty - cy, fz = tz - cz;
      const fLen = Math.hypot(fx, fy, fz);
      fx /= fLen; fy /= fLen; fz /= fLen;

      const wx = 0, wy = 0, wz = 1;
      let rx = fy*wz - fz*wy, ry = fz*wx - fx*wz, rz = fx*wy - fy*wx;
      const rLen = Math.hypot(rx, ry, rz);
      if (rLen > 1e-6) {{ rx /= rLen; ry /= rLen; rz /= rLen; }} else {{ rx = 1; ry = 0; rz = 0; }}

      const ux = ry*fz - rz*fy, uy = rz*fx - rx*fz, uz = rx*fy - ry*fx;

      const dx = px - cx, dy = py - cy, dz = pz - cz;
      const xCam = dx*rx + dy*ry + dz*rz;
      const yCam = dx*ux + dy*uy + dz*uz;
      const zCam = dx*fx + dy*fy + dz*fz;

      if (zCam <= 0.05) return null;

      const aspect = canvas.width / canvas.height;
      const fovRad = 50 * (Math.PI / 180);
      const f = 1.0 / Math.tan(fovRad / 2.0);

      const xProj = (xCam / zCam) * (f / aspect);
      const yProj = (yCam / zCam) * f;

      const u = (xProj + 1.0) * 0.5 * canvas.width;
      const v = (1.0 - yProj) * 0.5 * canvas.height;
      return {{ u, v, z: zCam }};
    }}

    function drawCapsule(p1, p2, radius, color, outlineColor="#1e222d") {{
      if (!p1 || !p2) return;
      ctx.beginPath();
      ctx.lineWidth = radius * 2 + 2;
      ctx.strokeStyle = outlineColor;
      ctx.lineCap = "round";
      ctx.moveTo(p1.u, p1.v);
      ctx.lineTo(p2.u, p2.v);
      ctx.stroke();

      ctx.beginPath();
      ctx.lineWidth = radius * 2;
      ctx.strokeStyle = color;
      ctx.moveTo(p1.u, p1.v);
      ctx.lineTo(p2.u, p2.v);
      ctx.stroke();

      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(p1.u, p1.v, radius, 0, Math.PI * 2);
      ctx.arc(p2.u, p2.v, radius, 0, Math.PI * 2);
      ctx.fill();
    }}

    const legs = [
      {{ name: "FL", hip: "fl_hip_link", femur: "fl_femur_link", tibia: "fl_tibia_link", color: "#f59e0b", contactId: "contact-fl" }},
      {{ name: "FR", hip: "fr_hip_link", femur: "fr_femur_link", tibia: "fr_tibia_link", color: "#d97706", contactId: "contact-fr" }},
      {{ name: "BL", hip: "bl_hip_link", femur: "bl_femur_link", tibia: "bl_tibia_link", color: "#f97316", contactId: "contact-bl" }},
      {{ name: "BR", hip: "br_hip_link", femur: "br_femur_link", tibia: "br_tibia_link", color: "#ea580c", contactId: "contact-br" }}
    ];

    function renderFrame() {{
      const frame = currentFrames[frameIdx];
      const basePos = frame.base_pos;
      const bodyPos = frame.body_pos || {{}};

      // Auto-follow base position
      camera.target[0] = basePos[0];
      camera.target[1] = basePos[1];

      // Calculate Camera Position
      const camX = camera.target[0] + camera.distance * Math.cos(camera.elevation) * Math.cos(camera.azimuth);
      const camY = camera.target[1] + camera.distance * Math.cos(camera.elevation) * Math.sin(camera.azimuth);
      const camZ = camera.target[2] + camera.distance * Math.sin(camera.elevation);
      const camPos = [camX, camY, camZ];

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 1. Draw Ground Grid
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(100, 116, 139, 0.25)";
      for (let gx = -0.5; gx <= 1.5; gx += 0.1) {{
        const p1 = project3D([gx, -0.8, 0], camPos, camera.target);
        const p2 = project3D([gx, 0.8, 0], camPos, camera.target);
        if (p1 && p2) {{
          ctx.beginPath(); ctx.moveTo(p1.u, p1.v); ctx.lineTo(p2.u, p2.v); ctx.stroke();
        }}
      }}
      for (let gy = -0.8; gy <= 0.8; gy += 0.1) {{
        const p1 = project3D([-0.5, gy, 0], camPos, camera.target);
        const p2 = project3D([1.5, gy, 0], camPos, camera.target);
        if (p1 && p2) {{
          ctx.beginPath(); ctx.moveTo(p1.u, p1.v); ctx.lineTo(p2.u, p2.v); ctx.stroke();
        }}
      }}

      // 2. Base Ground Shadow
      const baseShadow = project3D([basePos[0], basePos[1], 0.001], camPos, camera.target);
      if (baseShadow) {{
        ctx.beginPath();
        ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
        ctx.ellipse(baseShadow.u, baseShadow.v, 42, 22, 0, 0, Math.PI * 2);
        ctx.fill();
      }}

      // 3. Collect Depth Elements
      const elements = [];
      const baseProj = project3D(basePos, camPos, camera.target);
      if (baseProj) {{
        elements.push({{ type: "base", z: baseProj.z, proj: baseProj }});
      }}

      legs.forEach(leg => {{
        const hipP = bodyPos[leg.hip];
        const femurP = bodyPos[leg.femur];
        const tibiaP = bodyPos[leg.tibia];

        if (hipP && femurP && tibiaP) {{
          const hP = project3D(hipP, camPos, camera.target);
          const fP = project3D(femurP, camPos, camera.target);
          const tP = project3D(tibiaP, camPos, camera.target);

          if (baseProj && hP) {{
            elements.push({{ type: "limb", z: (baseProj.z + hP.z)/2, p1: baseProj, p2: hP, radius: 6, color: "#64748b" }});
          }}
          if (hP && fP) {{
            elements.push({{ type: "limb", z: (hP.z + fP.z)/2, p1: hP, p2: fP, radius: 5.5, color: leg.color }});
          }}
          if (fP && tP) {{
            elements.push({{ type: "limb", z: (fP.z + tP.z)/2, p1: fP, p2: tP, radius: 4.5, color: "#cbd5e1" }});
          }}
          if (tP) {{
            const isStance = tibiaP[2] < 0.025;
            elements.push({{ type: "foot", z: tP.z, p: tP, radius: 5.5, isStance: isStance }});
            // Update HUD Contact Indicator
            const el = document.getElementById(leg.contactId);
            if (el) {{
              if (isStance) {{
                el.className = "py-1 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
              }} else {{
                el.className = "py-1 rounded bg-slate-800 text-slate-500";
              }}
            }}
          }}
        }}
      }});

      // Depth Sort
      elements.sort((a, b) => b.z - a.z);

      // Draw Elements
      elements.forEach(el => {{
        if (el.type === "limb") {{
          drawCapsule(el.p1, el.p2, el.radius, el.color);
        }} else if (el.type === "foot") {{
          ctx.beginPath();
          ctx.fillStyle = el.isStance ? "#10b981" : "#f1f5f9";
          ctx.arc(el.p.u, el.p.v, el.radius, 0, Math.PI * 2);
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = "#0f172a";
          ctx.stroke();
        }} else if (el.type === "base") {{
          ctx.beginPath();
          ctx.fillStyle = "#38bdf8";
          ctx.ellipse(el.proj.u, el.proj.v, 38, 18, 0, 0, Math.PI * 2);
          ctx.fill();
          ctx.lineWidth = 2;
          ctx.strokeStyle = "#0284c7";
          ctx.stroke();
          ctx.beginPath();
          ctx.fillStyle = "#ffffff";
          ctx.arc(el.proj.u, el.proj.v, 5, 0, Math.PI * 2);
          ctx.fill();
        }}
      }});

      // Update HUD Numbers
      document.getElementById("hud-height").innerText = (basePos[2] * 100).toFixed(1) + " cm";
      document.getElementById("hud-pos-x").innerText = (basePos[0] >= 0 ? "+" : "") + (basePos[0] * 100).toFixed(1) + " cm";
      document.getElementById("hud-pos-y").innerText = (basePos[1] * 100).toFixed(1) + " cm";
      document.getElementById("time-current").innerText = frame.time.toFixed(2);
      document.getElementById("timeline-slider").value = frameIdx;
    }}

    // Animation Loop
    function animate(now) {{
      const dt = (now - lastTime) / 1000;
      lastTime = now;

      if (isPlaying) {{
        frameIdx += playbackSpeed * (dt / 0.04);
        if (frameIdx >= currentFrames.length) {{
          frameIdx = 0;
        }}
      }}

      renderFrame();
      requestAnimationFrame(animate);
    }}
    requestAnimationFrame(animate);

    // UI Listeners
    const btnPlay = document.getElementById("btn-play");
    const iconPlay = document.getElementById("icon-play");
    const iconPause = document.getElementById("icon-pause");
    btnPlay.addEventListener("click", () => {{
      isPlaying = !isPlaying;
      iconPlay.classList.toggle("hidden", isPlaying);
      iconPause.classList.toggle("hidden", !isPlaying);
    }});

    const timeline = document.getElementById("timeline-slider");
    timeline.max = currentFrames.length - 1;
    timeline.addEventListener("input", (e) => {{
      frameIdx = parseInt(e.target.value);
      renderFrame();
    }});

    document.getElementById("speed-select").addEventListener("change", (e) => {{
      playbackSpeed = parseFloat(e.target.value);
    }});

    const btnLoco = document.getElementById("btn-locomotion");
    const btnStand = document.getElementById("btn-standing");
    const taskTag = document.getElementById("hud-task-tag");

    btnLoco.addEventListener("click", () => {{
      currentFrames = locoData;
      currentTaskName = "LOCOMOTION";
      frameIdx = 0;
      timeline.max = currentFrames.length - 1;
      taskTag.innerText = "LOCOMOTION";
      taskTag.className = "text-blue-400 font-mono";
      btnLoco.className = "px-3 py-1 text-xs font-semibold rounded-md transition-colors bg-blue-600 text-white shadow";
      btnStand.className = "px-3 py-1 text-xs font-semibold rounded-md transition-colors text-[var(--muted-foreground,#94a3b8)] hover:text-white";
    }});

    btnStand.addEventListener("click", () => {{
      currentFrames = standData;
      currentTaskName = "STANDING";
      frameIdx = 0;
      timeline.max = currentFrames.length - 1;
      taskTag.innerText = "STANDING";
      taskTag.className = "text-emerald-400 font-mono";
      btnStand.className = "px-3 py-1 text-xs font-semibold rounded-md transition-colors bg-emerald-600 text-white shadow";
      btnLoco.className = "px-3 py-1 text-xs font-semibold rounded-md transition-colors text-[var(--muted-foreground,#94a3b8)] hover:text-white";
    }});
  </script>
</body>
</html>
"""

    out_file = "/home/chakradhar/robotics/dogu_locomotion/evaluations/dogu_3d_viewer.html"
    with open(out_file, "w") as f:
        f.write(html_content)

    print(f"[+] Successfully generated standalone 3D web viewer: {out_file}")

if __name__ == "__main__":
    main()
