#!/usr/bin/env python3
"""
Renders a high-definition 3D video/GIF of the DOGU robot trajectory
using perspective projection, depth-sorted polygonal limbs, ground shadows,
and real-time telemetry HUD. Outputs to AVI (Motion-JPEG) and animated GIF.
"""

import argparse
import json
import math
import os
import sys
import cv2
import imageio
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Render DOGU 3D trajectory to video/GIF")
    parser.add_argument("--trajectory", type=str, required=True, help="Path to trajectory JSON")
    parser.add_argument("--output_avi", type=str, required=True, help="Path to output AVI video")
    parser.add_argument("--output_gif", type=str, default=None, help="Path to output GIF")
    parser.add_argument("--fps", type=int, default=25, help="Playback FPS")
    parser.add_argument("--width", type=int, default=800, help="Video width")
    parser.add_argument("--height", type=int, default=500, help="Video height")
    return parser.parse_args()

def project_3d_to_2d(point_3d, camera_pos, camera_target, fov_deg, width, height):
    """Perspective projection of a 3D point (x, y, z) to 2D screen coordinates."""
    px, py, pz = point_3d
    cx, cy, cz = camera_pos
    tx, ty, tz = camera_target

    fx, fy, fz = tx - cx, ty - cy, tz - cz
    f_len = math.sqrt(fx*fx + fy*fy + fz*fz)
    fx, fy, fz = fx/f_len, fy/f_len, fz/f_len

    wx, wy, wz = 0.0, 0.0, 1.0

    rx = fy*wz - fz*wy
    ry = fz*wx - fx*wz
    rz = fx*wy - fy*wx
    r_len = math.sqrt(rx*rx + ry*ry + rz*rz)
    if r_len < 1e-6:
        rx, ry, rz = 1.0, 0.0, 0.0
    else:
        rx, ry, rz = rx/r_len, ry/r_len, rz/r_len

    ux = ry*fz - rz*fy
    uy = rz*fx - rx*fz
    uz = rx*fy - ry*fx

    dx, dy, dz = px - cx, py - cy, pz - cz

    x_cam = dx*rx + dy*ry + dz*rz
    y_cam = dx*ux + dy*uy + dz*uz
    z_cam = dx*fx + dy*fy + dz*fz

    if z_cam <= 0.05:
        return None, z_cam

    aspect = width / height
    fov_rad = math.radians(fov_deg)
    f = 1.0 / math.tan(fov_rad / 2.0)

    x_proj = (x_cam / z_cam) * (f / aspect)
    y_proj = (y_cam / z_cam) * f

    u = int((x_proj + 1.0) * 0.5 * width)
    v = int((1.0 - y_proj) * 0.5 * height)

    return (u, v), z_cam

def draw_capsule(img, p1_2d, p2_2d, radius_px, color, outline_color=(30, 30, 30)):
    if p1_2d is None or p2_2d is None:
        return
    u1, v1 = p1_2d
    u2, v2 = p2_2d
    cv2.line(img, (u1, v1), (u2, v2), outline_color, int(radius_px * 2 + 2), cv2.LINE_AA)
    cv2.line(img, (u1, v1), (u2, v2), color, int(radius_px * 2), cv2.LINE_AA)
    cv2.circle(img, (u1, v1), int(radius_px), color, -1, cv2.LINE_AA)
    cv2.circle(img, (u2, v2), int(radius_px), color, -1, cv2.LINE_AA)

def main():
    args = parse_args()

    with open(args.trajectory, "r") as f:
        data = json.load(f)

    trajectory = data["trajectory"]
    task_name = data.get("task", "DOGU Locomotion").upper()

    width, height = args.width, args.height
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_avi)), exist_ok=True)
    out_avi = cv2.VideoWriter(args.output_avi, fourcc, args.fps, (width, height))

    legs = [
        {"name": "FL", "hip": "fl_hip_link", "femur": "fl_femur_link", "tibia": "fl_tibia_link", "color": (210, 150, 40)},
        {"name": "FR", "hip": "fr_hip_link", "femur": "fr_femur_link", "tibia": "fr_tibia_link", "color": (190, 130, 30)},
        {"name": "BL", "hip": "bl_hip_link", "femur": "bl_femur_link", "tibia": "bl_tibia_link", "color": (220, 100, 30)},
        {"name": "BR", "hip": "br_hip_link", "femur": "br_femur_link", "tibia": "br_tibia_link", "color": (200, 90, 25)},
    ]

    grid_color = (55, 60, 70)
    shadow_color = (30, 33, 40)
    frames_rgb = []

    # Downsample trajectory if step rate is 50Hz and video is 25fps (step by 2)
    step_skip = max(1, int(50 / args.fps))
    sub_trajectory = trajectory[::step_skip]

    print(f"[*] Rendering {len(sub_trajectory)} frames for {task_name}...")

    for frame in sub_trajectory:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            c = int(22 + (y / height) * 18)
            img[y, :] = (c, c, c + 8)

        base_pos = frame["base_pos"]
        body_pos = frame.get("body_pos", {})

        cam_target = (base_pos[0], base_pos[1], 0.08)
        cam_pos = (base_pos[0] + 0.42, base_pos[1] - 0.50, 0.32)
        fov = 52.0

        # Ground Grid
        grid_range_x = np.arange(-0.5, 1.2, 0.1)
        grid_range_y = np.arange(-0.6, 0.6, 0.1)

        for gx in grid_range_x:
            p1, _ = project_3d_to_2d((gx, -0.6, 0.0), cam_pos, cam_target, fov, width, height)
            p2, _ = project_3d_to_2d((gx, 0.6, 0.0), cam_pos, cam_target, fov, width, height)
            if p1 and p2:
                cv2.line(img, p1, p2, grid_color, 1, cv2.LINE_AA)

        for gy in grid_range_y:
            p1, _ = project_3d_to_2d((-0.5, gy, 0.0), cam_pos, cam_target, fov, width, height)
            p2, _ = project_3d_to_2d((1.2, gy, 0.0), cam_pos, cam_target, fov, width, height)
            if p1 and p2:
                cv2.line(img, p1, p2, grid_color, 1, cv2.LINE_AA)

        # Shadows on ground
        base_shadow, _ = project_3d_to_2d((base_pos[0], base_pos[1], 0.001), cam_pos, cam_target, fov, width, height)
        if base_shadow:
            cv2.ellipse(img, base_shadow, (38, 20), 0, 0, 360, shadow_color, -1, cv2.LINE_AA)

        for leg in legs:
            tibia_pos = body_pos.get(leg["tibia"], None)
            if tibia_pos:
                foot_shadow, _ = project_3d_to_2d((tibia_pos[0], tibia_pos[1], 0.001), cam_pos, cam_target, fov, width, height)
                if foot_shadow:
                    cv2.circle(img, foot_shadow, 6, shadow_color, -1, cv2.LINE_AA)

        elements = []
        base_2d, base_z = project_3d_to_2d(base_pos, cam_pos, cam_target, fov, width, height)
        if base_2d:
            elements.append({"type": "base", "z": base_z, "center_2d": base_2d})

        for leg in legs:
            hip_p = body_pos.get(leg["hip"], None)
            femur_p = body_pos.get(leg["femur"], None)
            tibia_p = body_pos.get(leg["tibia"], None)

            if hip_p and femur_p and tibia_p:
                h_2d, hz = project_3d_to_2d(hip_p, cam_pos, cam_target, fov, width, height)
                f_2d, fz = project_3d_to_2d(femur_p, cam_pos, cam_target, fov, width, height)
                t_2d, tz = project_3d_to_2d(tibia_p, cam_pos, cam_target, fov, width, height)

                elements.append({"type": "limb", "z": (base_z + hz)/2.0, "p1": base_2d, "p2": h_2d, "radius": 6, "color": (120, 120, 130)})
                elements.append({"type": "limb", "z": (hz + fz)/2.0, "p1": h_2d, "p2": f_2d, "radius": 5, "color": leg["color"]})
                elements.append({"type": "limb", "z": (fz + tz)/2.0, "p1": f_2d, "p2": t_2d, "radius": 4, "color": (210, 215, 220)})
                elements.append({"type": "foot", "z": tz, "p": t_2d, "radius": 5, "color": (60, 220, 120) if tibia_p[2] < 0.025 else (230, 230, 230)})

        elements.sort(key=lambda item: item["z"], reverse=True)

        for el in elements:
            if el["type"] == "limb":
                draw_capsule(img, el["p1"], el["p2"], el["radius"], el["color"])
            elif el["type"] == "foot":
                if el["p"]:
                    cv2.circle(img, el["p"], el["radius"] + 1, (20, 20, 20), -1, cv2.LINE_AA)
                    cv2.circle(img, el["p"], el["radius"], el["color"], -1, cv2.LINE_AA)
            elif el["type"] == "base":
                b_u, b_v = el["center_2d"]
                cv2.ellipse(img, (b_u, b_v), (36, 18), 0, 0, 360, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.ellipse(img, (b_u, b_v), (34, 16), 0, 0, 360, (60, 140, 215), -1, cv2.LINE_AA)
                cv2.circle(img, (b_u, b_v), 5, (255, 255, 255), -1, cv2.LINE_AA)

        # Telemetry HUD
        hud_bg = (18, 20, 24)
        cv2.rectangle(img, (15, 15), (320, 190), hud_bg, -1)
        cv2.rectangle(img, (15, 15), (320, 190), (50, 55, 65), 1, cv2.LINE_AA)

        cv2.putText(img, f"DOGU | {task_name}", (25, 38), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, (25, 46), (310, 46), (70, 75, 85), 1)

        t = frame["time"]
        step = frame["step"]
        h = base_pos[2]
        x_dist = base_pos[0]

        cv2.putText(img, f"Time   : {t:0.2f} s (Step {step})", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 195, 205), 1, cv2.LINE_AA)
        cv2.putText(img, f"Height : {h*100:0.1f} cm (Ref: 10.8 cm)", (25, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 195, 205), 1, cv2.LINE_AA)
        cv2.putText(img, f"Displ. : {x_dist*100:+0.1f} cm", (25, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 195, 205), 1, cv2.LINE_AA)

        cv2.putText(img, "Feet   :", (25, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 195, 205), 1, cv2.LINE_AA)
        for i, leg in enumerate(legs):
            tibia_p = body_pos.get(leg["tibia"], [0, 0, 0.1])
            is_contact = tibia_p[2] < 0.025
            dot_color = (60, 220, 100) if is_contact else (90, 95, 105)
            lx = 85 + i * 40
            cv2.circle(img, (lx, 137), 5, dot_color, -1, cv2.LINE_AA)
            cv2.putText(img, leg["name"], (lx - 7, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 175, 185), 1, cv2.LINE_AA)

        cv2.putText(img, "Isaac Lab 17.0.2 | RSL-RL PPO", (25, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 180, 240), 1, cv2.LINE_AA)

        out_avi.write(img)
        if args.output_gif:
            # OpenCV BGR -> RGB for imageio
            frames_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    out_avi.release()
    print(f"[+] AVI video written: {args.output_avi}")

    if args.output_gif and len(frames_rgb) > 0:
        print(f"[*] Encoding animated GIF: {args.output_gif}...")
        imageio.mimsave(args.output_gif, frames_rgb, fps=args.fps, loop=0)
        print(f"[+] Animated GIF written: {args.output_gif}")

if __name__ == "__main__":
    main()
