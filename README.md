# Gesture-Driven Virtual Canvas

**CSE 5524 — Project 20 | The Ohio State University**

## Team

| Name | Username | Task |
|---|---|---|
| Salma Adem | adem.18 | Task 3 — Gesture Recognition & Motion Analysis |
| Emerson Frasure | frasure.49 | Task 2 — Geometric Calibration & Workspace Detection |
| Sam Schmidt | schmidt.1284 | Task 1 — Tracking & Pre-processing |

## Project Summary

A virtual whiteboard where a user draws in the air using a colored object or hand gesture. A standard laptop camera observes the scene. The system detects a physical drawing surface (a piece of paper on a desk), corrects the camera's perspective into a flat top-down canvas using homography, tracks a "brush" moving through that space, and interprets hand gestures as commands (draw, erase, clear).

## Task Breakdown

### Task 1 — Tracking & Pre-processing (Sam Schmidt)
Tracks a colored "brush" object in real time and outputs its position in camera space each frame.

### Task 2 — Geometric Calibration & Workspace Detection (Emerson Frasure)
Detects the physical drawing surface and computes a homography that maps camera-space coordinates to a flat, top-down canvas.

### Task 3 — Gesture Recognition & Motion Analysis (Salma Adem)
Classifies hand gestures from the camera feed to issue drawing commands (draw, erase, clear).
