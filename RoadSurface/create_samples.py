"""
Sample Road Surface Image Generator.

Generates realistic synthetic road pavement images containing:
- Natural asphalt texture and fine grain noise
- Realistic distressed regions (potholes with cavity shadows, jagged fissure cracks)
- Lane markings and perspective road cues
- Control images of undamaged pavement
"""

from pathlib import Path
import numpy as np
import cv2


def generate_asphalt_texture(width: int = 1280, height: int = 720, base_val: int = 115) -> np.ndarray:
    """Creates a base pavement texture with multi-scale aggregate grain."""
    np.random.seed(42)
    # Base asphalt gray
    base = np.full((height, width, 3), (base_val, base_val, base_val), dtype=np.uint8)

    # Multi-octave texture noise
    fine_noise = np.random.normal(0, 10, (height, width, 3)).astype(np.int16)
    coarse_noise = cv2.resize(
        np.random.normal(0, 18, (height // 4, width // 4, 3)).astype(np.int16),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    combined = base.astype(np.int16) + fine_noise + coarse_noise
    asphalt = np.clip(combined, 30, 210).astype(np.uint8)

    # Add subtle lane markings (dashed yellow/white)
    # Left white edge line
    cv2.line(asphalt, (120, height), (380, 0), (200, 200, 200), 10)
    # Right white edge line
    cv2.line(asphalt, (width - 120, height), (width - 380, 0), (200, 200, 200), 10)
    # Center dashed yellow line
    for y in range(0, height, 80):
        # Center line with perspective
        y1, y2 = y, y + 45
        x1 = int(width / 2 + (y1 - height / 2) * 0.1)
        x2 = int(width / 2 + (y2 - height / 2) * 0.1)
        cv2.line(asphalt, (x1, y1), (x2, y2), (20, 190, 230), 8)

    return asphalt


def add_pothole(
    img: np.ndarray,
    center: tuple,
    axes: tuple,
    angle: float = 0,
    seed: int = 101
) -> np.ndarray:
    """Renders a realistic pothole with deep shadow cavity and broken edges."""
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Base elliptical crater
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

    # Add jagged irregular perimeter distortion
    num_pts = 36
    distorted_pts = []
    cx, cy = center
    rx, ry = axes
    rad_ang = np.radians(angle)

    for i in range(num_pts):
        theta = 2.0 * np.pi * i / num_pts
        r_jitter = 1.0 + rng.uniform(-0.25, 0.25)
        local_x = rx * np.cos(theta) * r_jitter
        local_y = ry * np.sin(theta) * r_jitter

        rot_x = int(cx + local_x * np.cos(rad_ang) - local_y * np.sin(rad_ang))
        rot_y = int(cy + local_x * np.sin(rad_ang) + local_y * np.cos(rad_ang))
        distorted_pts.append([rot_x, rot_y])

    pts = np.array(distorted_pts, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # Cavity interior is significantly darker (shadow depth + exposed substrate)
    cavity = np.zeros_like(img)
    cavity[:] = (35, 38, 42)  # Dark cavity color
    cavity_noise = rng.normal(0, 8, img.shape).astype(np.int16)
    cavity = np.clip(cavity.astype(np.int16) + cavity_noise, 15, 75).astype(np.uint8)

    # Blend pothole into image using soft boundary
    mask_blur = cv2.GaussianBlur(mask, (7, 7), 2.0)
    alpha = (mask_blur.astype(np.float32) / 255.0)[:, :, np.newaxis]
    blended = (img.astype(np.float32) * (1.0 - alpha) + cavity.astype(np.float32) * alpha).astype(np.uint8)

    # Draw broken asphalt rim around crater
    cv2.polylines(blended, [pts], isClosed=True, color=(20, 20, 20), thickness=2)

    return blended


def add_crack(
    img: np.ndarray,
    start_pt: tuple,
    steps: int = 50,
    main_dir: tuple = (1, 1),
    thickness: int = 3,
    seed: int = 202
) -> np.ndarray:
    """Renders a meandering fissure crack with branches."""
    rng = np.random.RandomState(seed)
    curr_x, curr_y = start_pt
    pts = [(curr_x, curr_y)]

    dx, dy = main_dir
    norm = np.hypot(dx, dy)
    dx, dy = dx / norm, dy / norm

    for _ in range(steps):
        step_len = rng.uniform(6.0, 14.0)
        jitter_angle = rng.uniform(-0.7, 0.7)

        # Rotate direction slightly
        cos_j, sin_j = np.cos(jitter_angle), np.sin(jitter_angle)
        step_dx = (dx * cos_j - dy * sin_j) * step_len
        step_dy = (dx * sin_j + dy * cos_j) * step_len

        curr_x = int(curr_x + step_dx)
        curr_y = int(curr_y + step_dy)

        # Bounds check
        if 0 <= curr_x < img.shape[1] and 0 <= curr_y < img.shape[0]:
            pts.append((curr_x, curr_y))
        else:
            break

    # Draw main crack
    for i in range(len(pts) - 1):
        pt1, pt2 = pts[i], pts[i + 1]
        cv2.line(img, pt1, pt2, (25, 25, 28), thickness=thickness)

    # Add occasional branch fissures
    for i in range(5, len(pts) - 5, 12):
        bx, by = pts[i]
        b_pts = [(bx, by)]
        b_dx, b_dy = -dy, dx  # Perpendicular branch
        for _ in range(15):
            bx += int(b_dx * rng.uniform(4, 9) + rng.uniform(-3, 3))
            by += int(b_dy * rng.uniform(4, 9) + rng.uniform(-3, 3))
            if 0 <= bx < img.shape[1] and 0 <= by < img.shape[0]:
                b_pts.append((bx, by))
        for j in range(len(b_pts) - 1):
            cv2.line(img, b_pts[j], b_pts[j + 1], (30, 30, 32), thickness=max(1, thickness - 1))

    return img


def create_all_samples():
    output_dir = Path(__file__).resolve().parent / "data" / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standard Sample Road (Multiple potholes + cracks) -> Moderate/High Severity
    road1 = generate_asphalt_texture(1280, 720, base_val=118)
    road1 = add_pothole(road1, (480, 480), (75, 45), angle=15, seed=10)
    road1 = add_pothole(road1, (780, 390), (55, 35), angle=-20, seed=20)
    road1 = add_pothole(road1, (340, 290), (40, 25), angle=30, seed=30)
    road1 = add_crack(road1, (520, 450), steps=40, main_dir=(1.2, 0.8), thickness=3, seed=40)
    road1 = add_crack(road1, (680, 520), steps=35, main_dir=(-0.8, 1.0), thickness=2, seed=50)

    sample_road_path = output_dir / "sample_road.jpg"
    cv2.imwrite(str(sample_road_path), road1)
    print(f"Generated: {sample_road_path}")

    # 2. Pristine/Clean Road (Negative control) -> Zero/Low Severity
    road_clean = generate_asphalt_texture(1280, 720, base_val=125)
    clean_road_path = output_dir / "clean_road.jpg"
    cv2.imwrite(str(clean_road_path), road_clean)
    print(f"Generated: {clean_road_path}")

    # 3. Severe Hazard Road (Large craters + severe fatigue cracking) -> HIGH Severity
    road_severe = generate_asphalt_texture(1280, 720, base_val=110)
    road_severe = add_pothole(road_severe, (420, 500), (120, 70), angle=-10, seed=101)
    road_severe = add_pothole(road_severe, (820, 420), (95, 60), angle=25, seed=102)
    road_severe = add_pothole(road_severe, (310, 360), (65, 40), angle=-15, seed=103)
    road_severe = add_pothole(road_severe, (640, 260), (50, 30), angle=40, seed=104)
    road_severe = add_crack(road_severe, (350, 480), steps=55, main_dir=(1.0, 0.3), thickness=4, seed=105)
    road_severe = add_crack(road_severe, (700, 380), steps=45, main_dir=(-0.5, 1.2), thickness=3, seed=106)
    road_severe = add_crack(road_severe, (500, 250), steps=35, main_dir=(0.2, 1.0), thickness=3, seed=107)

    severe_road_path = output_dir / "severe_road.jpg"
    cv2.imwrite(str(severe_road_path), road_severe)
    print(f"Generated: {severe_road_path}")


if __name__ == "__main__":
    create_all_samples()
