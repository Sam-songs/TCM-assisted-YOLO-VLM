# -*- coding: utf-8 -*-
"""Command-line YOLO + Qwen analysis with an original/detection/English report image.

Run with --help for options. --report-json renders saved results without API calls.
The English assessment is a program-generated summary of prediction records.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont



def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pil_from_bgr(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def fit_image_to_box(image, size, bg=(246, 248, 250)):
    """Letterbox a PIL image into size."""
    box_w, box_h = size
    image = image.convert("RGB")
    scale = min(box_w / image.width, box_h / image.height)
    new_w = max(1, int(image.width * scale))
    new_h = max(1, int(image.height * scale))
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (box_w, box_h), bg)
    x = (box_w - new_w) // 2
    y = (box_h - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


def get_font(size=26, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_text_by_pixels(draw, text, font, max_width):
    lines = []
    for paragraph in str(text).splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue

        words = paragraph.split()
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            bbox = draw.textbbox((0, 0), trial, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)
    return lines


def draw_rounded_rect(draw, xy, radius, fill, outline=None, width=1):
    try:
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    except Exception:
        draw.rectangle(xy, fill=fill, outline=outline, width=width)


def create_text_panel(english_text, width, height):
    panel = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    title_font = get_font(28, bold=True)
    section_font = get_font(22, bold=True)
    body_font = get_font(20, bold=False)
    small_font = get_font(16, bold=False)

    draw.rectangle((0, 0, width, height), fill=(255, 255, 255))
    draw_rounded_rect(draw, (24, 24, width - 24, height - 24), 10, (248, 250, 252), (220, 226, 232), 2)
    draw.text((44, 42), "English Auxiliary Result", fill=(20, 27, 38), font=title_font)
    draw.line((44, 84, width - 44, 84), fill=(210, 218, 226), width=2)

    y = 108
    max_width = width - 88
    sections = []
    current_title = None
    current_lines = []

    for raw in english_text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line.strip("[]")
            current_lines = []
        else:
            current_lines.append(line)
    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines).strip()))
    if not sections:
        sections = [("Summary", english_text)]

    for title, body in sections:
        if y > height - 78:
            break

        draw.text((44, y), title, fill=(27, 80, 138), font=section_font)
        y += 32

        wrapped = wrap_text_by_pixels(draw, body, body_font, max_width)
        for line in wrapped:
            if y > height - 56:
                draw.text((44, y), "...", fill=(82, 91, 106), font=body_font)
                y += 24
                break
            draw.text((44, y), line, fill=(45, 55, 72), font=body_font)
            y += 28
        y += 14

    footer = "Visual AI assistance only. Not a medical diagnosis or treatment recommendation."
    draw.text((44, height - 46), footer, fill=(110, 118, 130), font=small_font)
    return panel


def create_three_column_report(original_bgr, detection_bgr, english_text, output_path):
    panel_w = 640
    panel_h = 760
    title_h = 54
    image_h = panel_h - title_h

    title_font = get_font(24, bold=True)
    canvas = Image.new("RGB", (panel_w * 3, panel_h), (236, 240, 244))
    draw = ImageDraw.Draw(canvas)

    def image_panel(img_bgr, title, x0):
        panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
        pdraw = ImageDraw.Draw(panel)
        pdraw.rectangle((0, 0, panel_w, panel_h), fill=(255, 255, 255))
        pdraw.rectangle((0, 0, panel_w, title_h), fill=(20, 27, 38))
        pdraw.text((22, 14), title, fill=(255, 255, 255), font=title_font)
        fitted = fit_image_to_box(pil_from_bgr(img_bgr), (panel_w, image_h), bg=(246, 248, 250))
        panel.paste(fitted, (0, title_h))
        canvas.paste(panel, (x0, 0))

    image_panel(original_bgr, "Original Image", 0)
    image_panel(detection_bgr, "YOLO Prediction", panel_w)
    text_panel = create_text_panel(english_text, panel_w, panel_h)
    canvas.paste(text_panel, (panel_w * 2, 0))

    draw.line((panel_w, 0, panel_w, panel_h), fill=(214, 220, 226), width=2)
    draw.line((panel_w * 2, 0, panel_w * 2, panel_h), fill=(214, 220, 226), width=2)

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    canvas.save(output_path, quality=94)
    return str(output_path)


def grade_from_records(fused_records):
    if not fused_records:
        return "No clear lesion", 0.0

    scores = [float(r.get("final_score", 0.0)) for r in fused_records]
    max_score = max(scores) if scores else 0.0
    lesion_count = len(fused_records)

    if max_score >= 0.85 or lesion_count >= 6:
        return "High visual priority", max_score
    if max_score >= 0.60 or lesion_count >= 2:
        return "Moderate visual priority", max_score
    if max_score > 0.0:
        return "Low visual priority", max_score
    return "No clear lesion", 0.0


def english_assessment(fused_records, image_level_top3, class_names, qwen_full_result):
    grade, max_score = grade_from_records(fused_records)
    lesion_count = len(fused_records)

    if image_level_top3:
        top = image_level_top3[0]
        top_name = top.get("class_name", "Unknown")
        top_prob = float(top.get("probability", 0.0))
    else:
        top_name = "No clear closed-set category"
        top_prob = 0.0

    top_lines = []
    for i, item in enumerate(image_level_top3[:3], 1):
        top_lines.append(
            f"{i}. {item.get('class_name', 'Unknown')} - {float(item.get('probability', 0.0)):.2f}"
        )
    top_text = "\n".join(top_lines) if top_lines else "No class-level ranking is available."

    evidence_lines = []
    for r in sorted(fused_records, key=lambda x: float(x.get("final_score", 0.0)), reverse=True)[:5]:
        evidence_lines.append(
            f"Box {r.get('box_index')}: {r.get('final_class_name')} "
            f"score {float(r.get('final_score', 0.0)):.2f}; "
            f"YOLO {r.get('yolo_class_name')} {float(r.get('yolo_confidence', 0.0)):.2f}; "
            f"fusion action: {r.get('fuse_action', 'keep_yolo')}."
        )
    evidence = "\n".join(evidence_lines) if evidence_lines else "No fused boxes were retained."

    qwen_quality = str(qwen_full_result.get("quality_comment", "")).strip()
    qwen_error = str(qwen_full_result.get("qwen_error", "")).strip()
    quality_note = qwen_quality or "Image quality was not specifically flagged by the full-image reviewer."
    if qwen_error:
        quality_note = f"{quality_note} Qwen full-image reviewer reported: {qwen_error[:180]}"

    return (
        "[Grade]\n"
        f"{grade}. Max fused confidence is {max_score:.2f}; retained lesion candidates: {lesion_count}.\n\n"
        "[Summary Inference]\n"
        f"The strongest closed-set visual support is for {top_name} with image-level probability {top_prob:.2f}. "
        "This is an algorithmic visual inference from YOLO detections, Qwen crop review, and Qwen full-image spatial support. "
        "It should be treated as decision support rather than a clinical conclusion.\n\n"
        "[Top Closed-Set Candidates]\n"
        f"{top_text}\n\n"
        "[Detection Evidence]\n"
        f"{evidence}\n\n"
        "[Image Quality]\n"
        f"{quality_note}\n\n"
        "[Limitations]\n"
        "The model cannot provide diagnosis, medication advice, or treatment planning. Please use clinical context and professional review for medical decisions."
    )


def draw_boxes_yolo(img, yolo_records):
    vis = img.copy()
    for r in yolo_records:
        x1, y1, x2, y2 = map(int, r["xyxy"])
        name = r["yolo_class_name"]
        conf = float(r["yolo_confidence"])
        label = f"YOLO {r['box_index']}: {name} {conf:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    return vis


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True, help='输入图片')
    parser.add_argument('--weights', type=Path, help='YOLO 权重；新推理时必需')
    parser.add_argument('--data', type=Path, help='data.yaml；新推理时必需')
    parser.add_argument('--out', type=Path, default=Path('runs/three_panel_report'), help='输出目录')
    parser.add_argument('--report-json', type=Path, help='从已有融合报告生成合成图，不运行模型或调用 API')
    parser.add_argument('--api-key-stdin', action='store_true', help='隐藏输入 API key；否则读取 DASHSCOPE_API_KEY')
    local_root = next((p for p in Path(__file__).resolve().parents
                       if (p/'ultralytics/__init__.py').is_file()), None)
    parser.add_argument('--yolo-root', type=Path, default=local_root, help='自定义 YOLO 仓库目录；本机自动查找')
    parser.add_argument('--qwen-model', default='qwen3.6-plus')
    parser.add_argument('--base-url', default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--iou', type=float, default=.70)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--sleep', type=float, default=.2)
    parser.add_argument('--max-qwen-boxes', type=int, default=10)
    parser.add_argument('--max-qwen-full-dets', type=int, default=10)
    parser.add_argument('--crop-margin', type=float, default=.18)
    parser.add_argument('--no-json-mode', action='store_true')
    parser.add_argument('--no-yolo-dedup', action='store_true')
    parser.add_argument('--dedup-iou-thr', type=float, default=.55)
    parser.add_argument('--dedup-contain-thr', type=float, default=.85)
    parser.add_argument('--dedup-area-ratio-thr', type=float, default=.65)
    parser.add_argument('--dedup-class-agnostic', action='store_true')
    parser.add_argument('--no-prefer-large-contained', action='store_true')
    parser.add_argument('--match-iou-thr', type=float, default=.30)
    parser.add_argument('--full-contain-thr', type=float, default=.70)
    parser.add_argument('--qwen-change-thr', type=float, default=.75)
    parser.add_argument('--qwen-full-support-thr', type=float, default=.50)
    return parser


def run_inference(args):
    """Run the existing non-GUI fusion program with an explicit argument list."""
    env = os.environ.copy()
    if args.api_key_stdin:
        key = getpass.getpass('Qwen API key（隐藏输入）: ') if sys.stdin.isatty() else sys.stdin.readline().strip()
    else:
        key = env.get('DASHSCOPE_API_KEY', '').strip()
    if not key:
        raise ValueError('请设置 DASHSCOPE_API_KEY，或添加 --api-key-stdin。')
    env['DASHSCOPE_API_KEY'] = key
    if args.yolo_root is not None:
        previous = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = str(args.yolo_root.resolve()) + (os.pathsep + previous if previous else '')
    env.setdefault('YOLO_CONFIG_DIR', str(args.out/'.ultralytics'))
    backend = Path(__file__).with_name('single_image_yolo_qwen_compare_dedup_conf_floor_contain.py')
    command = [sys.executable, str(backend)]
    values = ['weights', 'data', 'image', 'out', 'imgsz', 'conf', 'iou', 'qwen_model',
              'base_url', 'timeout', 'sleep', 'max_qwen_boxes', 'max_qwen_full_dets',
              'crop_margin', 'dedup_iou_thr', 'dedup_contain_thr', 'dedup_area_ratio_thr',
              'match_iou_thr', 'full_contain_thr', 'qwen_change_thr', 'qwen_full_support_thr']
    for name in values:
        command.extend(['--'+name.replace('_', '-'), str(getattr(args, name))])
    for name in ['no_json_mode', 'no_yolo_dedup', 'dedup_class_agnostic', 'no_prefer_large_contained']:
        if getattr(args, name):
            command.append('--'+name.replace('_', '-'))
    subprocess.run(command, env=env, check=True)
    return args.out/'reports/05_fused_final_report.json'


def render_report(image_path, report_path, out):
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if not isinstance(report.get('yolo_raw'), list) or not isinstance(report.get('fused_yolo_boxes'), list):
        raise ValueError('报告必须包含 yolo_raw 和 fused_yolo_boxes。')
    if Path(report.get('image', '')).name != image_path.name:
        raise ValueError('报告中的图片文件名与 --image 不一致。')
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f'无法读取图片：{image_path}')
    height, width = img.shape[:2]
    if (report.get('image_width'), report.get('image_height')) != (width, height):
        raise ValueError('报告中的图片尺寸与 --image 不一致。')
    english_text = english_assessment(
        report['fused_yolo_boxes'], report.get('image_level_top3_from_fused', []),
        [], report.get('qwen_full_image_detection', {}))
    detection = draw_boxes_yolo(img, report['yolo_raw'])
    composite = out/f'{image_path.stem}_one_row_report.jpg'
    create_three_column_report(img, detection, english_text, composite)
    text_path = out/f'{image_path.stem}_english_assessment.txt'
    text_path.write_text(english_text, encoding='utf-8')
    # Keep the source report intact; save a separate export with the full English text.
    report['english_assessment'] = english_text
    report['source_report_file'] = str(report_path)
    save_json(report, out/'reports'/f'{image_path.stem}_three_panel_report.json')
    print('三栏合成图：', composite, flush=True)
    print('完整英文结果：', text_path, flush=True)
    return composite


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.image = args.image.resolve()
    args.out = args.out.resolve()
    if not args.image.is_file():
        parser.error(f'输入图片不存在：{args.image}')
    if args.report_json is None:
        for name in ['weights', 'data']:
            path = getattr(args, name)
            if path is None or not path.is_file():
                parser.error(f'新推理需要有效的 --{name} 文件。')
            setattr(args, name, path.resolve())
        if args.yolo_root is not None and not (args.yolo_root/'ultralytics/__init__.py').is_file():
            parser.error('--yolo-root 应指向包含 ultralytics 包的仓库目录。')
        if args.imgsz <= 0 or args.timeout <= 0 or args.sleep < 0 or args.crop_margin < 0:
            parser.error('imgsz、timeout 必须为正；sleep、crop-margin 不能为负。')
        if args.max_qwen_boxes < 0 or args.max_qwen_full_dets < 0:
            parser.error('Qwen 框数不能为负。')
        for name in ['conf', 'iou', 'dedup_iou_thr', 'dedup_contain_thr', 'dedup_area_ratio_thr',
                     'match_iou_thr', 'full_contain_thr', 'qwen_change_thr', 'qwen_full_support_thr']:
            if not 0 <= getattr(args, name) <= 1:
                parser.error(f'--{name.replace("_", "-")} 必须在 0 到 1 之间。')
    elif not args.report_json.is_file():
        parser.error(f'报告不存在：{args.report_json}')
    try:
        report_path = args.report_json.resolve() if args.report_json else run_inference(args)
        render_report(args.image, report_path, args.out)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f'执行失败：{exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
