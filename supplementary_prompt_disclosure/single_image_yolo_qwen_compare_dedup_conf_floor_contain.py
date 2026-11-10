# -*- coding: utf-8 -*-
"""
单张皮肤病图像测试：YOLO原始检测 + Qwen整图闭集检测 + Qwen框内复判 + IoU重合对比 + 融合输出

功能：
1. 保存 YOLO 原始检测结果：图片、JSON、CSV。
2. 对 YOLO 原始框执行二次去重，删除小重复框/包含框，并保存过滤前后对比。
3. 保存 Qwen 整图闭集检测结果：图片、JSON、CSV。
4. 保存 Qwen 对过滤后 YOLO 框的框内复判结果：裁剪图、JSON、CSV。
5. 保存 YOLO + Qwen 融合后的最终结果：图片、JSON、CSV；融合时支持 IoU + 包含率双条件匹配。
6. 最终融合输出 final_score 强制不低于对应 YOLO 原始置信度。
7. 保存多图对比图：YOLO原始 / YOLO过滤 / Qwen整图 / 融合结果。

运行示例：
python single_image_yolo_qwen_compare.py \
  --weights runs/train/exp4/weights/best.pt \
  --data /home/sam/ultralytics-skin/CPD-10/data.yaml \
  --image /home/sam/ultralytics-skin/Atopic_dermatitis_1.jpg \
  --out runs/single_yolo_qwen_compare \
  --qwen-model qwen3.6-plus

环境变量：
export DASHSCOPE_API_KEY=你的API_KEY

注意：
该程序仅用于图像算法辅助分析与实验对比，不作为医学诊断依据。
"""

import os
import re
import cv2
import json
import yaml
import time
import base64
import argparse
import hashlib
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO
from openai import OpenAI


# =========================================================
# 1. 基础工具函数
# =========================================================

def load_class_names(data_yaml):
    """从 Ultralytics data.yaml 中读取类别名称。"""
    data_yaml = Path(data_yaml)
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", None)
    if names is None:
        raise RuntimeError(f"data.yaml 中没有 names 字段：{data_yaml}")

    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    else:
        class_names = list(names)

    if len(class_names) == 0:
        raise RuntimeError("类别列表为空，请检查 data.yaml。")

    return class_names


def image_to_base64_url(image_path):
    """将本地图像转为 OpenAI/DashScope 兼容的 data URL。"""
    image_path = Path(image_path)
    ext = image_path.suffix.lower()

    mime = "image/jpeg"
    if ext == ".png":
        mime = "image/png"
    elif ext == ".bmp":
        mime = "image/bmp"
    elif ext == ".webp":
        mime = "image/webp"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


def parse_json_from_text(text):
    """尽量从大模型输出中解析 JSON。"""
    if not isinstance(text, str):
        try:
            return dict(text)
        except Exception:
            text = json.dumps(text, ensure_ascii=False)

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # 兼容 ```json ... ```
    text2 = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    text2 = re.sub(r"```$", "", text2.strip()).strip()
    try:
        return json.loads(text2)
    except Exception:
        pass

    # 兜底提取最外层对象
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None


def file_md5(path):
    """计算文件 MD5，用于 Qwen 缓存。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def xyxy_iou(a, b):
    """计算两个 xyxy 框的 IoU。"""
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 1e-9:
        return 0.0
    return float(inter / union)




def xyxy_area(box):
    """计算 xyxy 框面积。"""
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def xyxy_intersection(a, b):
    """计算两个 xyxy 框的交集面积。"""
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def xyxy_containment_ratio(small_box, big_box):
    """
    返回 small_box 被 big_box 覆盖的面积比例。
    例如返回 0.90 表示 small_box 的 90% 面积落在 big_box 中。
    """
    area_small = xyxy_area(small_box)
    if area_small <= 1e-9:
        return 0.0
    return float(xyxy_intersection(small_box, big_box) / area_small)


def remove_duplicate_yolo_records(
    yolo_records,
    iou_thr=0.55,
    contain_thr=0.85,
    area_ratio_thr=0.65,
    class_agnostic=False,
    prefer_large_when_contained=True,
    replace_large_conf_ratio=0.75,
):
    """
    YOLO 原始框二次去重，重点处理“小框套大框”的重复检测。

    删除逻辑：
    1. 同类框 IoU >= iou_thr，认为是重复框，只保留高置信度框。
    2. 小框大部分被大框包含，并且面积明显更小，删除小框。
    3. 如果先遇到小框、后遇到较大框，且大框置信度不低于小框的一定比例，允许用大框替换小框。

    返回：
    kept_records: 过滤后保留框
    removed_records: 被删除/替换框，包含 duplicate_reason、duplicate_with_box_index 等字段
    """
    records = list(yolo_records or [])
    if len(records) == 0:
        return [], []

    records = sorted(records, key=lambda x: float(x.get("yolo_confidence", 0.0)), reverse=True)

    kept = []
    removed = []

    for cand in records:
        cand = dict(cand)
        cand_box = cand["xyxy"]
        cand_cls = int(cand["yolo_class_id"])
        cand_area = xyxy_area(cand_box)
        cand_conf = float(cand.get("yolo_confidence", 0.0))

        should_drop = False
        replace_index = None
        drop_info = {}

        for k, keep in enumerate(kept):
            keep_box = keep["xyxy"]
            keep_cls = int(keep["yolo_class_id"])
            keep_area = xyxy_area(keep_box)
            keep_conf = float(keep.get("yolo_confidence", 0.0))

            if (not class_agnostic) and cand_cls != keep_cls:
                continue

            iou = xyxy_iou(cand_box, keep_box)
            cand_in_keep = xyxy_containment_ratio(cand_box, keep_box)
            keep_in_cand = xyxy_containment_ratio(keep_box, cand_box)

            # 1. 高 IoU 重复框：当前候选置信度更低，删除当前候选框。
            if iou >= iou_thr:
                should_drop = True
                drop_info = {
                    "duplicate_reason": "high_iou_duplicate",
                    "duplicate_with_box_index": keep.get("box_index", ""),
                    "duplicate_iou": float(iou),
                    "duplicate_containment": float(cand_in_keep),
                    "duplicate_area_ratio": float(cand_area / max(keep_area, 1e-9)),
                }
                break

            # 2. 当前候选是小框，且大部分被已保留的大框包含，删除当前候选框。
            if cand_area < keep_area * area_ratio_thr and cand_in_keep >= contain_thr:
                should_drop = True
                drop_info = {
                    "duplicate_reason": "small_box_contained_by_kept_box",
                    "duplicate_with_box_index": keep.get("box_index", ""),
                    "duplicate_iou": float(iou),
                    "duplicate_containment": float(cand_in_keep),
                    "duplicate_area_ratio": float(cand_area / max(keep_area, 1e-9)),
                }
                break

            # 3. 已保留的是小框，当前候选是更合理的大框，允许替换。
            if (
                prefer_large_when_contained
                and keep_area < cand_area * area_ratio_thr
                and keep_in_cand >= contain_thr
                and cand_conf >= keep_conf * replace_large_conf_ratio
            ):
                replace_index = k
                drop_info = {
                    "duplicate_reason": "kept_small_box_replaced_by_larger_candidate",
                    "duplicate_with_box_index": cand.get("box_index", ""),
                    "duplicate_iou": float(iou),
                    "duplicate_containment": float(keep_in_cand),
                    "duplicate_area_ratio": float(keep_area / max(cand_area, 1e-9)),
                }
                break

        if replace_index is not None:
            old = dict(kept[replace_index])
            old.update(drop_info)
            removed.append(old)
            kept[replace_index] = cand
            continue

        if should_drop:
            cand.update(drop_info)
            removed.append(cand)
        else:
            kept.append(cand)

    kept = sorted(kept, key=lambda x: float(x.get("yolo_confidence", 0.0)), reverse=True)

    # 保留原始 box_index，并重排新的 filtered_box_index / box_index，便于后续图上序号一致。
    for i, r in enumerate(kept):
        r.setdefault("raw_box_index", r.get("box_index", i))
        r["filtered_box_index"] = i
        r["box_index"] = i
        r["dedup_status"] = "kept"

    for i, r in enumerate(removed):
        r.setdefault("raw_box_index", r.get("box_index", i))
        r["removed_index"] = i
        r["dedup_status"] = "removed"

    return kept, removed

def safe_crop(img, xyxy, margin=0.18, max_side=1024):
    """根据 YOLO 框裁剪病灶区域，并适当扩大边缘。"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(float, xyxy)

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None

    x1 = max(0, int(x1 - bw * margin))
    y1 = max(0, int(y1 - bh * margin))
    x2 = min(w - 1, int(x2 + bw * margin))
    y2 = min(h - 1, int(y2 + bh * margin))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2].copy()

    ch, cw = crop.shape[:2]
    m = max(ch, cw)
    if m > max_side:
        scale = max_side / float(m)
        crop = cv2.resize(
            crop,
            (max(1, int(cw * scale)), max(1, int(ch * scale))),
            interpolation=cv2.INTER_AREA,
        )

    return crop


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(rows, path):
    path = Path(path)
    ensure_dir(path.parent)
    if rows is None:
        rows = []
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def class_list_text(class_names):
    return "\n".join([f"{i}: {name}" for i, name in enumerate(class_names)])


# =========================================================
# 2. Qwen 返回结果规范化
# =========================================================

def ensure_top3_valid(top3, class_names):
    """保证 Qwen 返回的 top3 格式稳定，类别ID合法，概率归一化。"""
    fixed = []

    if not isinstance(top3, list):
        top3 = []

    used = set()

    for item in top3:
        if not isinstance(item, dict):
            continue

        try:
            cid = int(item.get("class_id", -1))
        except Exception:
            cid = -1

        if cid < 0 or cid >= len(class_names):
            continue

        if cid in used:
            continue

        used.add(cid)

        try:
            prob = float(item.get("probability", 0.0))
        except Exception:
            prob = 0.0

        prob = max(0.0, min(1.0, prob))

        fixed.append({
            "class_id": cid,
            "class_name": class_names[cid],
            "probability": prob,
            "analysis": str(item.get("analysis", item.get("visual_reason", "")))[:300],
        })

    # 不足3类时补齐
    for cid, name in enumerate(class_names):
        if len(fixed) >= 3:
            break
        if cid not in used:
            fixed.append({
                "class_id": cid,
                "class_name": name,
                "probability": 0.0,
                "analysis": "候选类别补齐",
            })

    fixed = fixed[:3]

    s = sum(x["probability"] for x in fixed)
    if s > 1e-9:
        for x in fixed:
            x["probability"] = float(x["probability"] / s)
    else:
        for x in fixed:
            x["probability"] = 1.0 / max(1, len(fixed))

    fixed = sorted(fixed, key=lambda x: x["probability"], reverse=True)
    return fixed


def normalize_qwen_crop_result(data, class_names):
    """规范化 Qwen 对 YOLO 裁剪框的复判结果。"""
    if not isinstance(data, dict):
        data = {}

    is_lesion = bool(data.get("is_lesion", True))
    lesion_description = str(data.get("lesion_description", ""))[:500]
    uncertain = bool(data.get("uncertain", False))
    quality_comment = str(data.get("quality_comment", ""))[:300]

    top3 = ensure_top3_valid(data.get("top3_diseases", []), class_names)

    try:
        final_class_id = int(data.get("final_class_id", top3[0]["class_id"]))
    except Exception:
        final_class_id = top3[0]["class_id"]

    if final_class_id < 0 or final_class_id >= len(class_names):
        final_class_id = top3[0]["class_id"]

    try:
        final_probability = float(data.get("final_probability", top3[0]["probability"]))
    except Exception:
        final_probability = top3[0]["probability"]

    final_probability = max(0.0, min(1.0, final_probability))

    return {
        "is_lesion": is_lesion,
        "lesion_description": lesion_description,
        "top3_diseases": top3,
        "final_class_id": final_class_id,
        "final_class_name": class_names[final_class_id],
        "final_probability": final_probability,
        "uncertain": uncertain,
        "quality_comment": quality_comment,
        "medical_disclaimer": "该结果仅为图像算法辅助分析，不作为医学诊断依据。",
    }


def normalize_bbox_xyxy(bbox, image_w, image_h):
    """规范化 Qwen 返回的 bbox，要求 xyxy 像素坐标。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return None

    # 如果模型返回了 0~1 归一化坐标，做一次兼容转换
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= image_w
        x2 *= image_w
        y1 *= image_h
        y2 *= image_h

    # 修正可能的反向坐标
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = clamp(x1, 0, image_w - 1)
    y1 = clamp(y1, 0, image_h - 1)
    x2 = clamp(x2, 0, image_w - 1)
    y2 = clamp(y2, 0, image_h - 1)

    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    return [float(x1), float(y1), float(x2), float(y2)]


def normalize_qwen_full_result(data, class_names, image_w, image_h):
    """规范化 Qwen 对整图闭集检测的返回结果。"""
    if not isinstance(data, dict):
        data = {}

    image_has_lesion = bool(data.get("image_has_lesion", data.get("has_lesion", True)))
    quality_comment = str(data.get("quality_comment", ""))[:300]

    raw_dets = data.get("detections", [])
    if not isinstance(raw_dets, list):
        raw_dets = []

    dets = []
    for i, det in enumerate(raw_dets):
        if not isinstance(det, dict):
            continue

        bbox = normalize_bbox_xyxy(det.get("bbox", det.get("xyxy", None)), image_w, image_h)
        if bbox is None:
            continue

        try:
            cid = int(det.get("class_id", -1))
        except Exception:
            cid = -1

        if cid < 0 or cid >= len(class_names):
            continue

        try:
            prob = float(det.get("probability", det.get("confidence", 0.0)))
        except Exception:
            prob = 0.0

        prob = max(0.0, min(1.0, prob))

        dets.append({
            "qwen_det_index": len(dets),
            "bbox": bbox,
            "class_id": cid,
            "class_name": class_names[cid],
            "probability": prob,
            "visual_reason": str(det.get("visual_reason", det.get("analysis", "")))[:500],
        })

    dets = sorted(dets, key=lambda x: x["probability"], reverse=True)
    for i, det in enumerate(dets):
        det["qwen_det_index"] = i

    return {
        "image_has_lesion": image_has_lesion if len(dets) > 0 else False,
        "detections": dets,
        "quality_comment": quality_comment,
        "medical_disclaimer": "该结果仅为图像算法辅助分析，不作为医学诊断依据。",
    }


# =========================================================
# 3. Qwen 视觉分析模块
# =========================================================

class QwenDiseaseAnalyzer:
    def __init__(
        self,
        model_name="qwen3.6-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        cache_path="qwen_single_cache.json",
        sleep_sec=0.2,
        use_json_mode=True,
        timeout=None,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.cache_path = Path(cache_path)
        self.sleep_sec = sleep_sec
        self.use_json_mode = use_json_mode
        self.timeout = timeout

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "没有检测到环境变量 DASHSCOPE_API_KEY。\n"
                "请先执行：export DASHSCOPE_API_KEY=你的API_KEY"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}
        else:
            self.cache = {}

    def save_cache(self):
        ensure_dir(self.cache_path.parent)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _call_qwen_json(self, prompt, image_path):
        """调用 Qwen，并尽量返回 JSON 对象。"""
        image_url = image_to_base64_url(image_path)

        kwargs = dict(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=0.0,
        )

        if self.timeout is not None:
            kwargs["timeout"] = self.timeout

        if self.use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            completion = self.client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            data = parse_json_from_text(content)
            if data is None:
                raise RuntimeError("Qwen返回内容无法解析为JSON")
            return data, content, None
        except Exception as e1:
            # 如果 JSON Mode 不兼容，自动去掉 response_format 重试一次
            if self.use_json_mode:
                try:
                    kwargs.pop("response_format", None)
                    completion = self.client.chat.completions.create(**kwargs)
                    content = completion.choices[0].message.content
                    data = parse_json_from_text(content)
                    if data is None:
                        raise RuntimeError("Qwen普通模式返回内容无法解析为JSON")
                    return data, content, None
                except Exception as e2:
                    return None, "", f"Qwen调用失败:{str(e2)[:300]}"
            return None, "", f"Qwen调用失败:{str(e1)[:300]}"

    def analyze_crop(self, crop_path, class_names, yolo_cls, yolo_conf):
        """对 YOLO 裁剪框进行 Qwen 闭集复判。"""
        crop_path = Path(crop_path)
        crop_hash = file_md5(crop_path)

        yolo_cls = int(yolo_cls)
        yolo_conf = float(yolo_conf)

        cache_key = f"crop_{crop_hash}_{self.model_name}_{yolo_cls}_{round(yolo_conf, 4)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""
你是一个皮肤病图像视觉复核模型。当前输入图像是 YOLO 检测框裁剪出来的疑似病灶区域。

重要限制：
1. 你不能给出医学诊断、治疗建议、用药建议。
2. 你只能根据图像视觉特征，在给定数据集类别集合中做闭集视觉判断。
3. 不能新增类别，不能输出类别集合之外的疾病名称。
4. 请重点判断 YOLO 给出的类别是否与图像视觉特征一致。
5. 如果 YOLO 类别与图像一致，请将该类别作为 top1，并给出合理 probability。
6. 如果 YOLO 类别与图像不一致，请说明原因，并在闭集类别中选择更合理类别。
7. 如果图像不像有效皮肤病区域，请设置 is_lesion=false。
8. 如果图像模糊、遮挡、光照差或难以判断，请设置 uncertain=true。
9. top3_diseases 必须输出3个候选类别，并按照 probability 从高到低排序。
10. probability 为0到1之间的小数，三个候选类别概率之和尽量接近1。
11. 输出必须是严格JSON。

YOLO初始检测结果：
- yolo_class_id: {yolo_cls}
- yolo_class_name: {class_names[yolo_cls]}
- yolo_confidence: {yolo_conf:.4f}

允许的数据集类别集合如下：
{class_list_text(class_names)}

请只返回如下JSON格式：
{{
  "is_lesion": true,
  "lesion_description": "对当前病灶区域的视觉描述，例如颜色、形态、边界、分布、是否有红斑/丘疹/水疱/结痂/鳞屑等，不超过100字",
  "top3_diseases": [
    {{
      "class_id": 0,
      "class_name": "类别名称",
      "probability": 0.60,
      "analysis": "该类别的辅助视觉依据，不超过80字"
    }},
    {{
      "class_id": 1,
      "class_name": "类别名称",
      "probability": 0.25,
      "analysis": "该类别的辅助视觉依据，不超过80字"
    }},
    {{
      "class_id": 2,
      "class_name": "类别名称",
      "probability": 0.15,
      "analysis": "该类别的辅助视觉依据，不超过80字"
    }}
  ],
  "final_class_id": 0,
  "final_class_name": "类别名称",
  "final_probability": 0.60,
  "uncertain": false,
  "quality_comment": "图像质量评价，例如清晰/模糊/曝光不足/病灶区域较小",
  "medical_disclaimer": "该结果仅为图像算法辅助分析，不作为医学诊断依据"
}}
"""

        data, raw_content, err = self._call_qwen_json(prompt, crop_path)
        if err is not None:
            result = self._fallback_crop_result(class_names, yolo_cls, yolo_conf, err)
            result["raw_qwen_content"] = raw_content
            result["qwen_error"] = err
        else:
            result = normalize_qwen_crop_result(data, class_names)
            result["raw_qwen_content"] = raw_content
            result["qwen_error"] = ""

        self.cache[cache_key] = result
        self.save_cache()

        if self.sleep_sec > 0:
            time.sleep(self.sleep_sec)

        return result

    def analyze_full_image_boxes(self, image_path, class_names, image_w, image_h, max_detections=10):
        """让 Qwen 直接根据 yaml 类别对整图进行闭集候选框检测。"""
        image_path = Path(image_path)
        img_hash = file_md5(image_path)
        cache_key = f"full_boxes_{img_hash}_{self.model_name}_{image_w}x{image_h}_{max_detections}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""
你是一个皮肤病图像视觉检测复核模型。当前输入是完整皮肤图像。

任务：
请根据给定的闭集类别集合，在整张图像中找出疑似皮肤病区域，并输出候选框、类别和概率。

重要限制：
1. 你不能给出医学诊断、治疗建议、用药建议。
2. 你只能使用给定类别集合，不能新增类别，不能输出类别集合之外的疾病名称。
3. bbox 必须使用原图像素坐标，格式为 [x1, y1, x2, y2]。
4. 原图尺寸为 width={image_w}, height={image_h}。
5. x1,y1 是左上角，x2,y2 是右下角。
6. 如果没有明确病灶，请返回空 detections 列表。
7. 最多输出 {max_detections} 个候选框。
8. probability 为 0 到 1 之间的小数。
9. 输出必须是严格JSON。

允许的数据集类别集合如下：
{class_list_text(class_names)}

请只返回如下JSON格式：
{{
  "image_has_lesion": true,
  "detections": [
    {{
      "bbox": [100, 80, 320, 260],
      "class_id": 0,
      "class_name": "类别名称",
      "probability": 0.82,
      "visual_reason": "候选区域的视觉依据，不超过80字"
    }}
  ],
  "quality_comment": "整图质量评价，例如清晰/模糊/曝光不足/病灶区域较小",
  "medical_disclaimer": "该结果仅为图像算法辅助分析，不作为医学诊断依据"
}}
"""

        data, raw_content, err = self._call_qwen_json(prompt, image_path)
        if err is not None:
            result = {
                "image_has_lesion": False,
                "detections": [],
                "quality_comment": "Qwen整图检测失败",
                "medical_disclaimer": "该结果仅为图像算法辅助分析，不作为医学诊断依据。",
                "raw_qwen_content": raw_content,
                "qwen_error": err,
            }
        else:
            result = normalize_qwen_full_result(data, class_names, image_w, image_h)
            result["raw_qwen_content"] = raw_content
            result["qwen_error"] = ""

        self.cache[cache_key] = result
        self.save_cache()

        if self.sleep_sec > 0:
            time.sleep(self.sleep_sec)

        return result

    def _fallback_crop_result(self, class_names, yolo_cls, yolo_conf, reason):
        """Qwen 失败时，框内复判回退到 YOLO。"""
        yolo_cls = int(yolo_cls)
        yolo_conf = float(yolo_conf)
        top3 = [{
            "class_id": yolo_cls,
            "class_name": class_names[yolo_cls],
            "probability": 1.0,
            "analysis": "Qwen调用失败，保留YOLO类别",
        }]
        top3 = ensure_top3_valid(top3, class_names)
        return {
            "is_lesion": True,
            "lesion_description": reason,
            "top3_diseases": top3,
            "final_class_id": yolo_cls,
            "final_class_name": class_names[yolo_cls],
            "final_probability": yolo_conf,
            "uncertain": True,
            "quality_comment": "Qwen调用失败或解析失败",
            "medical_disclaimer": "该结果仅为图像算法辅助分析，不作为医学诊断依据。",
        }


# =========================================================
# 4. YOLO 与 Qwen 融合逻辑
# =========================================================

def find_best_qwen_full_match(yolo_xyxy, qwen_full_detections):
    """
    为一个 YOLO 框找到最合适的 Qwen 整图候选框。

    旧版本只按照 IoU 最大匹配，这会导致一个问题：
    - YOLO 框通常比较小、比较贴近病灶；
    - Qwen 整图框通常比较大；
    - 小框完全落在大框里时，IoU 仍可能很低，导致 Qwen full 的高分无法参与融合。

    因此这里同时计算：
    1. qwen_full_iou：YOLO 框与 Qwen 整图框的 IoU；
    2. yolo_in_qwen_full：YOLO 框有多少比例面积落在 Qwen 整图框内；
    3. qwen_full_in_yolo：Qwen 整图框有多少比例面积落在 YOLO 框内；
    4. support_score=max(IoU, yolo_in_qwen_full)，用于选择最有支持性的 Qwen 框。

    返回：
    best, best_iou, best_yolo_in_full, best_full_in_yolo, best_support_score
    """
    best = None
    best_iou = 0.0
    best_yolo_in_full = 0.0
    best_full_in_yolo = 0.0
    best_support_score = 0.0
    best_rank_tuple = (-1.0, -1.0, -1.0, -1.0)

    for det in qwen_full_detections:
        qwen_box = det.get("bbox", None)
        if qwen_box is None:
            continue

        iou = float(xyxy_iou(yolo_xyxy, qwen_box))
        yolo_in_full = float(xyxy_containment_ratio(yolo_xyxy, qwen_box))
        full_in_yolo = float(xyxy_containment_ratio(qwen_box, yolo_xyxy))
        support_score = float(max(iou, yolo_in_full))
        if support_score <= 1e-12:
            continue

        try:
            prob = float(det.get("probability", 0.0))
        except Exception:
            prob = 0.0

        # 先看 support_score，再看 yolo_in_full，再看 IoU，再看 Qwen 概率。
        # 这样当小 YOLO 框被大 Qwen 框包含时，不会因为 IoU 低而被忽略。
        rank_tuple = (support_score, yolo_in_full, iou, prob)
        if rank_tuple > best_rank_tuple:
            best_rank_tuple = rank_tuple
            best = det
            best_iou = iou
            best_yolo_in_full = yolo_in_full
            best_full_in_yolo = full_in_yolo
            best_support_score = support_score

    return (
        best,
        float(best_iou),
        float(best_yolo_in_full),
        float(best_full_in_yolo),
        float(best_support_score),
    )


def _make_fuse_result(final_cls, final_score, action, reason, class_names, yolo_conf):
    """
    统一构造融合结果。

    关键约束：最终输出 final_score 不能低于 YOLO 原始置信度。
    因此无论 Qwen 给出降低、不确定、拒识等判断，最终分数都会执行：
        final_score = max(final_score, yolo_conf)
    """
    yolo_conf = float(yolo_conf)
    raw_score = float(clamp(final_score, 0.0, 1.0))
    final_score = float(clamp(max(raw_score, yolo_conf), 0.0, 1.0))
    score_floor_applied = final_score > raw_score + 1e-12

    if score_floor_applied:
        reason = (
            f"{reason} 同时根据约束：最终检测置信度不得低于YOLO原始置信度，"
            f"因此将分数由 {raw_score:.4f} 提升/回退到 {final_score:.4f}。"
        )

    return {
        "final_class_id": int(final_cls),
        "final_class_name": class_names[int(final_cls)],
        "final_score": final_score,
        "score_before_yolo_floor": raw_score,
        "score_floor_applied": bool(score_floor_applied),
        "yolo_confidence_floor": float(clamp(yolo_conf, 0.0, 1.0)),
        "fuse_action": action,
        "fuse_reason": reason,
    }


def fuse_yolo_qwen_v2(
    yolo_cls,
    yolo_conf,
    qwen_crop_result,
    qwen_full_match,
    qwen_full_iou,
    qwen_full_yolo_containment,
    qwen_full_box_containment,
    class_names,
    iou_thr=0.30,
    full_contain_thr=0.70,
    change_thr=0.75,
    full_support_thr=0.50,
):
    """
    融合目标：最终仍以 YOLO 框为输出框，但类别和分数由 YOLO + Qwen crop + Qwen full boxes 共同决定。

    新增两项关键约束：
    1. 最终输出 final_score 一定不低于该框的 YOLO 原始置信度 yolo_conf。
    2. Qwen 整图框支持 YOLO 框时，不再只看 IoU；如果 YOLO 框有足够比例面积落在 Qwen 整图框内，
       也认为 Qwen full 支持该 YOLO 框。

    匹配条件：
        has_full_match = qwen_full_iou >= iou_thr or qwen_full_yolo_containment >= full_contain_thr

    这样可以解决：Qwen 整图框较大、YOLO 框较小，导致 IoU 偏低，但实际二者指向同一病灶的问题。
    """
    yolo_cls = int(yolo_cls)
    yolo_conf = float(yolo_conf)

    crop_is_lesion = bool(qwen_crop_result.get("is_lesion", True))
    crop_uncertain = bool(qwen_crop_result.get("uncertain", False))
    crop_cls = int(qwen_crop_result.get("final_class_id", yolo_cls))
    crop_prob = float(qwen_crop_result.get("final_probability", yolo_conf))

    qwen_full_iou = float(qwen_full_iou or 0.0)
    qwen_full_yolo_containment = float(qwen_full_yolo_containment or 0.0)
    qwen_full_box_containment = float(qwen_full_box_containment or 0.0)
    full_support_metric = float(max(qwen_full_iou, qwen_full_yolo_containment))

    has_full_match = (
        qwen_full_match is not None
        and (
            qwen_full_iou >= float(iou_thr)
            or qwen_full_yolo_containment >= float(full_contain_thr)
        )
    )

    full_cls = None
    full_prob = 0.0
    if has_full_match:
        full_cls = int(qwen_full_match.get("class_id", -1))
        full_prob = float(qwen_full_match.get("probability", 0.0))

    full_support_text = (
        f"Qwen整图匹配指标：IoU={qwen_full_iou:.3f}, "
        f"YOLO框被Qwen框包含率={qwen_full_yolo_containment:.3f}, "
        f"Qwen框被YOLO框包含率={qwen_full_box_containment:.3f}。"
    )

    matched_by_containment_only = (
        has_full_match
        and qwen_full_iou < float(iou_thr)
        and qwen_full_yolo_containment >= float(full_contain_thr)
    )

    # 1. Qwen crop 认为不是病灶。
    # 如果 Qwen 整图又强支持 YOLO 类别，则不降分，反而允许使用 full_prob 增强。
    if not crop_is_lesion:
        if has_full_match and full_cls == yolo_cls and full_prob >= full_support_thr:
            final_score = max(yolo_conf, full_prob)
            return _make_fuse_result(
                final_cls=yolo_cls,
                final_score=final_score,
                action="qwen_crop_reject_but_full_support_keep_yolo",
                reason=(
                    "Qwen框内复判认为该区域不像有效病灶，但Qwen整图候选框与YOLO类别一致并形成空间支持，"
                    "因此保留YOLO类别并采用YOLO/Qwen整图较高分数。" + full_support_text
                ),
                class_names=class_names,
                yolo_conf=yolo_conf,
            )

        final_score = yolo_conf * 0.10
        return _make_fuse_result(
            final_cls=yolo_cls,
            final_score=final_score,
            action="qwen_crop_reject_non_lesion_keep_yolo_score",
            reason="Qwen框内复判认为该区域不像有效病灶，算法记录该风险，但最终检测分数不允许低于YOLO原始置信度。",
            class_names=class_names,
            yolo_conf=yolo_conf,
        )

    # 2. Qwen crop 不确定，并且整图没有强支持。
    if crop_uncertain and crop_prob < 0.65 and not has_full_match:
        final_score = yolo_conf * 0.80
        return _make_fuse_result(
            final_cls=yolo_cls,
            final_score=final_score,
            action="qwen_crop_uncertain_keep_yolo_score",
            reason="Qwen框内复判不确定，且整图Qwen没有足够重合/包含支持，因此保守保留YOLO类别。",
            class_names=class_names,
            yolo_conf=yolo_conf,
        )

    # 3. Qwen crop 与 YOLO 一致。
    if crop_cls == yolo_cls:
        final_score = max(yolo_conf, crop_prob)
        action = "qwen_crop_agree_boost"
        reason = "Qwen框内复判与YOLO类别一致，使用二者较高概率作为最终分数。"

        if has_full_match:
            if full_cls == yolo_cls:
                # 重点修改：只要 Qwen full 通过 IoU 或包含率支持 YOLO，直接把 full_prob 纳入 max。
                # 这样 03 Qwen Full Image 的高分可以传递到最终结果。
                final_score = max(final_score, full_prob)
                final_score = max(
                    final_score,
                    0.50 * final_score + 0.30 * full_prob + 0.20 * full_support_metric,
                )
                final_score = min(1.0, final_score + 0.05)

                if matched_by_containment_only:
                    action = "qwen_crop_and_full_containment_agree_boost"
                    reason = (
                        "Qwen框内复判与YOLO类别一致；Qwen整图框虽然IoU偏低，但YOLO小框大部分落在Qwen大框内，"
                        "因此认为Qwen整图结果支持该YOLO框，并采用YOLO/Qwen框内/Qwen整图三者较高分数进行增强。"
                        + full_support_text
                    )
                else:
                    action = "qwen_crop_and_full_agree_boost"
                    reason = (
                        "Qwen框内复判、Qwen整图候选框与YOLO均一致，采用YOLO/Qwen框内/Qwen整图三者较高分数进行增强。"
                        + full_support_text
                    )
            else:
                action = "qwen_crop_agree_full_class_conflict"
                reason = (
                    "Qwen框内复判与YOLO一致，但Qwen整图匹配框类别不同，因此不采用Qwen整图类别和分数。"
                    + full_support_text
                )
        else:
            # crop 支持，但整图没有检测到对应框，旧逻辑会轻微降分。
            # 这里保留保守计算结果，但由 _make_fuse_result 保证不低于 YOLO。
            final_score = final_score * 0.95
            action = "qwen_crop_agree_no_full_support_keep_yolo_floor"
            reason = "Qwen框内复判与YOLO一致，但Qwen整图未给出足够IoU或包含率支持，原融合分数轻微保守处理。"

        return _make_fuse_result(
            final_cls=yolo_cls,
            final_score=final_score,
            action=action,
            reason=reason,
            class_names=class_names,
            yolo_conf=yolo_conf,
        )

    # 4. Qwen crop 与 YOLO 不一致，但 Qwen crop 高置信。
    if crop_cls != yolo_cls and crop_prob >= change_thr:
        final_cls = crop_cls
        final_score = max(yolo_conf, crop_prob)
        action = "qwen_crop_change_class"
        reason = "Qwen框内复判与YOLO类别不同，但Qwen概率较高，因此将最终类别修正为Qwen框内类别。"

        if has_full_match:
            if full_cls == crop_cls and full_prob >= full_support_thr:
                final_score = max(final_score, full_prob)
                final_score = max(
                    final_score,
                    0.55 * crop_prob + 0.30 * full_prob + 0.15 * full_support_metric,
                )
                final_score = min(1.0, final_score + 0.05)
                if matched_by_containment_only:
                    action = "qwen_crop_change_class_full_containment_support"
                    reason = (
                        "Qwen框内复判改类，且Qwen整图框通过包含率支持该类别，因此采用Qwen修正类别并增强分数。"
                        + full_support_text
                    )
                else:
                    action = "qwen_crop_change_class_full_support"
                    reason = (
                        "Qwen框内复判改类，且Qwen整图重合框也支持该类别，因此采用Qwen修正类别并增强分数。"
                        + full_support_text
                    )
            elif full_cls == yolo_cls:
                final_score = max(yolo_conf, 0.85 * crop_prob)
                action = "qwen_crop_change_class_full_conflict"
                reason = (
                    "Qwen框内复判倾向改类，但Qwen整图匹配框支持YOLO类别，因此采用Qwen框内类别但不额外增强。"
                    + full_support_text
                )

        return _make_fuse_result(
            final_cls=final_cls,
            final_score=final_score,
            action=action,
            reason=reason,
            class_names=class_names,
            yolo_conf=yolo_conf,
        )

    # 5. Qwen full 强支持 YOLO。
    if has_full_match and full_cls == yolo_cls and full_prob >= full_support_thr:
        final_score = max(
            yolo_conf,
            full_prob,
            0.60 * yolo_conf + 0.25 * full_prob + 0.15 * full_support_metric,
        )
        if matched_by_containment_only:
            action = "qwen_full_containment_support_keep_yolo"
            reason = (
                "Qwen框内复判未达到改类阈值，但Qwen整图框通过包含率支持YOLO类别，因此保留YOLO类别并采用较高分数。"
                + full_support_text
            )
        else:
            action = "qwen_full_support_keep_yolo"
            reason = (
                "Qwen框内复判未达到改类阈值，但Qwen整图重合框支持YOLO类别，因此保留YOLO类别并采用较高分数。"
                + full_support_text
            )

        return _make_fuse_result(
            final_cls=yolo_cls,
            final_score=final_score,
            action=action,
            reason=reason,
            class_names=class_names,
            yolo_conf=yolo_conf,
        )

    # 6. 默认保留 YOLO。
    return _make_fuse_result(
        final_cls=yolo_cls,
        final_score=yolo_conf,
        action="keep_yolo",
        reason="Qwen证据不足以增强或修正类别，保留YOLO原始结果。",
        class_names=class_names,
        yolo_conf=yolo_conf,
    )


def aggregate_image_top3_from_fused(fused_records, class_names):
    """根据融合后的框分数汇总整图 Top-3 类别。"""
    scores = {i: 0.0 for i in range(len(class_names))}
    for item in fused_records:
        cid = int(item.get("final_class_id", -1))
        score = float(item.get("final_score", 0.0))
        if 0 <= cid < len(class_names):
            scores[cid] += score

    total = sum(scores.values())
    if total > 1e-9:
        for k in scores:
            scores[k] /= total

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return [
        {
            "class_id": cid,
            "class_name": class_names[cid],
            "probability": float(prob),
        }
        for cid, prob in ranked
    ]


# =========================================================
# 5. 结果展开与可视化保存
# =========================================================

def flatten_top3(prefix, top3):
    row = {}
    if not isinstance(top3, list):
        top3 = []
    for i in range(3):
        if i < len(top3):
            item = top3[i]
            row[f"{prefix}_top{i+1}_class_id"] = item.get("class_id", "")
            row[f"{prefix}_top{i+1}_class_name"] = item.get("class_name", "")
            row[f"{prefix}_top{i+1}_probability"] = item.get("probability", "")
            row[f"{prefix}_top{i+1}_analysis"] = item.get("analysis", "")
        else:
            row[f"{prefix}_top{i+1}_class_id"] = ""
            row[f"{prefix}_top{i+1}_class_name"] = ""
            row[f"{prefix}_top{i+1}_probability"] = ""
            row[f"{prefix}_top{i+1}_analysis"] = ""
    return row


def yolo_records_to_csv_rows(records):
    rows = []
    for r in records:
        xyxy = r["xyxy"]
        rows.append({
            "image": r.get("image", ""),
            "box_index": r.get("box_index", ""),
            "raw_box_index": r.get("raw_box_index", ""),
            "filtered_box_index": r.get("filtered_box_index", ""),
            "dedup_status": r.get("dedup_status", ""),
            "x1": xyxy[0],
            "y1": xyxy[1],
            "x2": xyxy[2],
            "y2": xyxy[3],
            "yolo_class_id": r.get("yolo_class_id", ""),
            "yolo_class_name": r.get("yolo_class_name", ""),
            "yolo_confidence": r.get("yolo_confidence", ""),
        })
    return rows



def removed_yolo_records_to_csv_rows(removed_records):
    rows = []
    for r in removed_records:
        xyxy = r["xyxy"]
        rows.append({
            "raw_box_index": r.get("raw_box_index", r.get("box_index", "")),
            "removed_index": r.get("removed_index", ""),
            "x1": xyxy[0],
            "y1": xyxy[1],
            "x2": xyxy[2],
            "y2": xyxy[3],
            "yolo_class_id": r.get("yolo_class_id", ""),
            "yolo_class_name": r.get("yolo_class_name", ""),
            "yolo_confidence": r.get("yolo_confidence", ""),
            "duplicate_reason": r.get("duplicate_reason", ""),
            "duplicate_with_box_index": r.get("duplicate_with_box_index", ""),
            "duplicate_iou": r.get("duplicate_iou", ""),
            "duplicate_containment": r.get("duplicate_containment", ""),
            "duplicate_area_ratio": r.get("duplicate_area_ratio", ""),
        })
    return rows


def qwen_full_to_csv_rows(qwen_full_result, image_path):
    rows = []
    for det in qwen_full_result.get("detections", []):
        bbox = det["bbox"]
        rows.append({
            "image": str(image_path),
            "qwen_det_index": det.get("qwen_det_index", ""),
            "x1": bbox[0],
            "y1": bbox[1],
            "x2": bbox[2],
            "y2": bbox[3],
            "qwen_class_id": det.get("class_id", ""),
            "qwen_class_name": det.get("class_name", ""),
            "qwen_probability": det.get("probability", ""),
            "visual_reason": det.get("visual_reason", ""),
        })
    return rows


def crop_records_to_csv_rows(crop_records):
    rows = []
    for r in crop_records:
        q = r.get("qwen_crop_analysis", {})
        xyxy = r["xyxy"]
        row = {
            "image": r.get("image", ""),
            "box_index": r.get("box_index", ""),
            "crop_path": r.get("crop_path", ""),
            "x1": xyxy[0],
            "y1": xyxy[1],
            "x2": xyxy[2],
            "y2": xyxy[3],
            "yolo_class_id": r.get("yolo_class_id", ""),
            "yolo_class_name": r.get("yolo_class_name", ""),
            "yolo_confidence": r.get("yolo_confidence", ""),
            "qwen_is_lesion": q.get("is_lesion", ""),
            "qwen_final_class_id": q.get("final_class_id", ""),
            "qwen_final_class_name": q.get("final_class_name", ""),
            "qwen_final_probability": q.get("final_probability", ""),
            "qwen_uncertain": q.get("uncertain", ""),
            "lesion_description": q.get("lesion_description", ""),
            "quality_comment": q.get("quality_comment", ""),
            "qwen_error": q.get("qwen_error", ""),
        }
        row.update(flatten_top3("qwen_crop", q.get("top3_diseases", [])))
        rows.append(row)
    return rows


def fused_records_to_csv_rows(fused_records):
    rows = []
    for r in fused_records:
        xyxy = r["xyxy"]
        row = {
            "image": r.get("image", ""),
            "box_index": r.get("box_index", ""),
            "crop_path": r.get("crop_path", ""),
            "x1": xyxy[0],
            "y1": xyxy[1],
            "x2": xyxy[2],
            "y2": xyxy[3],
            "yolo_class_id": r.get("yolo_class_id", ""),
            "yolo_class_name": r.get("yolo_class_name", ""),
            "yolo_confidence": r.get("yolo_confidence", ""),
            "qwen_crop_class_id": r.get("qwen_crop_class_id", ""),
            "qwen_crop_class_name": r.get("qwen_crop_class_name", ""),
            "qwen_crop_probability": r.get("qwen_crop_probability", ""),
            "qwen_crop_uncertain": r.get("qwen_crop_uncertain", ""),
            "qwen_crop_is_lesion": r.get("qwen_crop_is_lesion", ""),
            "qwen_full_match_index": r.get("qwen_full_match_index", ""),
            "qwen_full_iou": r.get("qwen_full_iou", ""),
            "qwen_full_yolo_containment": r.get("qwen_full_yolo_containment", ""),
            "qwen_full_box_containment": r.get("qwen_full_box_containment", ""),
            "qwen_full_support_score": r.get("qwen_full_support_score", ""),
            "qwen_full_class_id": r.get("qwen_full_class_id", ""),
            "qwen_full_class_name": r.get("qwen_full_class_name", ""),
            "qwen_full_probability": r.get("qwen_full_probability", ""),
            "final_class_id": r.get("final_class_id", ""),
            "final_class_name": r.get("final_class_name", ""),
            "final_score": r.get("final_score", ""),
            "score_before_yolo_floor": r.get("score_before_yolo_floor", ""),
            "score_floor_applied": r.get("score_floor_applied", ""),
            "yolo_confidence_floor": r.get("yolo_confidence_floor", ""),
            "fuse_action": r.get("fuse_action", ""),
            "fuse_reason": r.get("fuse_reason", ""),
        }
        rows.append(row)
    return rows


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


def draw_boxes_qwen_full(img, qwen_full_result):
    vis = img.copy()
    for det in qwen_full_result.get("detections", []):
        x1, y1, x2, y2 = map(int, det["bbox"])
        name = det["class_name"]
        prob = float(det["probability"])
        label = f"Qwen {det['qwen_det_index']}: {name} {prob:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 120, 0), 2)
        cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 0), 2)
    return vis


def draw_boxes_fused(img, fused_records):
    vis = img.copy()
    for r in fused_records:
        x1, y1, x2, y2 = map(int, r["xyxy"])
        final_name = r["final_class_name"]
        final_score = float(r["final_score"])
        action = str(r["fuse_action"])
        label = f"Final {r['box_index']}: {final_name} {final_score:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 120), 2)
        cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)
        cv2.putText(vis, action, (x1, min(vis.shape[0] - 5, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
    return vis


def add_title(img, title):
    vis = img.copy()
    h, w = vis.shape[:2]
    bar_h = 36
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[bar_h:, :] = vis
    cv2.rectangle(canvas, (0, 0), (w, bar_h), (30, 30, 30), -1)
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return canvas


def make_compare_canvas(*items, max_width_each=640):
    """
    生成横向对比图。
    用法：make_compare_canvas((img1, "title1"), (img2, "title2"), ...)
    兼容旧用法：make_compare_canvas(yolo_img, qwen_img, fused_img)
    """
    if len(items) == 3 and not isinstance(items[0], (tuple, list)):
        items = [
            (items[0], "YOLO RAW"),
            (items[1], "QWEN FULL IMAGE"),
            (items[2], "FUSED RESULT"),
        ]

    imgs = []
    for im, title in items:
        h, w = im.shape[:2]
        if w > max_width_each:
            scale = max_width_each / float(w)
            im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        imgs.append(add_title(im, title))

    if len(imgs) == 0:
        return np.zeros((100, 300, 3), dtype=np.uint8)

    max_h = max(im.shape[0] for im in imgs)
    padded = []
    for im in imgs:
        h, w = im.shape[:2]
        if h < max_h:
            pad = np.zeros((max_h - h, w, 3), dtype=np.uint8)
            im = np.vstack([im, pad])
        padded.append(im)
    return np.hstack(padded)

# =========================================================
# 6. 主流程
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, required=True, help="YOLO权重路径，例如 best.pt")
    parser.add_argument("--data", type=str, required=True, help="data.yaml路径")
    parser.add_argument("--image", type=str, required=True, help="待分析的单张图片路径")
    parser.add_argument("--out", type=str, default="runs/single_yolo_qwen_compare", help="输出目录")

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)

    parser.add_argument("--no-yolo-dedup", action="store_true", help="关闭YOLO二次去重；默认开启")
    parser.add_argument("--dedup-iou-thr", type=float, default=0.55, help="YOLO二次去重的IoU阈值")
    parser.add_argument("--dedup-contain-thr", type=float, default=0.85, help="小框被大框包含比例阈值")
    parser.add_argument("--dedup-area-ratio-thr", type=float, default=0.65, help="小框/大框面积比例阈值")
    parser.add_argument("--dedup-class-agnostic", action="store_true", help="不同类别之间也执行重复框过滤；默认只过滤同类别")
    parser.add_argument("--no-prefer-large-contained", action="store_true", help="关闭大框替换被包含小框的策略")

    parser.add_argument("--qwen-model", type=str, default="qwen3.6-plus")
    parser.add_argument("--base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--no-json-mode", action="store_true", help="如果模型不支持JSON Mode，添加该参数")
    parser.add_argument("--timeout", type=float, default=None, help="Qwen API超时时间，可选")

    parser.add_argument("--max-qwen-boxes", type=int, default=10, help="最多对多少个YOLO框调用Qwen框内复判")
    parser.add_argument("--max-qwen-full-dets", type=int, default=10, help="Qwen整图最多输出多少个候选框")
    parser.add_argument("--crop-margin", type=float, default=0.18)

    parser.add_argument("--match-iou-thr", type=float, default=0.30, help="Qwen整图框与YOLO框的IoU重合阈值")
    parser.add_argument("--full-contain-thr", type=float, default=0.70, help="Qwen整图框支持YOLO框的包含率阈值；当YOLO框有该比例面积落在Qwen框内时，即使IoU较低也认为匹配")
    parser.add_argument("--qwen-change-thr", type=float, default=0.75, help="Qwen框内复判改类阈值")
    parser.add_argument("--qwen-full-support-thr", type=float, default=0.50, help="Qwen整图候选框支持类别的概率阈值")

    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = ensure_dir(args.out)
    crop_dir = ensure_dir(out_dir / "crops")
    report_dir = ensure_dir(out_dir / "reports")
    vis_dir = ensure_dir(out_dir / "visualizations")

    class_names = load_class_names(args.data)

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"图片读取失败：{image_path}")
    image_h, image_w = img.shape[:2]

    print("=" * 100)
    print("Single Image YOLO + Qwen Compare Test")
    print("image:", image_path)
    print("weights:", args.weights)
    print("data:", args.data)
    print("classes:", class_names)
    print("output:", out_dir)
    print("=" * 100)

    # -----------------------------------------------------
    # 6.1 YOLO 原始检测
    # -----------------------------------------------------
    print("\n[1/5] Running YOLO detection...")
    model = YOLO(args.weights)
    result = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )[0]

    yolo_records = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box_idx, box in enumerate(result.boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(float).tolist()
            yolo_cls = int(box.cls[0].cpu().item())
            yolo_conf = float(box.conf[0].cpu().item())
            yolo_records.append({
                "image": str(image_path),
                "box_index": box_idx,
                "xyxy": xyxy,
                "yolo_class_id": yolo_cls,
                "yolo_class_name": class_names[yolo_cls],
                "yolo_confidence": yolo_conf,
            })

    yolo_records = sorted(yolo_records, key=lambda x: x["yolo_confidence"], reverse=True)
    # 重排 box_index，方便和图上序号一致
    for i, r in enumerate(yolo_records):
        r["box_index"] = i

    save_json({"image": str(image_path), "detections": yolo_records}, report_dir / "01_yolo_raw.json")
    save_csv(yolo_records_to_csv_rows(yolo_records), report_dir / "01_yolo_raw.csv")

    yolo_vis = draw_boxes_yolo(img, yolo_records)
    yolo_vis_path = vis_dir / "01_yolo_raw.jpg"
    cv2.imwrite(str(yolo_vis_path), yolo_vis)

    print(f"YOLO raw boxes: {len(yolo_records)}")
    print("Saved YOLO raw:", yolo_vis_path)

    # -----------------------------------------------------
    # 6.2 YOLO 二次去重：删除小重复框/包含框
    # -----------------------------------------------------
    print("\n[2/6] Filtering duplicate YOLO boxes...")
    if args.no_yolo_dedup:
        yolo_filtered_records = [dict(r) for r in yolo_records]
        yolo_removed_records = []
        for i, r in enumerate(yolo_filtered_records):
            r.setdefault("raw_box_index", r.get("box_index", i))
            r["filtered_box_index"] = i
            r["box_index"] = i
            r["dedup_status"] = "kept_no_dedup"
    else:
        yolo_filtered_records, yolo_removed_records = remove_duplicate_yolo_records(
            yolo_records,
            iou_thr=args.dedup_iou_thr,
            contain_thr=args.dedup_contain_thr,
            area_ratio_thr=args.dedup_area_ratio_thr,
            class_agnostic=args.dedup_class_agnostic,
            prefer_large_when_contained=not args.no_prefer_large_contained,
        )

    save_json({
        "image": str(image_path),
        "dedup_enabled": not args.no_yolo_dedup,
        "dedup_params": {
            "iou_thr": args.dedup_iou_thr,
            "contain_thr": args.dedup_contain_thr,
            "area_ratio_thr": args.dedup_area_ratio_thr,
            "class_agnostic": args.dedup_class_agnostic,
            "prefer_large_when_contained": not args.no_prefer_large_contained,
        },
        "raw_count": len(yolo_records),
        "filtered_count": len(yolo_filtered_records),
        "removed_count": len(yolo_removed_records),
        "detections": yolo_filtered_records,
        "removed_detections": yolo_removed_records,
    }, report_dir / "02_yolo_filtered.json")
    save_csv(yolo_records_to_csv_rows(yolo_filtered_records), report_dir / "02_yolo_filtered.csv")
    save_csv(removed_yolo_records_to_csv_rows(yolo_removed_records), report_dir / "02_yolo_removed_duplicates.csv")

    yolo_filtered_vis = draw_boxes_yolo(img, yolo_filtered_records)
    yolo_filtered_vis_path = vis_dir / "02_yolo_filtered.jpg"
    cv2.imwrite(str(yolo_filtered_vis_path), yolo_filtered_vis)

    print(f"YOLO filtered boxes: {len(yolo_filtered_records)}")
    print(f"YOLO removed duplicate boxes: {len(yolo_removed_records)}")
    print("Saved YOLO filtered:", yolo_filtered_vis_path)

    # -----------------------------------------------------
    # 6.3 初始化 Qwen
    # -----------------------------------------------------
    qwen = QwenDiseaseAnalyzer(
        model_name=args.qwen_model,
        base_url=args.base_url,
        cache_path=out_dir / "qwen_cache.json",
        sleep_sec=args.sleep,
        use_json_mode=not args.no_json_mode,
        timeout=args.timeout,
    )

    # -----------------------------------------------------
    # 6.3 Qwen 整图闭集检测
    # -----------------------------------------------------
    print("\n[3/6] Running Qwen full-image closed-set detection...")
    qwen_full_result = qwen.analyze_full_image_boxes(
        image_path=image_path,
        class_names=class_names,
        image_w=image_w,
        image_h=image_h,
        max_detections=args.max_qwen_full_dets,
    )

    save_json({"image": str(image_path), **qwen_full_result}, report_dir / "03_qwen_full_image_detection.json")
    save_csv(qwen_full_to_csv_rows(qwen_full_result, image_path), report_dir / "03_qwen_full_image_detection.csv")

    qwen_full_vis = draw_boxes_qwen_full(img, qwen_full_result)
    qwen_full_vis_path = vis_dir / "03_qwen_full_image_detection.jpg"
    cv2.imwrite(str(qwen_full_vis_path), qwen_full_vis)

    print(f"Qwen full-image boxes: {len(qwen_full_result.get('detections', []))}")
    if qwen_full_result.get("qwen_error"):
        print("Qwen full-image error:", qwen_full_result.get("qwen_error"))
    print("Saved Qwen full-image:", qwen_full_vis_path)

    # -----------------------------------------------------
    # 6.4 Qwen 对 YOLO 框进行框内复判
    # -----------------------------------------------------
    print("\n[4/6] Running Qwen crop-level recheck for filtered YOLO boxes...")
    crop_records = []
    process_yolo_records = yolo_filtered_records[:max(0, int(args.max_qwen_boxes))]

    for r in process_yolo_records:
        box_idx = r["box_index"]
        xyxy = r["xyxy"]
        yolo_cls = r["yolo_class_id"]
        yolo_conf = r["yolo_confidence"]

        crop = safe_crop(img, xyxy, margin=args.crop_margin)
        if crop is None:
            print(f"Box {box_idx}: crop failed, skip Qwen crop recheck.")
            continue

        crop_path = crop_dir / f"{image_path.stem}_box{box_idx}_yolo_{yolo_cls}_{yolo_conf:.3f}.jpg"
        cv2.imwrite(str(crop_path), crop)

        qwen_crop_result = qwen.analyze_crop(
            crop_path=crop_path,
            class_names=class_names,
            yolo_cls=yolo_cls,
            yolo_conf=yolo_conf,
        )

        crop_records.append({
            "image": str(image_path),
            "box_index": box_idx,
            "crop_path": str(crop_path),
            "xyxy": xyxy,
            "yolo_class_id": yolo_cls,
            "yolo_class_name": class_names[yolo_cls],
            "yolo_confidence": yolo_conf,
            "qwen_crop_analysis": qwen_crop_result,
        })

        print(
            f"Box {box_idx}: YOLO={class_names[yolo_cls]} {yolo_conf:.3f} | "
            f"QwenCrop={qwen_crop_result.get('final_class_name')} {qwen_crop_result.get('final_probability', 0):.3f}"
        )

    save_json({"image": str(image_path), "crop_recheck": crop_records}, report_dir / "04_qwen_crop_recheck.json")
    save_csv(crop_records_to_csv_rows(crop_records), report_dir / "04_qwen_crop_recheck.csv")
    print("Saved Qwen crop recheck JSON/CSV.")

    crop_record_map = {r["box_index"]: r for r in crop_records}

    # -----------------------------------------------------
    # 6.5 IoU 对比与融合输出
    # -----------------------------------------------------
    print("\n[5/6] Matching Qwen full-image boxes with filtered YOLO boxes and fusing results...")
    fused_records = []
    qwen_full_dets = qwen_full_result.get("detections", [])

    for yr in yolo_filtered_records:
        box_idx = yr["box_index"]
        xyxy = yr["xyxy"]
        yolo_cls = yr["yolo_class_id"]
        yolo_conf = yr["yolo_confidence"]

        (
            qwen_full_match,
            qwen_full_iou,
            qwen_full_yolo_containment,
            qwen_full_box_containment,
            qwen_full_support_score,
        ) = find_best_qwen_full_match(xyxy, qwen_full_dets)
        crop_record = crop_record_map.get(box_idx, None)

        if crop_record is not None:
            qwen_crop_result = crop_record["qwen_crop_analysis"]
            crop_path = crop_record.get("crop_path", "")
            fuse = fuse_yolo_qwen_v2(
                yolo_cls=yolo_cls,
                yolo_conf=yolo_conf,
                qwen_crop_result=qwen_crop_result,
                qwen_full_match=qwen_full_match,
                qwen_full_iou=qwen_full_iou,
                qwen_full_yolo_containment=qwen_full_yolo_containment,
                qwen_full_box_containment=qwen_full_box_containment,
                class_names=class_names,
                iou_thr=args.match_iou_thr,
                full_contain_thr=args.full_contain_thr,
                change_thr=args.qwen_change_thr,
                full_support_thr=args.qwen_full_support_thr,
            )
        else:
            # 如果该框没有送入 Qwen crop 复判，仍然允许 Qwen full 通过 IoU/包含率对 YOLO 进行增强。
            qwen_crop_result = {
                "is_lesion": True,
                "uncertain": True,
                "final_class_id": yolo_cls,
                "final_class_name": class_names[yolo_cls],
                "final_probability": yolo_conf,
                "lesion_description": "该框未送入Qwen框内复判，使用YOLO结果作为框内默认结果。",
                "quality_comment": "not_sent_to_qwen_crop",
            }
            crop_path = ""
            fuse = fuse_yolo_qwen_v2(
                yolo_cls=yolo_cls,
                yolo_conf=yolo_conf,
                qwen_crop_result=qwen_crop_result,
                qwen_full_match=qwen_full_match,
                qwen_full_iou=qwen_full_iou,
                qwen_full_yolo_containment=qwen_full_yolo_containment,
                qwen_full_box_containment=qwen_full_box_containment,
                class_names=class_names,
                iou_thr=args.match_iou_thr,
                full_contain_thr=args.full_contain_thr,
                change_thr=args.qwen_change_thr,
                full_support_thr=args.qwen_full_support_thr,
            )
            if fuse.get("fuse_action", "") == "qwen_crop_uncertain_keep_yolo_score":
                fuse["fuse_action"] = "not_sent_to_qwen_crop_keep_yolo"
                fuse["fuse_reason"] = "该YOLO框未送入Qwen框内复判，且Qwen整图没有足够IoU/包含率支持，保留YOLO原始结果。"

        if qwen_full_match is not None:
            full_match_index = qwen_full_match.get("qwen_det_index", "")
            full_cls_id = qwen_full_match.get("class_id", "")
            full_cls_name = qwen_full_match.get("class_name", "")
            full_prob = qwen_full_match.get("probability", "")
            full_bbox = qwen_full_match.get("bbox", "")
        else:
            full_match_index = ""
            full_cls_id = ""
            full_cls_name = ""
            full_prob = ""
            full_bbox = ""

        record = {
            "image": str(image_path),
            "box_index": box_idx,
            "crop_path": crop_path,
            "xyxy": xyxy,
            "yolo_class_id": yolo_cls,
            "yolo_class_name": class_names[yolo_cls],
            "yolo_confidence": yolo_conf,

            "qwen_crop_is_lesion": qwen_crop_result.get("is_lesion", ""),
            "qwen_crop_class_id": qwen_crop_result.get("final_class_id", ""),
            "qwen_crop_class_name": qwen_crop_result.get("final_class_name", ""),
            "qwen_crop_probability": qwen_crop_result.get("final_probability", ""),
            "qwen_crop_uncertain": qwen_crop_result.get("uncertain", ""),
            "qwen_crop_description": qwen_crop_result.get("lesion_description", ""),
            "qwen_crop_quality": qwen_crop_result.get("quality_comment", ""),

            "qwen_full_match_index": full_match_index,
            "qwen_full_iou": float(qwen_full_iou),
            "qwen_full_yolo_containment": float(qwen_full_yolo_containment),
            "qwen_full_box_containment": float(qwen_full_box_containment),
            "qwen_full_support_score": float(qwen_full_support_score),
            "qwen_full_bbox": full_bbox,
            "qwen_full_class_id": full_cls_id,
            "qwen_full_class_name": full_cls_name,
            "qwen_full_probability": full_prob,

            "final_class_id": fuse["final_class_id"],
            "final_class_name": fuse["final_class_name"],
            "final_score": fuse["final_score"],
            "score_before_yolo_floor": fuse.get("score_before_yolo_floor", fuse["final_score"]),
            "score_floor_applied": fuse.get("score_floor_applied", False),
            "yolo_confidence_floor": fuse.get("yolo_confidence_floor", yolo_conf),
            "fuse_action": fuse["fuse_action"],
            "fuse_reason": fuse["fuse_reason"],
        }
        fused_records.append(record)

        print(
            f"Box {box_idx}: YOLO={class_names[yolo_cls]} {yolo_conf:.3f} | "
            f"FullIoU={qwen_full_iou:.3f} | FullContain={qwen_full_yolo_containment:.3f} | "
            f"Final={record['final_class_name']} {record['final_score']:.3f} | "
            f"{record['fuse_action']}"
        )

    image_level_top3 = aggregate_image_top3_from_fused(fused_records, class_names)
    final_report = {
        "image": str(image_path),
        "image_width": image_w,
        "image_height": image_h,
        "num_yolo_raw_boxes": len(yolo_records),
        "num_yolo_filtered_boxes": len(yolo_filtered_records),
        "num_yolo_removed_duplicate_boxes": len(yolo_removed_records),
        "num_qwen_full_boxes": len(qwen_full_dets),
        "fusion_match_params": {
            "match_iou_thr": args.match_iou_thr,
            "full_contain_thr": args.full_contain_thr,
            "qwen_change_thr": args.qwen_change_thr,
            "qwen_full_support_thr": args.qwen_full_support_thr,
            "note": "Qwen整图框支持YOLO框时，满足 IoU 阈值或 YOLO框被Qwen框包含率阈值任一条件即可。",
        },
        "image_level_top3_from_fused": image_level_top3,
        "yolo_raw": yolo_records,
        "yolo_filtered": yolo_filtered_records,
        "yolo_removed_duplicates": yolo_removed_records,
        "qwen_full_image_detection": qwen_full_result,
        "qwen_crop_recheck": crop_records,
        "fused_yolo_boxes": fused_records,
        "medical_disclaimer": "该结果仅为图像算法辅助分析，不作为医学诊断依据。",
    }

    save_json(final_report, report_dir / "05_fused_final_report.json")
    save_csv(fused_records_to_csv_rows(fused_records), report_dir / "05_fused_final_boxes.csv")

    fused_vis = draw_boxes_fused(img, fused_records)
    fused_vis_path = vis_dir / "05_fused_yolo_qwen_result.jpg"
    cv2.imwrite(str(fused_vis_path), fused_vis)

    compare_canvas = make_compare_canvas(
        (yolo_vis, "YOLO RAW"),
        (yolo_filtered_vis, "YOLO FILTERED"),
        (qwen_full_vis, "QWEN FULL IMAGE"),
        (fused_vis, "FUSED RESULT"),
    )
    compare_path = vis_dir / "06_compare_yolo_qwen_fused.jpg"
    cv2.imwrite(str(compare_path), compare_canvas)

    # -----------------------------------------------------
    # 6.6 控制台汇总输出
    # -----------------------------------------------------
    print("\n[6/6] Saved all results.")
    print("\n========== Image-level Top-3 from fused results ==========")
    for i, item in enumerate(image_level_top3, 1):
        print(f"Top-{i}: {item['class_name']} class_id={item['class_id']} prob={item['probability']:.3f}")

    print("\n========== Output files ==========")
    print("YOLO原始图:", yolo_vis_path)
    print("Qwen整图检测图:", qwen_full_vis_path)
    print("融合结果图:", fused_vis_path)
    print("三图对比图:", compare_path)
    print("YOLO原始JSON/CSV:", report_dir / "01_yolo_raw.json", report_dir / "01_yolo_raw.csv")
    print("Qwen整图JSON/CSV:", report_dir / "03_qwen_full_image_detection.json", report_dir / "03_qwen_full_image_detection.csv")
    print("Qwen框内复判JSON/CSV:", report_dir / "04_qwen_crop_recheck.json", report_dir / "04_qwen_crop_recheck.csv")
    print("融合最终JSON/CSV:", report_dir / "05_fused_final_report.json", report_dir / "05_fused_final_boxes.csv")
    print("裁剪图目录:", crop_dir)
    print("缓存文件:", out_dir / "qwen_cache.json")
    print("\n注意：该结果仅为图像算法辅助分析，不作为医学诊断依据。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR] 程序运行失败：", repr(e))
        traceback.print_exc()
        raise
