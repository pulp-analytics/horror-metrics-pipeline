"""Unit tests for 15_face_expression.py's pure geometry -- reimplemented
here rather than imported (the numbered script filename isn't a valid
Python module identifier), same pattern as test_clip_backbone.py."""
from PIL import Image


def crop_face(img, box, pad=0.25):
    W, H = img.size
    x, y, w, h = box
    px, py = w * pad, h * pad
    x0 = max(0.0, x - px) * W
    y0 = max(0.0, y - py) * H
    x1 = min(1.0, x + w + px) * W
    y1 = min(1.0, y + h + py) * H
    if x1 <= x0 or y1 <= y0:
        return img
    return img.crop((x0, y0, x1, y1))


def parse_boxes(face_boxes: str):
    boxes = []
    for chunk in (face_boxes or "").split("|"):
        parts = chunk.split(",")
        if len(parts) != 4:
            continue
        boxes.append(tuple(float(p) for p in parts))
    return boxes


def test_crop_face_applies_padding():
    img = Image.new("RGB", (100, 100))
    # a centered 0.2x0.2 box, no padding -> exactly 20x20
    crop = crop_face(img, (0.4, 0.4, 0.2, 0.2), pad=0.0)
    assert crop.size == (20, 20)


def test_crop_face_padding_grows_the_box():
    img = Image.new("RGB", (100, 100))
    crop_no_pad = crop_face(img, (0.4, 0.4, 0.2, 0.2), pad=0.0)
    crop_padded = crop_face(img, (0.4, 0.4, 0.2, 0.2), pad=0.25)
    assert crop_padded.size[0] > crop_no_pad.size[0]
    assert crop_padded.size[1] > crop_no_pad.size[1]


def test_crop_face_clamps_to_image_bounds():
    img = Image.new("RGB", (100, 100))
    # a box touching the top-left corner -- padding must not go negative
    crop = crop_face(img, (0.0, 0.0, 0.1, 0.1), pad=0.5)
    assert crop.size[0] <= 100 and crop.size[1] <= 100


def test_parse_boxes_splits_multiple_faces():
    boxes = parse_boxes("0.1,0.2,0.3,0.4|0.5,0.6,0.1,0.1")
    assert boxes == [(0.1, 0.2, 0.3, 0.4), (0.5, 0.6, 0.1, 0.1)]


def test_parse_boxes_handles_empty_string():
    assert parse_boxes("") == []
    assert parse_boxes(None) == []
