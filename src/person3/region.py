REGIONS = (
    "top-left",
    "top",
    "top-right",
    "left",
    "center",
    "right",
    "bottom-left",
    "bottom",
    "bottom-right",
)


def clamp_1000(value: float) -> int:
    return max(0, min(1000, int(round(value))))


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def point_to_region(x: float, y: float) -> str:
    if y < 1000 / 3:
        row = "top"
    elif y < 2000 / 3:
        row = "middle"
    else:
        row = "bottom"

    if x < 1000 / 3:
        col = "left"
    elif x < 2000 / 3:
        col = "center"
    else:
        col = "right"

    if row == "middle" and col == "center":
        return "center"
    if row == "middle":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


def bbox_to_region(bbox) -> str:
    x, y = bbox_center(bbox)
    return point_to_region(x, y)


def pixel_bbox_to_1000(bbox_pixel, width: int, height: int):
    x1, y1, x2, y2 = bbox_pixel
    return [
        clamp_1000(x1 / width * 1000),
        clamp_1000(y1 / height * 1000),
        clamp_1000(x2 / width * 1000),
        clamp_1000(y2 / height * 1000),
    ]


def point_in_bbox(point, bbox) -> bool:
    if point is None or bbox is None:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def size_bucket(bbox) -> str:
    area = bbox_area(bbox)
    if area < 2500:
        return "small"
    if area < 15000:
        return "medium"
    return "large"

