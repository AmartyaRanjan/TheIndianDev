"""Hand-written clean functions shared by the mutation tests."""

SAMPLES = [
    '''def find_index(xs, target):
    """Return the index of target in xs, or -1."""
    for i in range(len(xs)):
        if xs[i] == target:
            return i
    return -1
''',
    '''def safe_mean(values, default=None):
    if values is None:
        return default
    total = 0
    count = 0
    for v in values:
        if v is not None and v >= 0:  # skip missing and negative readings
            total += v
            count += 1
    if count == 0:
        return default
    return total / count
''',
    '''def merge(a, b):
    result = []
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result
''',
    '''def parse_pairs(text, sep=","):
    items = text.split(sep) or []
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        if not key:
            continue
        out[key.strip()] = int(value) if value else 0
    return out
''',
    '''def clamp(x, lo, hi):
    return max(lo, min(x, hi))
''',
    '''def label(score, threshold=0.5, name=None):
    tag = name if name is not None else "item"
    ok = score >= threshold
    return f"{tag}:{'pass' if ok else 'fail'}"
''',
]
