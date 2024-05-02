import cupy as cp

BEZIER_TOLERANCE = 0.03
CATMULL_DETAIL = 50
CIRCULAR_ARC_TOLERANCE = 0.01


def lengthSq(x):
    return cp.dot(x, x)


def approximateBezier(ctrlPts):
    output = []
    count = len(ctrlPts)

    if count == 0:
        return output

    subdiv1 = cp.empty((count, 2), cp.float32)
    subdiv2 = cp.empty((count * 2 - 1, 2), cp.float32)
    left = subdiv2

    toFlatten = []
    freeBufs = []
    toFlatten.append(ctrlPts.copy())

    while len(toFlatten) > 0:
        parent = toFlatten.pop()
        if bezierFlatTolerance(parent):
            bezierApproximate(parent, output, subdiv1, subdiv2, count)

            freeBufs.append(parent)
            continue

        right = (
            freeBufs.pop() if len(freeBufs) > 0 else cp.empty((count, 2), cp.float32)
        )
        bezierSubdivide(parent, left, right, subdiv1, count)

        for i in range(count):
            parent[i] = left[i]

        toFlatten.append(right)
        toFlatten.append(parent)

    output.append(ctrlPts[-1])
    return output


def approximateCatmull(ctrlPts):
    result = []

    for i in range(len(ctrlPts) - 1):
        v1 = ctrlPts[i - 1] if i > 0 else ctrlPts[i]
        v2 = ctrlPts[i]
        v3 = ctrlPts[i + 1] if i < len(ctrlPts) - 1 else v2 + v2 - v1
        v4 = ctrlPts[i + 2] if i < len(ctrlPts) - 2 else v3 + v3 - v2

        for c in range(CATMULL_DETAIL):
            result.append(catmullGetPt(v1, v2, v3, v4, c / CATMULL_DETAIL))
            result.append(catmullGetPt(v1, v2, v3, v4, (c + 1) / CATMULL_DETAIL))

    return result


def approximateCircle(ctrlPts):
    a = ctrlPts[0]
    b = ctrlPts[1]
    c = ctrlPts[2]

    aSq = lengthSq(b - c)
    bSq = lengthSq(a - c)
    cSq = lengthSq(a - b)

    if cp.isclose(aSq, 0) or cp.isclose(bSq, 0) or cp.isclose(cSq, 0):
        return []

    s = aSq * (bSq + cSq - aSq)
    t = bSq * (aSq + cSq - bSq)
    u = cSq * (aSq + bSq - cSq)

    sum = s + t + u

    if cp.isclose(sum, 0):
        return []

    centre = (s * a + t * b + u * c) / sum
    dA = a - centre
    dC = c - centre

    r = cp.linalg.norm(dA)

    startAng = cp.arctan2(dA[1], dA[0])
    endAng = cp.arctan2(dC[1], dC[0])

    while endAng < startAng:
        endAng += 2 * cp.pi

    direct = 1
    arcRange = endAng - startAng

    orthoAtoC = c - a
    orthoAtoC = cp.array([orthoAtoC[1], -orthoAtoC[0]], cp.float32, copy=False)
    if cp.dot(orthoAtoC, b - a) < 0:
        direct = -1
        arcRange = 2 * cp.pi - arcRange

    vertexCount = (
        2
        if 2 * r <= CIRCULAR_ARC_TOLERANCE
        else int(
            max(
                2,
                cp.ceil(arcRange / (2 * cp.arccos(1 - CIRCULAR_ARC_TOLERANCE / r))),
            )
        )
    )

    output = vertexCount * []
    for i in range(vertexCount):
        theta = startAng + direct * i / (vertexCount - 1) * arcRange
        output.append(
            centre
            + cp.array([cp.cos(theta), cp.sin(theta)], cp.float32, copy=False) * r
        )

    return output


def bezierFlatTolerance(ctrlPts):
    for i in range(1, len(ctrlPts) - 1):
        p = ctrlPts[i - 1] - 2 * ctrlPts[i] + ctrlPts[i + 1]
        if lengthSq(p) > BEZIER_TOLERANCE * BEZIER_TOLERANCE * 4:
            return False
    return True


def bezierSubdivide(ctrlPts, left, right, midpoints, count):
    for i in range(count):
        midpoints[i] = ctrlPts[i]

    for i in range(count):
        left[i] = midpoints[0]
        right[count - i - 1] = midpoints[count - i - 1]

        for j in range(count - i - 1):
            midpoints[j] = (midpoints[j] + midpoints[j + 1]) / 2


def bezierApproximate(ctrlPts, output, right, left, count):
    bezierSubdivide(ctrlPts, left, right, right, count)
    for i in range(count - 1):
        left[count + i] = right[i + 1]

    output.append(ctrlPts[0].copy())
    for i in range(1, count - 1):
        index = 2 * i
        output.append(0.25 * (left[index - 1] + 2 * left[index] + left[index + 1]))


def catmullGetPt(vec1, vec2, vec3, vec4, t):
    t2 = t * t
    t3 = t * t2

    return cp.array(
        [
            0.5
            * (
                2 * vec2[0]
                + (-vec1[0] + vec3[0]) * t
                + (2 * vec1[0] - 5 * vec2[0] + 4 * vec3[0] - vec4[0]) * t2
                + (-vec1[0] + 3 * vec2[0] - 3 * vec3[0] + vec4[0]) * t3
            ),
            0.5
            * (
                2 * vec2[1]
                + (-vec1[1] + vec3[1]) * t
                + (2 * vec1[1] - 5 * vec2[1] + 4 * vec3[1] - vec4[1]) * t2
                + (-vec1[1] + 3 * vec2[1] - 3 * vec3[1] + vec4[1]) * t3
            ),
        ],
        cp.float32,
        copy=False,
    )
