import numpy as np
from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.special import binom
import time

global plt, fig, ax1, ax2, ax3, ax4


def encodeAnchors(anchors, s=1, o=np.zeros(2, np.float32)):
    li = np.round(anchors * s + o)
    for i in range(len(li) - 1):
        if (li[i] == li[i + 1]).all():
            li[i + 1] += 1
            i -= 2

    ret = "B"
    for p in li[1:]:
        ret += f"|{p[0]:.0f}:{p[1]:.0f}"

    return li[0], ret


def writeConverted(anchors, plen, s=192, o=np.array([256, 192], np.float32, copy=False)):
    p1, ret = encodeAnchors(anchors, s, o)
    with open("slidercode.txt", "w+") as f:
        f.write("%s,%s,0,2,0,%s,1,%s" % (int(p1[0]), int(p1[1]), ret, plen * s))

    print("Successfully saved slidercode to slidercode.txt")


def writeConvertedToFile(anchors, values, s, out, verbose):
    p1, ret = encodeAnchors(anchors, s)
    values[0] = f"{p1[0]:.0f}"
    values[1] = f"{p1[1]:.0f}"
    values[5] = ret

    with open(out, "w+") as f:
        f.write(",".join(values))

    if verbose:
        print("Successfully saved slidercode to slidercode.txt")


def printConvertedToConsole(anchors, values, s=1):
    p1, ret = encodeAnchors(anchors, s)
    values[0] = f"{p1[0]:.0f}"
    values[1] = f"{p1[1]:.0f}"
    values[5] = ret
    print(",".join(values))


def printAnchorsConsole(anchors):
    ret = ""
    for p in anchors:
        ret += f"|{p[0]:.0f}:{p[1]:.0f}"
    print(ret)


def plotPts(p):
    ax2.cla()
    ax2.axis("equal")
    ax2.plot(p[:, 0], p[:, 1], color="green")


def plot(ll, a, p, li):
    ax1.cla()
    if ll is not None:
        ax1.plot(ll)
    ax1.set_yscale("log")

    ax2.cla()
    ax2.axis("equal")
    if p is not None:
        ax2.plot(p[:, 0], p[:, 1], color="green")
    if a is not None:
        ax2.plot(a[:, 0], a[:, 1], color="red")

    if li is not None:
        a = distArr(li)
        ax3.cla()
        ax3.plot(a, color="red")

    if p is not None:
        a = distArr(p)
        ax4.cla()
        ax4.plot(a, color="green")

    plt.pause(0.0001)


def plotAlpha(li):
    ax2.cla()
    ax2.axis("equal")
    ax2.scatter(
        li[:, 0], li[:, 1], color="green", alpha=np.clip(30 / len(li), 0, 1), marker="."
    )

    plt.draw()
    plt.pause(0.0001)


def plotVelDistr(li):
    a = distArr(li)
    ax3.cla()
    ax3.plot(a)

    plt.draw()
    plt.pause(0.0001)


def shapeLength(shape):
    return np.sum(distArr(shape))


def distArr(shape):
    return norm(np.diff(shape, axis=0), axis=1)


def distCumulative(shape):
    return np.pad(np.cumsum(distArr(shape)), (1, 0))


def bezier(anchors, steps):
    return np.matmul(weighBezier(anchors.shape[0], steps), anchors)


def bSpline(anchors, order, steps):
    return np.matmul(
        bSplineBasis(order, anchors.shape[0], np.linspace(0, 1, steps)), anchors
    )


def pathify(pred, interp):
    predCumulative = distCumulative(pred)
    return interp(predCumulative / predCumulative[-1])


def bSplineBasis(p, n, x):
    p = min(max(p, 1), n - 1)
    xb = x[:, None]
    u = np.pad(np.linspace(0, 1, n + 1 - p), (p, p), constant_values=(0, 1))
    prev_order = np.zeros((len(x), n - p), np.float32)
    prev_order[
        np.arange(len(x)), np.clip((x * (n - p)).astype(np.int32), 0, n - p - 1)
    ] = 1

    for c in range(1, p + 1):
        alpha = (xb - u[None, p - c + 1 : n]) / (u[p + 1 : n + c] - u[p - c + 1 : n])[
            None, :
        ]
        order = np.zeros((len(x), n - p + c), np.float32)
        order[:, 1:] += alpha * prev_order
        order[:, :-1] += (1 - alpha) * prev_order
        prev_order = order

    return prev_order


def weighFromT(steps, t):
    p = np.power(t[:, np.newaxis], np.arange(steps))
    return binom(np.asarray(steps - 1), np.arange(steps)) * p[::-1, ::-1] * p


def weighBezier(steps, res):
    if steps > 1000:
        n = steps - 1
        w = np.zeros([res, steps], np.float32)
        for i in range(res):
            t = i / (res - 1)

            middle = int(round(t * n))
            cm = ntm = 0
            ntp = 0
            b = middle
            if b > n // 2:
                b = n - b
            cm = 1
            for i in range(b):
                cm = cm * (n - i) / (i + 1)
                while cm > 1 and ntm < (n - middle):
                    cm *= 1 - t
                    ntm += 1
                while cm > 1 and ntp < middle:
                    cm *= t
                    ntp += 1

            cm = cm * (1 - t) ** (n - middle - ntm) * t ** (middle - ntp)
            w[i, middle] = cm

            c = cm
            for k in range(middle, n):
                c = c * (n - k) / (k + 1) / (1 - t) * t
                w[i, k + 1] = c
                if c == 0:
                    break

            c = cm
            for k in range(middle - 1, -1, -1):
                c = c / (n - k) * (k + 1) * (1 - t) / t
                w[i, k] = c
                if c == 0:
                    break

        return w

    return weighFromT(steps, np.linspace(0, 1, res, dtype=np.float32))


def getInterp(shape):
    shapeCumulative = distCumulative(shape)
    return interp1d(
        shapeCumulative / shapeCumulative[-1],
        shape,
        axis=0,
        copy=False,
        assume_sorted=True,
    )


def test_loss(new_shape, shape):
    labels = pathify(new_shape, getInterp(shape))
    loss = np.mean(np.square(labels - new_shape))
    print("loss: %s" % loss)


def plot_distribution(new_shape, shape):
    reduced_labels = pathify(new_shape, getInterp(shape))
    plotAlpha(reduced_labels)


def plot_interpolation(new_shape, shape):
    reduced_labels = pathify(new_shape, getInterp(shape))
    plot(None, new_shape, reduced_labels, None)


def PiecewiseLinearToSpline(
    shape,
    weights,
    anchorCount,
    steps,
    learnRate,
    b1,
    b2,
    verbose,
    plot,
):
    transposedWeights = np.transpose(weights)

    if verbose:
        print("Initializing interpolation...")
    interp = getInterp(shape)

    if verbose:
        print("Initializing anchors and test points...")
    anchors = interp(np.linspace(0, 1, anchorCount, dtype=np.float32))
    points = np.matmul(weights, anchors)
    labels = pathify(points, interp)

    m = np.zeros(anchors.shape, np.float32)
    v = np.zeros(anchors.shape, np.float32)

    learnMask = np.ones(anchors.shape, np.float32)
    learnMask[0] = 0
    learnMask[-1] = 0

    losses = []
    step = 0

    if verbose:
        print("Starting optimization loop")
    for step in range(1, steps):
        points = np.matmul(weights, anchors)

        if step % 11 == 0:
            labels = pathify(points, interp)

        diff = labels - points
        loss = np.mean(np.square(diff))
        grad = -1 / anchorCount * np.matmul(transposedWeights, diff) * learnMask

        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * np.square(grad)
        anchors -= learnRate * m / (1 - b1**step) / (np.sqrt(v / (1 - b2**step)) + 4.94065645841247E-324)

        losses.append(loss)
        if plot and step % 100 == 0:
            print("Step ", step, "Loss ", loss, "Rate ", learnRate)
            plot(losses, anchors, points, labels)

    # plot(loss_list, anchors, points, labels)
    if verbose:
        points = np.matmul(weights, anchors)
        print("Final loss: ", np.mean(np.square(labels - points)), step + 1)

    return anchors


def PiecewiseLinearToBezier(
    shape,
    anchorCount,
    steps,
    res,
    learnRate,
    b1,
    b2,
    verbose=True,
    plot=False,
):
    return PiecewiseLinearToSpline(
        shape,
        weighBezier(anchorCount, res),
        anchorCount,
        steps,
        learnRate,
        b1,
        b2,
        verbose,
        plot,
    )


def PiecewiseLinearToBSpline(
    shape,
    order,
    anchorCount,
    optSteps,
    res,
    learnRate,
    b1,
    b2,
    verbose=True,
    plot=False,
):
    return PiecewiseLinearToSpline(
        shape,
        bSplineBasis(order, anchorCount, np.linspace(0, 1, res)),
        anchorCount,
        optSteps,
        learnRate,
        b1,
        b2,
        verbose,
        plot,
    )


def init_plot():
    global plt, fig, ax1, ax2, ax3, ax4
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)


if __name__ == "__main__":
    init_plot()

    anchors = 6
    steps = 200
    testPoints = 200

    order = 3

    from shapes import CircleArc

    shape = CircleArc(np.zeros(2), 100, 0, 2 * np.pi)
    shape = shape.make_shape(100)
    # from shapes import GosperCurve
    # shape = GosperCurve(100)
    # shape = shape.make_shape(1)
    # from shapes import Wave
    # shape = Wave(3, 100)
    # shape = shape.make_shape(1000)

    firstTime = time.time()
    anchors = PiecewiseLinearToBSpline(
        shape,
        order,
        anchors,
        steps,
        testPoints,
        learnRate=6,
        b1=0.94,
        b2=0.86,
    )
    print("Time took:", time.time() - firstTime)

    writeConverted(anchors, shapeLength(shape))

    new_shape = bSpline(anchors, order, 10000)
    test_loss(new_shape, shape)
    plot_interpolation(new_shape, anchors)
    plt.pause(1000)
