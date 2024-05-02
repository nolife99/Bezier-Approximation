import numpy as cp
from scipy.interpolate import interp1d
from scipy.special import binom
import time
from slider_approximator import progressBar, writeConverted

global plt, fig, ax1, ax2, ax3, ax4


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
        li[:, 0], li[:, 1], color="green", alpha=cp.clip(30 / len(li), 0, 1), marker="."
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
    return cp.sum(distArr(shape))


def distArr(shape):
    return cp.linalg.norm(cp.diff(shape, axis=0), axis=1)


def distCumulative(shape):
    return cp.pad(cp.cumsum(distArr(shape)), (1, 0))


def bezier(anchors, steps):
    return weighBezier(anchors.shape[0], steps) @ anchors


def bSpline(anchors, order, steps):
    return bSplineBasis(order, anchors.shape[0], cp.linspace(0, 1, steps)) @ anchors


def pathify(pred, interp):
    predCumulative = distCumulative(pred)
    return interp(predCumulative / predCumulative[-1])


def bSplineBasis(p, n, x):
    p = min(max(p, 1), n - 1)
    xb = x[:, None]
    u = cp.pad(cp.linspace(0, 1, n + 1 - p), (p, p), constant_values=(0, 1))
    prev_order = cp.zeros((len(x), n - p), cp.float32)
    prev_order[
        cp.arange(len(x)), cp.clip((x * (n - p)).astype(cp.int32), 0, n - p - 1)
    ] = 1

    for c in range(1, p + 1):
        alpha = (xb - u[None, p - c + 1 : n]) / (u[p + 1 : n + c] - u[p - c + 1 : n])[
            None, :
        ]
        order = cp.zeros((len(x), n - p + c), cp.float32)
        order[:, 1:] += alpha * prev_order
        order[:, :-1] += (1 - alpha) * prev_order
        prev_order = order

    return prev_order


def weigh(anchor, n, t):
    ntm = 0
    ntp = 0
    b = anchor
    if b > n / 2:
        b = n - b
    cm = 1
    for i in range(b):
        cm = cm * (n - i) / (i + 1)
        while cm > 1 and ntm < n - anchor:
            cm *= 1 - t
            ntm += 1
        while cm > 1 and ntp < anchor:
            cm *= t
            ntp += 1

    return cm * (1 - t) ** (n - anchor - ntm) * t ** (anchor - ntp)


def weighFromT(steps, t):
    p = t[:, cp.newaxis] ** cp.arange(steps)
    return binom(cp.asarray(steps - 1), cp.arange(steps)) * p[::-1, ::-1] * p


def weighBezier(steps, res):
    if steps > 1000:
        n = steps - 1
        w = cp.zeros((res, steps), cp.float32)
        for i in range(res):
            t = i / (res - 1)

            middle = round(t * n)
            cm = weigh(middle, n, t)
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

    return weighFromT(steps, cp.linspace(0, 1, res, dtype=cp.float32))


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
    loss = cp.mean((labels - new_shape) ** 2)
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
    transposedWeights = cp.transpose(weights)
    interp = getInterp(shape)

    anchors = interp(cp.linspace(0, 1, anchorCount, dtype=cp.float32))
    points = weights @ anchors
    labels = pathify(points, interp)

    m = cp.zeros(anchors.shape, cp.float32)
    v = cp.zeros(anchors.shape, cp.float32)

    learnMask = cp.zeros(anchors.shape, cp.float32)
    learnMask[1:-1] = 1

    if plot:
        losses = (steps - 1) * [None]

    b1E = 1 - b1
    b2E = 1 - b2

    for i in progressBar(1, steps) if verbose else range(1, steps):
        points = weights @ anchors
        if i % 11 == 0:
            labels = pathify(points, interp)

        diff = labels - points
        grad = -1 / anchorCount * transposedWeights @ diff * learnMask

        m = b1 * m + b1E * grad
        v = b2 * v + b2E * grad**2
        anchors -= learnRate * m / (1 - b1**i) / (cp.sqrt(v / (1 - b2**i)) + 1e-9)

        if plot:
            loss = cp.mean(diff**2)
            losses.append(loss)
            if i % 100 == 0:
                print("Step ", i, "Loss ", loss, "Rate ", learnRate)
                plot(losses, anchors, points, labels)

    # plot(loss_list, anchors, points, labels)
    if verbose:
        points = weights @ anchors
        print(f"Final loss: {cp.mean((labels - points) ** 2):.1%}", end=" | ")

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
        bSplineBasis(order, anchorCount, cp.linspace(0, 1, res)),
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

    shape = CircleArc(cp.zeros(2), 100, 0, 2 * cp.pi)
    shape = shape.make_shape(100)
    # from shapes import GosperCurve
    # shape = GosperCurve(100)
    # shape = shape.make_shape(1)

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
    print(f"Elapsed {time.time() - firstTime:.2f}s")

    writeConverted(anchors, shapeLength(shape))

    new_shape = bSpline(anchors, order, 10000)
    test_loss(new_shape, shape)
    plot_interpolation(new_shape, anchors)
    plt.pause(1000)
