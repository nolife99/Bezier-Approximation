import shape_approximator as shapes
import time
import cupy as cp
import path_approximator
from structs import getAngle, shortestAngleDelta


def convertPathToAnchors(shape, steps, args):
    if args.plot:
        shapes.plotAlpha(shape)

    firstTime = time.time()
    anchors = (
        shapes.PiecewiseLinearToBSpline(
            shape,
            args.order,
            steps,
            args.steps,
            args.testpoints,
            args.learnrate,
            args.b1,
            args.b2,
            not args.silent,
            args.plot,
        )
        if args.mode == "bspline"
        else shapes.PiecewiseLinearToBezier(
            shape,
            steps,
            args.steps,
            args.testpoints,
            args.learnrate,
            args.b1,
            args.b2,
            not args.silent,
            args.plot,
        )
    )

    if not args.silent:
        print(f"Elapsed {time.time() - firstTime:.2f}s")

    return anchors


def getShape(values):
    pts = [
        cp.array(i.split(":"), cp.float32, copy=False)
        for i in (values[0] + ":" + values[1] + values[5][1:]).split("|")
    ]
    calculatedPath = []
    start = 0
    end = 0
    pathType = values[5][0]

    for i in range(len(pts)):
        end += 1
        if i == len(pts) - 1 or (pts[i] == pts[i + 1]).all():
            subPts = pts[start:end]
            if pathType == "L":
                subpath = subPts
            elif pathType == "P":
                if len(pts) != 3 or len(subPts) != 3:
                    subpath = path_approximator.approximateBezier(subPts)

                subpath = path_approximator.approximateCircle(subPts)
                if len(subpath) == 0:
                    subpath = path_approximator.approximateBezier(subPts)

            elif pathType == "C":
                subpath = path_approximator.approximateCatmull(subPts)
            else:
                subpath = path_approximator.approximateBezier(subPts)

            for t in subpath:
                if len(calculatedPath) == 0 or (calculatedPath[-1] != t).any():
                    calculatedPath.append(t)

            start = end

    return cp.vstack(calculatedPath), pts


def estimateCtrlPtSteps(shape, ctrlPts, args):
    reds = 0
    anchors = len(ctrlPts)
    for i in range(1, anchors - 2):
        if (ctrlPts[i] == ctrlPts[i + 1]).all():
            if (
                abs(
                    shortestAngleDelta(
                        getAngle(ctrlPts[i] - ctrlPts[i - 1]),
                        getAngle(ctrlPts[i + 2] - ctrlPts[i + 1]),
                    )
                )
                > 0.06
            ):
                reds += 1
            else:
                reds += 0.2

    prev_a = None
    angleTotal = 0
    for i in range(len(shape) - 1):
        a = getAngle(shape[i + 1] - shape[i])
        if prev_a is not None:
            angleTotal += abs(prev_a - a)
        prev_a = a

    return int(
        angleTotal * 1.3
        + reds * (min(50, args.order) if args.mode == "bspline" else anchors)
    )


def progressBar(start, total, length=60):
    spinner = "|/-\\"
    for i in range(start, total):
        yield i

        iteration = i + 1
        filledLength = length * iteration // total
        bar = "*" * filledLength + "." * (length - filledLength)

        print(
            f"\rProgress: [{bar}] {100 * (iteration / total):.1f}% ({iteration}/{total})  ",
            end=spinner[i & 3],
        )

    print(end="\r\033[J")


def encodeAnchors(anchors, s=1, o=cp.zeros(2, cp.float32)):
    li = cp.round(anchors * s + o)
    for i in range(len(li) - 1):
        if (li[i] == li[i + 1]).all():
            li[i + 1] += 1
            i -= 2

    ret = "B"
    for p in li[1:]:
        ret += f"|{p[0]:.0f}:{p[1]:.0f}"

    return li[0], ret


def writeConverted(
    anchors, plen, s=192, o=cp.array([256, 192], cp.float32, copy=False)
):
    p1, ret = encodeAnchors(anchors, s, o)
    with open("slidercode.txt", "w+") as f:
        f.write("%s,%s,0,2,0,%s,1,%s" % (int(p1[0]), int(p1[1]), ret, plen * s))

    print("Successfully saved slidercode to slidercode.txt")


def writeConvertedToFile(anchors, values, s, out, verbose):
    p1, ret = encodeAnchors(anchors, s)
    values[0] = str(int(p1[0]))
    values[1] = str(int(p1[1]))
    values[5] = ret

    with open(out, "w+") as f:
        f.write(",".join(values))

    if verbose:
        print("Successfully saved slidercode to slidercode.txt")


def printConvertedToConsole(anchors, values, s=1):
    p1, ret = encodeAnchors(anchors, s)
    values[0] = str(int(p1[0]))
    values[1] = str(int(p1[1]))
    values[5] = ret
    print(",".join(values))


def printAnchorsConsole(anchors):
    ret = ""
    for p in anchors:
        ret += f"|{p[0]:.0f}:{p[1]:.0f}"
    print(ret)


def main(args):
    if args.slidercode is not None:
        inp = args.slidercode
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                inp = line
                break

    values = inp.strip("\ufeff").split(",")
    shape, ctrlPts = getShape(values)
    if args.testpoints is None:
        args.testpoints = cp.sum(cp.ceil(shapes.distArr(shape)), dtype=cp.int32)
        print(
            "Using",
            args.testpoints,
            "resolution",
            end=" | " if args.anchors is None else "\n",
        )

    anchors = (
        args.anchors
        if args.anchors is not None
        else estimateCtrlPtSteps(shape, ctrlPts, args)
    )
    if not args.silent and args.anchors is None:
        print("Using", anchors, "anchors for new slider")

    anchors = convertPathToAnchors(shape, anchors, args)

    if args.print_output:
        printConvertedToConsole(anchors, values, 1)
    else:
        writeConvertedToFile(anchors, values, 1, args.output, not args.silent)


def main2(args):
    hitobjects = []
    at = False

    with open(args.input, "r") as f:
        for line in f:
            ls = line.strip()
            if not at:
                if ls == "[HitObjects]":
                    at = True
                continue

            if ls == "":
                continue

            hitobjects.append(ls)

    with open(args.output, "w+") as f:
        f.write("[HitObjects]\n")

    for ho in hitobjects[len(hitobjects) - 15 :]:
        values = ho.split(",")

        if values[3] != "2" and values[3] != "6" or len(values) < 8:
            with open(args.output, "a") as f:
                f.write(ho + "\n")
            continue

        shape, ctrlPts = getShape(values)
        steps = estimateCtrlPtSteps(shape, ctrlPts, args)
        if not args.silent:
            print("anchors: %s" % steps)

        p1, ret = encodeAnchors(convertPathToAnchors(shape, steps, args))
        values[0] = str(int(p1[0]))
        values[1] = str(int(p1[1]))
        values[5] = ret

        with open(args.output, "a") as f:
            f.write(",".join(values) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slidercode", type=str, default=None, help=".osu code of slider to convert."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="input.txt",
        help="Path to text file containing .osu code of slider to convert.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="slidercode.txt",
        help="Path for the output file containing the .osu code of the new slider.",
    )
    parser.add_argument(
        "--anchors",
        type=int,
        default=None,
        help="Number of anchors to use for the new slider.",
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of optimization steps"
    )
    parser.add_argument(
        "--testpoints",
        type=int,
        default=None,
        help="Resolution to compare against the new path for optimization, bigger value improves longer sliders",
    )
    parser.add_argument(
        "--learnrate",
        type=float,
        default=5,
        help="The rate of optimization for Adam optimizer.",
    )
    parser.add_argument(
        "--b1",
        type=float,
        default=0.85,
        help="The B1 parameter for the Adam optimizer. Between 0 and 1.",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.99,
        help="The B2 parameter for the Adam optimizer. Between 0 and 1.",
    )
    parser.add_argument(
        "--full-map",
        type=bool,
        default=False,
        help="If True, interprets the input file as a full beatmap and attempts to convert every slider in the beatmap.",
    )
    parser.add_argument(
        "--silent",
        type=bool,
        default=False,
        help="If True, removes all unnecessary console output.",
    )
    parser.add_argument(
        "--print-output",
        type=bool,
        default=False,
        help="Whether to print the .osu code of the converted slider to the console.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="bezier",
        choices=["bezier", "bspline"],
        help="The kind of spline to use for the converted path.",
    )
    parser.add_argument("--order", type=int, default=3, help="The B-Spline order.")
    parser.add_argument(
        "--plot", type=bool, default=False, help="Whether to plot the progress."
    )
    args = parser.parse_args()

    if args.plot:
        shapes.init_plot()

    if args.full_map:
        main2(args)
    else:
        main(args)
