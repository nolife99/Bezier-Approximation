import shape_approximator as shapes
import time
import numpy as np
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
        print("Time took:", time.time() - firstTime)

    return anchors


def getShape(values):
    pts = np.vstack(
        [
            np.array(i.split(":"), np.float32, copy=False)
            for i in (values[0] + ":" + values[1] + values[5][1:]).split("|")
        ]
    )
    calculatedPath = []
    start = 0
    end = 0
    pathType = values[5][0]

    for i in range(len(pts)):
        end += 1
        if i == len(pts) - 1 or (pts[i] == pts[i + 1]).all():
            subPts = pts[start:end]
            if pathType == "L":
                subpath = path_approximator.approximateLinear(subPts)
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

    return np.vstack(calculatedPath), pts


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
                > 0.1
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

    return min(
        int(
            2
            + angleTotal * 1.13
            + reds * (min(50, args.order) if args.mode == "bspline" else 50)
        ),
        10000,
    )


def main(args):
    if args.slidercode is not None:
        inp = args.slidercode
    else:
        with open(args.input, "r") as f:
            lines = f.readlines()
        inp = lines[0][3:]

    if not args.silent:
        print(inp)
    values = inp.split(",")
    shape, ctrlPts = getShape(values)

    anchors = (
        args.anchors
        if args.anchors is not None
        else estimateCtrlPtSteps(shape, ctrlPts, args)
    )
    if not args.silent:
        print("anchor count: %s" % anchors)

    anchors = convertPathToAnchors(shape, anchors, args)

    if args.print_output:
        shapes.printConvertedToConsole(anchors, values, 1)
    else:
        shapes.writeConvertedToFile(anchors, values, 1, args.output, not args.silent)


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

        p1, ret = shapes.encodeAnchors(convertPathToAnchors(shape, steps, args))
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
        help="Path for the output file containing the .osu code of the converted slider.",
    )
    parser.add_argument(
        "--anchors",
        type=int,
        default=None,
        help="Number of anchors to use for the converted slider.",
    )
    parser.add_argument(
        "--steps", type=int, default=10000, help="Number of optimization steps."
    )
    parser.add_argument(
        "--testpoints",
        type=int,
        default=1000,
        help="Number of points to evaluate the converted path at for optimization, basically a resolution.",
    )
    parser.add_argument(
        "--learnrate",
        type=float,
        default=4,
        help="The rate of optimization for Adam optimizer.",
    )
    parser.add_argument(
        "--b1",
        type=float,
        default=0.5,
        help="The B1 parameter for the Adam optimizer. Between 0 and 1.",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.5,
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
        default=True,
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
