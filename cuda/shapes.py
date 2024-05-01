import structs
import cupy as cp
import gosper


def generate_shape(func, mini, maxi, step):
    shape = []
    t = mini
    while t <= maxi + 0.00001:  # Extra to account for rounding error
        point = func(t)
        shape.append(point)
        t += step

    return cp.vstack(shape)


class ArchimedeanSpiral:
    def __init__(self, a, b, rotations):
        self.a = a
        self.b = b
        self.rotations = rotations

    def make_shape(self, num):
        delta_theta = 2 * cp.pi / num
        theta_max = self.rotations * 2 * cp.pi
        theta = 0

        return generate_shape(self.get_point, theta, theta_max, delta_theta)

    def get_point(self, theta):
        r = self.a + self.b * theta
        point = r * cp.array([cp.cos(theta), cp.sin(theta)])
        return point


class Epitrochoid:
    def __init__(self, R, r, d, size):
        self.R = R
        self.r = r
        self.d = d
        self.rotations = structs.denom((self.R + self.r) / self.r)
        self.size = size

    def get_point(self, t):
        return (
            cp.array(
                [
                    (self.R + self.r) * cp.cos(t)
                    - self.d * cp.cos(((self.R + self.r) / self.r) * t),
                    (self.R + self.r) * cp.sin(t)
                    - self.d * cp.sin(((self.R + self.r) / self.r) * t),
                ]
            )
            * self.size
        )

    def make_shape(self, num):
        step = 1 / num
        mini = 0
        maxi = 2 * cp.pi * self.rotations

        return generate_shape(self.get_point, mini, maxi, step)


class GosperCurve:
    def __init__(self, size):
        self.size = size

    def make_shape(self, level):
        g = gosper.create_gosper_fractal(level)
        x, y = gosper.generate_level(g[level])
        shape = cp.array(list(zip(x, y))) * self.size
        return shape


class EulerSpiral:
    def __init__(self, T, scale):
        self.T = T
        self.scale = scale

    def make_shape(self, N):
        shape = []

        t = 0
        n = N
        dt = self.T / N

        prev = cp.zeros([2])
        shape.append(prev)
        while n > 0:
            dx = cp.cos(t * t) * dt
            dy = cp.sin(t * t) * dt
            t += dt

            point = prev + cp.array([dx, dy]) * self.scale

            shape.append(point)

            prev = point
            n -= 1

        return cp.vstack(shape)


class Polygon:
    def __init__(self, num, skip=1):
        self.num = num
        self.skip = skip

    def make_shape(self):
        shape = []
        for i in range(self.num + 1):
            a = 2 * cp.pi / self.num * i * self.skip
            shape.append([cp.cos(a), cp.sin(a)])
        return cp.array(shape)


class Wave:
    def __init__(self, waviness, scale):
        self.waviness = waviness
        self.scale = scale

        self.arc1 = circle_arc_3points(
            cp.array([-2 * self.scale, 0]),
            cp.array([-self.scale, waviness * scale]),
            cp.array([0, 0]),
        )
        self.arc2 = circle_arc_3points(
            cp.array([0, 0]),
            cp.array([self.scale, -waviness * scale]),
            cp.array([2 * self.scale, 0]),
        )

    def get_point(self, t):
        if t < 0.5:
            return self.arc1.get_point(t * 2 * self.arc1.length + self.arc1.angle)
        else:
            return self.arc2.get_point((t * 2 - 1) * self.arc2.length + self.arc2.angle)

    def make_shape(self, num):
        step = 1 / num
        mini = 0
        maxi = 1

        return generate_shape(self.get_point, mini, maxi, step)


class CircleArc:
    def __init__(self, middle, radius, angle, length):
        self.middle = middle
        self.radius = radius
        self.angle = angle
        self.length = length

    def get_point(self, t):
        return (
            cp.array([cp.cos(t) * self.radius, cp.sin(t) * self.radius]) + self.middle
        )

    def make_shape(self, num):
        step = self.length / num
        mini = self.angle
        maxi = self.length + self.angle

        return generate_shape(self.get_point, mini, maxi, step)


def circle_arc_3points(p1, p2, p3):
    p1 = structs.arrayToPoint(p1)
    p2 = structs.arrayToPoint(p2)
    p3 = structs.arrayToPoint(p3)

    pc = structs.circle_center(p1, p2, p3)
    r = pc.distance(p1)

    d1 = p1 - pc
    d2 = p2 - pc
    d3 = p3 - pc

    a1 = d1.getAngle()
    a2 = d2.getAngle()
    a3 = d3.getAngle()

    da1 = structs.shortestAngleDelta(a1, a2)
    da2 = structs.shortestAngleDelta(a2, a3)

    if da1 * da2 > 0:
        a2 = (da1 + da2) / 2 + a1
    else:
        a2 = (da1 + da2) / 2 + cp.pi + a1

    da = structs.shortestAngleDelta(a1, a2) + structs.shortestAngleDelta(a2, a3)
    return CircleArc(pc.to_array(), r, a1, da)


class LineSegment:
    def __init__(self, start, delta):
        self.start = start
        self.delta = delta

    def get_point(self, t):
        return self.start + t * self.delta

    def make_shape(self, num):
        step = 1 / num
        mini = 0
        maxi = 1

        return generate_shape(self.get_point, mini, maxi, step)


def points_to_line_segment(p1, p2, norm=False):
    if norm:
        t = p2 - p1
        return LineSegment(p1, t / cp.linalg.norm(t))
    return LineSegment(p1, p2 - p1)


def point_angle_to_line_segment(point, angle):
    return LineSegment(point, cp.array([cp.cos(angle), cp.sin(angle)]))
