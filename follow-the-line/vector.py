class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def magnitude_squared(self):
        return self.x ** 2 + self.y ** 2

    def magnitude(self):
        return math.sqrt(self.magnitude_squared())

    def clamp(self, min_val=-1, max_val=1):
        self.x = min(max_val, max(min_val, self.x))
        self.y = min(max_val, max(min_val, self.y))

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Vector(x, y)

    def __truediv__(self, other):
        x = self.x / other.x
        y = self.y / other.y
        return Vector(x, y)

    def __str__(self):
        return f"({self.x:.0f}, {self.y:.0f})"