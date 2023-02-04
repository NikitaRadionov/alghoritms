from alghoritms import *


p0 = Pvector(1, 1)
p1 = Pvector(-2, 4)
p2 = Pvector(3, 6)
a = p1 - p0 # p0p1
b = p2 - p0 # p0p2
triangle = Triangle(p0, p1, p2)
v = Pvector(1, 3)
u = Pvector(5, 1)
print(triangle.clockwise)
print(triangle.is_point_in_triangle(v))
print(triangle.is_point_in_triangle(u))
