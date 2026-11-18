import numpy as np
import matplotlib.pyplot as pp

#T(x,y) = T_0 exp[-(x^2+y^2)/sigma^2]

sigma = 1 #m
T0 = 100 #k
dx=0.1
dy=0.1

#making grid
x = np.arange(-3,3+dx,dx)
y = np.arange(-3,3+dy,dy)

X,Y = np.meshgrid(x,y)

temperature = T0 * np.exp(-(X**2+Y**2)/(sigma**2))
analytic = -(2*T0*x/sigma**2) * np.exp(-(x**2)/sigma**2)

contours = pp.contourf(X, Y, temperature)
pp.colorbar()
pp.ylabel("y distance (m)")
pp.xlabel("x distance (m)")
pp.title("Temperature On A Heated Plate")
pp.savefig("temperature.png")
pp.close()

#forward = (T(x + dx) - T(x))/ dx )

dTdx = ((temperature[1:-1,2:] - temperature[1:-1,:-2]) / (2*dx))
dTdy = ((temperature[2:,1:-1] - temperature[:-2,1:-1]) / (2*dy))

#central difference
central = (temperature[len(y)//2, 2:] - temperature[len(y)//2, :-2])/(2*dx)

Xn= X[1:-1,1:-1]
Yn= Y[1:-1,1:-1]

pp.contourf(Xn,Yn,dTdx)
pp.savefig("dtdx_contour.png")
pp.close()

dTdy, dTdx = np.gradient(temperature, dy, dx)
grad_y=dTdx[len(y)//2, :]

#part d.i
pp.figure()

pp.plot(x, analytic, label="Analytical")
pp.plot(x[1:-1], central, label="Central Difference")
pp.plot(x, grad_y, label="np.gradient")

pp.xlabel("x distance (m)")
pp.ylabel("dT/dx (K/m)")
pp.title("Temperature Derivative Along y = 0")
pp.legend()
pp.grid()

pp.savefig("dtdx.png")
pp.close()

difference = np.abs(central - grad_y[1:-1])

print("Maximum absolute difference:", np.max(difference))

#d.ii
interior_error = np.abs(grad_y[1:-1] - analytic[1:-1])

edge_error = np.abs(grad_y[[0, -1]] - analytic[[0, -1]])

print("Maximum interior error:", np.max(interior_error))
print("Maximum edge error:", np.max(edge_error))

#part e

qx = -dTdx
qy = -dTdy

pp.figure()

pp.contourf(X, Y, temperature, levels=30)
pp.colorbar(label="Temperature (K)")

pp.quiver(X[::5, ::5], Y[::5, ::5],
          qx[::5, ::5], qy[::5, ::5])

pp.xlabel("x distance (m)")
pp.ylabel("y distance (m)")
pp.title("Heat Flow Vector Field")

pp.savefig("heat_flow.png")
pp.close()
