# KerrGeodesicSolver
Using analytical method mentioned in arXiv:1906.05090: "Analytic solutions for parallel transport along generic bound geodesics in Kerr spacetime" to do Kerr Geodesics' numerical simulations
# Some Discription
Currently it can just solve for bounded **TIMELIKE** geodesic with input:
- r1 & r2: perihelion & aphelion
- a: spin parameter
- z1: $z_1=\cos(\theta_{max})$ where $\theta_{max}$ is the maximun theta angle in BL coordinate particle can get in Kerr spacetime
- delta_lambda & N: step size and max steps in Mino time

Originally, `main.py` will produce pictures of motions with respect to Mino time $\lambda$ in different dimensions.

## UPDATE
add on a package for visualization, including coordinate evolution & 3d trajectory & 3d animation
