# KerrGeodesicSolver
Using analytical method mentioned in arXiv:1906.05090: "Analytic solutions for parallel transport along generic bound geodesics in Kerr spacetime" to do Kerr Geodesics' numerical simulations
# Some Discription
Currently it can just solve for bounded **TIMELIKE** geodesic with input:
- r1 & r2: aphelion & perihelion 
- a: spin parameter
- z1: $z_1=\cos(\theta_{max})$ where $\theta_{max}$ is the maximun theta angle in BL coordinate particle can get in Kerr spacetime
- delta_lambda & N: step size and max steps in Mino time

Originally, `main.py` will produce pictures of motions with respect to Mino time $\lambda$ in different dimensions.

## UPDATE 24.10.21
add on a package for visualization, including coordinate evolution & 3d trajectory & 3d animation
## BUG FIX 24.10.22
some typos are fixed and a mistake in arXiv:1906.05090 has been figure out and corrected as well (there should be a **PLUS** sign in front of eq.33)
## UPDATE 25.3.28
Updated new version of Kerr code. Use `KerrFast` to generate fast but with some loss of accuracy; use `Kerr` to generate accurate geodesic with single thread computation; use `KerrPara` to run `Kerr` with multicore CPU to reach a better performance.
