# poppy-cupy-tests
This repository demonstrates the use of POPPY with CuPy for faster computation times. 

To begin with, to use the CuPy accelarted math option in POPPY, CuPy must be installed along with additional packages that are required for some cupyx.scipy functions. The overall instructions for installation can be found on the [CuPy installation page](https://docs.cupy.dev/en/stable/install.html), but the recommended command for installation is shown below.

conda install -c conda-forge cupy cudnn cutensor nccl

Overall, this repo contains many files used for testing and debugging while CuPy's implementation was still in its early stages. The primary files though are poppy_cpu_vs_cupy_demo.ipynb, which is a simple demonstartion of using optical elements and optical systems to calculate a PSF, and poppy_cpu_vs_cupy_speed_tests.ipynb, which simply runs timing tests for the same systems in the demo notebook. 

The variety of optics used for debugging can be found and imported from the poppy_optics.py file. This is where the optical elements are instantiated so they can be used in the notebooks for debugging. 

