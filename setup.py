from setuptools import setup, find_packages
import sys


# def is_colab():
#     for module in sys.modules.keys():
#         is_in = "sys" in module
#         if is_in:
#             return True

# def main():
#     print("hi")

# sys.path.insert(0, "/content/gdrive/MyDrive/Marcel_Moczarski/University/Semester/Master/tutorials/own_projects/template_project")
setup(
    name="template_project",
    version=0.1,
    description="template for future projects",
    url="https://github.com/MarcelMoczarski/template_project",
    author="Marcel Moczarski",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "toml",
    ]
)

#     if is_colab():
#         sys.path.insert(0, "/content/gdrive/MyDrive/Marcel_Moczarski/University/Semester/Master/tutorials/own_projects/template_project")

# # if __name__ == "__main__":
#     sys.path.insert(0, "/content/gdrive/MyDrive/Marcel_Moczarski/University/Semester/Master/tutorials/own_projects/template_project")
#     print("hi")
#     main()

# setup(
#     name="radionets",
#     version="0.1.9",
#     description="Imaging radio interferometric data with neural networks",
#     url="https://github.com/Kevin2/radionets",
#     author="Kevin Schmidt, Felix Geyer, Kevin Laudamus",
#     author_email="kevin3.schmidt@tu-dortmund.de",
#     license="MIT",
#     packages=find_packages(),
#     install_requires=[
#         "fastai <= 2.3.0",
#         "fastcore <= 1.3.1",
#         "kornia",
#         "pytorch-msssim",
#         "numpy",
#         "astropy",
#         "tqdm",
#         "click",
#         "geos",
#         "shapely",
#         "proj",
#         "cartopy",
#         "ipython",
#         "jupyter",
#         "jupytext",
#         "h5py",
#         "scikit-image",
#         "pandas",
#         "requests",
#         "toml",
#         "pytest",
#         "pytest-cov",
#         "pytest-order",
#     ],
#     setup_requires=["pytest-runner"],
#     tests_require=["pytest"],
#     zip_safe=False,
#     entry_points={
#         "console_scripts": [
#             "radionets_simulations = radionets.simulations.scripts.simulate_images:main",
#             "radionets_training = radionets.dl_training.scripts.start_training:main",
#             "radionets_evaluation = radionets.evaluation.scripts.start_evaluation:main",
#         ],
#     },
#     classifiers=[
#         "Development Status :: 4 - Beta",
#         "Intended Audience :: Science/Research",
#         "License :: OSI Approved :: MIT License",
#         "Natural Language :: English",
#         "Operating System :: OS Independent",
#         "Programming Language :: Python",
#         "Programming Language :: Python :: 3.6",
#         "Programming Language :: Python :: 3.7",
#         "Programming Language :: Python :: 3.8",
#         "Programming Language :: Python :: 3 :: Only",
#         "Topic :: Scientific/Engineering :: Astronomy",
#         "Topic :: Scientific/Engineering :: Physics",
#         "Topic :: Scientific/Engineering :: Artificial Intelligence",
#         "Topic :: Scientific/Engineering :: Information Analysis",
#     ],
# )
