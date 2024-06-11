# Inactivation rate of highly pathogenic avian influenza H5N1 virus (clade 2.3.4.4b) in raw milk at 63 and 72 degrees Celsius.

Franziska Kaiser, Dylan H. Morris, Arthur Wickenhagen, Reshma Mukesh, Shane Gallogly, Kwe Claude Yinda, Emmie de Wit, James O. Lloyd-Smith, Vincent J. Munster


## Repository information
This repository accompanies the article "Inactivation rate of highly pathogenic avian influenza H5N1 virus (clade 2.3.4.4b) in raw milk at 63 and 72 degrees Celsius" (F Kaiser, DH Morris et al.). It provides code for reproducing Bayesian inference analyses from the paper and producing display figures.


## License and citation information
If you use the code or data provided here, please make sure to do so in light of the project [license](LICENSE) and please cite our work as below:

- F. Kaiser, D.H. Morris, et al. Inactivation rate of highly pathogenic avian influenza H5N1 virus (clade 2.3.4.4b) in raw milk at 63 and 72 degrees Celsius. *in press* May 2024.

Bibtex record:
```
@article{kaiser2024inactivationh5n1,
  title={Inactivation rate of highly pathogenic avian influenza {H5N1} virus (clade 2.3.4.4b) in raw milk at 63 and 72 degrees {Celsius}},
  author={Kaiser, Franziska and
          Morris, Dylan H. and
		  Wickenhagen, Arthur and
		  Mukesh, Reshma and
		  Gallogly, Shane and
		  Yinda, Kwe Claude and
		  de Wit, Emmie and
		  Lloyd-Smith, James O. and
		  Munster, Vincent J.},
  journal={in press},
  year={2024},
}
```

## Directories
- `src`: all code, including data cleaning, Bayesian inference, and figure generation.
- `dat`: raw data (in the `raw` subdirectory), cleaned data once generated (in the `clean` subdirectory), and `.toml` configuration files that determine MCMC settings and specify prior parameter choices.
- `out`: mcmc, figure, and table output files, including diagnostics

# Reproducing analysis

A guide to reproducing the analysis from the paper follows. Code for the project assumes a Unix-like system (Linux, macOS, etc). It has not been tested in Windows. If you are on a Windows machine, you can try running the project within a [WSL2](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux) environment containing Python. The guide below assumes you are familiar with the [Unix command line](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview).

You will also need a working $\TeX$ installation to render the text for the figures as they appear in the paper. If you do not have $\TeX$, you can either:
1. Install [TeXLive](https://tug.org/texlive/) (or another $\TeX$ distribution)
2. Turn off $\TeX$-based rendering by setting ``mpl.rcParams['text.usetex'] = False`` in this project's `src/plotting.py` file.

## Getting the code
First download the code. The recommended way is to ``git clone`` our Github repository from the command line:

    git clone https://github.com/LloydSmithLab/h5n1-inactivation.git

Downloading it manually via Github's download button should also work.

## Basic software requirements

The analysis can be auto-run from the project `Makefile`, but you may need to install some external dependencies first. In the first instance, you'll need a working installation of Python 3 (tested on Python 3.12) with the package manager `pip`, a working installation of Gnu Make (version 4.0 or later), and a `bash` shell. Verify that you do by typing `which python3`, `which pip`, `which make`, and `which bash` at the command line.

## Virtual environments
If you would like to isolate this project's required dependencies from the rest of your system Python 3 installation, you can use a Python [virtual environment](https://docs.python.org/3/library/venv.html).

With an up-to-date Python installation, you can create one by running the following command in the top-level project directory.

```
python3 -m venv .
```

Then activate it by running the following command, also from the top-level project directory.
```
source bin/activate
```

Note that if you close and reopen your Terminal window, you may need to reactivate that virtual environment by again running `source bin/activate`.

## Python packages
A few external python packages need to be installed. You can do so by typing the following from the top-level project directory.

    pip install -r requirements.txt

Most of these packages are installed from the [Python Package Index](https://pypi.org/) except for the packages [`Pyter`](https://github.com/dylanhmorris/pyter) and [`Grizzlyplot`](https://github.com/dylanhmorris/grizzlyplot), which are pre-release Python packages developed by project co-author Dylan H. Morris, and which must be installed from Github. Those installs are linked to a specific git commit (i.e. version), with the goal of making it less likely that future changes to those packages make it difficult to reproduce the analysis here.

## Running analyses

The simplest approach is simply to type `make` at the command line, which should produce a full set of figures and results.

If you want to do things piecewise, typing `make <filename>` for any of the files present in the complete repository uploaded here should also work.

Some shortcuts are available, including:

- `make clean` removes all generated files, including even cleaned data, leaving only source code (though it does not uninstall packages)

## Note
While pseudorandom number generator seeds are set for reproducibility, numerical results may not be exactly identical depending on operating system and setup.
