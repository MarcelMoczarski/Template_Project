# from IPython.core.magic import register_cell_magic
# from IPython import get_ipython
# from mypy import api

# @register_cell_magic
# def mypy(line, cell):
#   for output in api.run(['-c', '\n' + cell] + line.split()):
#     if output and not output.startswith('Success'):
#       raise TypeError(output)
#   get_ipython().run_cell(cell)