import subprocess

# Install packages
subprocess.check_call(['pip', 'install', 'numpy'])
subprocess.check_call(['pip', 'install', 'openpyxl'])
subprocess.check_call(['pip', 'install', 'pandas==1.5.1']) ### don't use pandas >= 2.0.0
subprocess.check_call(['pip', 'install', 'bokeh'])
subprocess.check_call(['pip', 'install', 'matplotlib'])
subprocess.check_call(['pip', 'install', 'seaborn'])
subprocess.check_call(['pip', 'install', 'scipy'])
subprocess.check_call(['pip', 'install', 'neuprint-python'])
subprocess.check_call(['pip', 'install', 'navis'])
subprocess.check_call(['pip', 'install', 'networkx'])
subprocess.check_call(['pip', 'install', 'plotly'])
subprocess.check_call(['pip', 'install', 'img2pdf'])