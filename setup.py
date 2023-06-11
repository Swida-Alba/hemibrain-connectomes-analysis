import subprocess

# Install packages
subprocess.check_call(['python3.11', '-m', 'pip', 'install', 'numpy'])
subprocess.check_call(['python3.11', '-m', 'pip', 'install', 'openpyxl'])
subprocess.check_call(['python3.11', '-m', 'pip', 'install', 'pandas==1.5.1']) ### don't use pandas >= 2.0.0
subprocess.check_call(['python3.11', '-m', 'pip', 'install', 'plotly'])
subprocess.check_call(['python3.11', '-m', 'pip', 'install', 'bokeh'])
subprocess.check_call(['python3.11', '-m', 'pip', 'install', 'matplotlib'])
subprocess.check_call(['python3.11', '-m', 'pip', 'install', 'seaborn'])
subprocess.check_call(['python3.11', '-m', 'pip', 'install', 'scipy'])
subprocess.check_call(['python3.11', '-m', 'pip', 'install', 'neuprint-python'])
subprocess.check_call(['python3.11', '-m', 'pip', 'install', 'navis'])
subprocess.check_call(['python3.11', '-m', 'pip', 'install', 'networkx'])
subprocess.check_call(['python3.11', '-m', 'pip', 'install', 'img2pdf'])