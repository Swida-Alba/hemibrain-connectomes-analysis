import subprocess

# Install packages
subprocess.check_call(['pip3', 'install', 'numpy'])
subprocess.check_call(['pip3', 'install', 'openpyxl'])
subprocess.check_call(['pip3', 'install', 'pandas<2']) ### don't use pandas >= 2.0.0
subprocess.check_call(['pip3', 'install', 'plotly'])
subprocess.check_call(['pip3', 'install', 'bokeh'])
subprocess.check_call(['pip3', 'install', 'matplotlib'])
subprocess.check_call(['pip3', 'install', 'opencv-python'])
subprocess.check_call(['pip3', 'install', 'seaborn'])
subprocess.check_call(['pip3', 'install', 'scipy'])
subprocess.check_call(['pip3', 'install', 'neuprint-python'])
subprocess.check_call(['pip3', 'install', 'navis'])
subprocess.check_call(['pip3', 'install', 'networkx'])
subprocess.check_call(['pip3', 'install', 'img2pdf'])
subprocess.check_call(['pip3', 'install', 'kaleido'])
