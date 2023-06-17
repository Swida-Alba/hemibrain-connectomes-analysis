# FindNetwork v2.1 (Beta)

Use these python codes to visit the NeuPrint hemibrain connectomes datasets.

Find direct or indirect connections between neuron clusters and visulize them.

Docstrings of most classes, and functions are available in the codes.

## Basic functions

### FindDirect.py

This script is used for finding direct connections between neuron clusters.

```python
fc = FindNeuronConnection(
    token='',
    dataset = 'hemibrain:v1.2.1',
    data_folder=R'D:\connectome_data',
    sourceNeurons = ['KC.*'], 
    targetNeurons = ['MBON03'], 
    custom_source_name = '', 
    custom_target_name = '',
    min_synapse_num = 1,
    min_traversal_probability = 0.001,
    showfig = False,
)
```

in the main codes, we call the FindNeuronConnection class at first. In the class, you should input your own token obtained from [Neuprint Account Page](https://neuprint.janelia.org/account).

```python
fc = FindNeuronConnection(
    ... # other parameters
    token = 'Your Auth Token',
    ... # other parameters
)
```

And you can specify the dataset to use (default is "hemibrain:v1.2.1").

```python
fc = FindNeuronConnection(
    ... # other parameters
    dataset = 'hemibrain:v1.2.1',
    ... # other parameters
)

# All available datasets are listed below:
'''
'fib19:v1.0', 
'hemibrain:v0.9', 
'hemibrain:v1.0.1', 
'hemibrain:v1.1', 
'hemibrain:v1.2.1', 
'manc:v1.0'
'''
```

If data_folder was not specified, all fetched data will be saved in the "connection_data" folder in the current directory. We highly recommand you to specify the data_folder to save all data in a specific directory. Then each time you run the codes, the data will be saved in a new folder with auto-generated name in the specified data_folder directory.

```python
fc = FindNeuronConnection(
    ... # other parameters
    data_folder = R'D:\connectome_data',
    ... # other parameters
)

```

Alternatively, you can specify the save_folder to save the current data in a specific directory for this time only.

```python
fc = FindNeuronConnection(
    ... # other parameters
    save_folder = R'D:\connectome_data\current_data',
    ... # other parameters
)
```

Source neurons and target neurons should be specified as bodyId, type, or instance (use regular expression to search for instances matching the regular expression). See details in the docstrings by hanging your cursor over the parameter name.

```python
fc = FindNeuronConnection(
    ... # other parameters
    sourceNeurons = ['KC.*'],
    targetNeurons = ['MBON03'],
    # sourceNeurons = pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist(),
    # targetNeurons = pd.read_excel('targetNeurons.xlsx', header=None).iloc[:,0].tolist(),
    ... # other parameters
)


# sourceNeurons and targetNeurons can also be read from other files, e.g. xlsx, csv, txt, etc.
# when reading xlsx and csv files, you can use:
import pandas as pd
neuron_list_1 = pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist()
neuron_list_2 = pd.read_csv('sourceNeurons.csv', header=None).iloc[:,0].tolist()
# to read the first column of the file as a list of bodyIds, types, or instances, without the header.
```

If your source or target neurons are too many items in a list, you can specify a custom name for them.

```python
fc = FindNeuronConnection(
    ... # other parameters
    custom_source_name = 'all_KCs',
    custom_target_name = 'my_MBON',
    ... # other parameters
)
```

In the min_synapse_num parameter, you can specify the minimum number of synapses between each pair of the connected neurons.

```python
fc = FindNeuronConnection(
    ... # other parameters
    min_synapse_num = 10,
    ... # other parameters
)
```

In the min_traversal_probability parameter, you can specify the minimum traversal probability between each pair of the connected neurons. This probability is calculated by the number of synapses between each pair of the connected neurons divided by the (30% total number) of input synapses of the downstream neuron. $p = max(1, w_{ij} / (W_j*0.3))$, where $w_{ij}$ is the number of synapses between neuron $i$ and $j$, and $W_j$ is the total number of input synapses of neuron $j$.

```python
fc = FindNeuronConnection(
    ... # other parameters
    min_traversal_probability = 1e-3, # 0.001
    ... # other parameters
)
```

If you want to show the figure of the connection matrix, set showfig to True, otherwise set it to False.

```python
fc = FindNeuronConnection(
    ... # other parameters
    showfig = False, # default is True
    ... # other parameters
)
```

after specified necessary parameters, you can run the codes to find the direct connections between the source neurons and the target neurons.

To find the direct connections, we call the FindNeuronConnection.InitializeNeuronInfo() method to initialize before running the FindNeuronConnection.FindDirectConnection() method.

```python
fc.InitializeNeuronInfo()
fc.FindDirectConnection()
```

In the FindNeuronConnection.FindDirectConnection() method, you can specify the full_data parameter to True (defaulty False) to do clustering and other analysis on the connection data.

```python
fc.FindDirectConnection(full_data=True) # defaultly, full_data is False
```

### FindPath.py

use this function to find direct and indirect connection paths between the neuron clusters.

### plot3dSkeleton.py

use this function to plot the 3D skeleton of the neuron clusters at different layers, you can also input a single layer to plot the skeleton of all the neurons in that layer.

## Installation: For users who can prepare the python environments by themselves

package requirements are in the setup.py (pandas should be 1.5.1)

## Installation: For users who have troubles with preparing the environments

I. Download python 3.11.3 in [Python Downloads](https://www.python.org/downloads/release/python-3113/). Please scroll down and download the circled installer according to your operating system (MacOS or Windows) and install it. Remember to check the "Add Python 3.11 to PATH" at the bottom of the installer.

![Download Version](assets/python_download.jpg)

![Install Python](assets/python_install.jpg)

II. Download Visual Studio Code (vscode) in [Visual Studio Code](https://code.visualstudio.com/) and install it. Click the "Extensions" button at the left bar to search for "Python" and "VSC-Essentials" and install them, respectively. Then, It's highly recommended to press 'Ctrl + ,' (or click the "manage" button in the bottom left corner and click 'Settings') to open the 'Settings' and search for "Auto Save" and select "OnFocusChange" to automatically save the codes, and search for "Execute In File Dir" and check the box to automatically run the codes in the current file directory.

![Add Extensions](assets/add_extensions.jpg)

![Install Extensions](assets/add_extensions1.jpg)

III. Use vscode to open setup.py, then select the python3.11.3 at the bottom right corner. Click run at the top right corner (red circle in the picture below) to install the requirements. If any error raised, please disconnect the VPN proxy and try again.

![Run Setup](assets/run_setup.jpg)

IV. Get your own token from [NeuPrint](https://neuprint.janelia.org/account). You should log in with your Google account and found your token in the "Account" page by clicking the "LOGIN" button at the top right corner.

V. Input the token in the downloaded codes, you can specify the token in statvis.LogInHemigrain() or coana.FindConnection() functions, which should be embraced by the quotation marks ('  ') as:

```python
token = 'your_token',
```

![Add Token](assets/add_token1.jpg)
![Add Token](assets/add_token2.jpg)

VI. You can find the introduction of the functions and their arguments in the codes by move your cursor over the name, e.g.

![Function Introduction](assets/function_introduction.jpg)

Now you can run the codes and get the results.
