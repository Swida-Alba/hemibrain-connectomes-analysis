# To visit neuPrint and find connection path between neurons
# Created on Sep, 16, 2020 by KRL
# Last modified on Nov, 4, 2020 by KRL
# E-mail: krleng@pku.edu.cn

# V1.0       Sep, 16, 2020, by KRL
# Based on FindNeuron_v1.0.5
# This program is aimed to find the connection path between neurons
# V1.0.1     Sep, 19, 2020, by KRL
# Classifies paths by the number of inter-layers and does rough statistics
# Optimizes input parameters.
# Adds progress bar
# V1.0.2     Oct, 27, 2020, by KRL
# Fixes bugs that cannot recognize bodyId correctly when packed in a list
# Fixes bugs that error arises when not found paths
# V1.0.3     Nov, 5, 2020, by KRL
# Adds statistics based on single path, instead of layers
# V1.0.4     Dec, 17, 2020, by KRL
# if error occurs, try to re-run the program, for badgateway.
# V1.0.5     Jan, 01, 2021, by KRL
# Fixes that errors ocurr when no paths were found, which is better than V1.0.2

from neuprint import *
def findpath():
    ################################ parameters ##########################################
    # appoint the target neuron and source neuron of interest and input parameters
    from datetime import datetime
    import pandas as pd
    database = 'hemibrain:v1.2.1' # specify dataset version
    # directory = '/Users/apple/Desktop/Oscillator_to_DAN/Oscillator_to_PAM_subsets/' # absolute directory to save all results, format for MacOS.
    directory = R'C:\Users\krlen\Desktop'
    # directory = r'C:\???\???\''  # for Windows
    # source_Neurons = ['PPL101']  # every item in this list will be save as a file.
    # Supports the criteria class search by 'type', 'bodyId', or 'instance', if customize it, fill it in the 'criteria' in the next loop and DON't fill anything in this line
    # if pack all neurons in one file, especially search by body ID, use the next clause and remember to custom 'filename'

    # !!!!!!!!!!!!!!!!!!! lengths of following 3 variables should be equal, including filenames !!!!!!!!!!
    # df = pd.read_excel(r'F:\conn_PVLP.xlsx', sheet_name='source_conn', header=0)
    # source_Neurons = [df['bodyId'].tolist()]
    # df = pd.read_excel(r'F:\conn_PVLP.xlsx', sheet_name='target_conn', header=0)
    # target_Neurons = [df['bodyId'].tolist()]
    
    source_Neurons = ['aMe12']
    target_Neurons = ['ER5']
    filename_custom = [] # ['Morning_to_Evening']

    Notes = ''
    minSnpFind = 10  # minimal synapse counts for screening connections in every layer
    maxTime = 5.0 # seconds
    maxLayer = 3 # max inter-layers for statistics
    timeFlag = 1  # if 1, print time checkpoints, else, nothing, for improving performance and debugging

    visualFlag = 1 # if 1, print figures
    maxPathNum = 1 # set a limit and plot at most ? paths in each pair
    PNG_export = 1 # if 1, export skeleton results to png, else only save in html. You can only export html for better performance because picture rendering is time-consuming.

    import matplotlib
    matplotlib.use('TkAgg') # for MAC OS only, if not, comment it (I don't know what if run in Windows)
    import datetime
    curr_time = datetime.datetime.now()
    import warnings
    warnings.filterwarnings('ignore')
    ########################################################################################

    # figure properties
    fontSize = 14
    max_Y = 40000
    max_X = 50000
    figHeight = 9.6 # inch
    figWidth = figHeight * (max_X / max_Y)
    figureSize = [figWidth, figHeight] # inch. first argument is width(x), second argument is height(y)
    dpi = 144 # dots per inch, no less than 72

    if (len(source_Neurons) != len(target_Neurons)) & (len(source_Neurons) != 1) & (len(target_Neurons) != 1):
        raise ValueError("Lengths of source_Neurons and target_Neurons not match")
    elif len(source_Neurons) == 1:
        source_Neurons = source_Neurons * len(target_Neurons)
    elif len(target_Neurons) == 1:
        target_Neurons = target_Neurons * len(source_Neurons)

    filenames = [] # defaut filename
    for g in range(len(target_Neurons)):
        if type(source_Neurons[g]) == list:
            if type(source_Neurons[g][0]) == int:
                filename_1 = str(source_Neurons[g][0]) + '_etc'
                # filename_1 = 'VP2VP3_PN'
                # filename_1 = 'Oscillators'
            elif str(source_Neurons[g][0])[-1] == '*':
                filename_1 = source_Neurons[g][0][:-2] + '_etc'
            else:
                filename_1 = str(source_Neurons[g][0]) + '_etc'
        elif type(source_Neurons[g]) == int:
            filename_1 = str(source_Neurons[g])
        elif str(source_Neurons[g])[-1] == '*':
            filename_1 = source_Neurons[g][:-2]
        else:
            filename_1 = str(source_Neurons[g])

        if type(target_Neurons[g]) == list:
            if type(target_Neurons[g][0]) == int:
                filename_2 = str(target_Neurons[g][0]) + '_etc'
            elif str(target_Neurons[g][0])[-1] == '*':
                filename_2 = target_Neurons[g][0][:-2] + '_etc'
            else:
                filename_2 = str(target_Neurons[g][0]) + '_etc'
        elif type(target_Neurons[g]) == int:
            filename_2 = str(target_Neurons[g])
        elif str(target_Neurons[g])[-1] == '*':
            filename_2 = target_Neurons[g][:-2]
        else:
            filename_2 = str(target_Neurons[g])
        filenames.append(str(filename_1) + '_to_' + str(filename_2))

    parameters = {'dataset': database,
                  'date': curr_time,
                  'min_synapses': minSnpFind,
                  'max_searching_time': maxTime,
                  'max_path_num_print': maxPathNum,
                  'max_layer_stat': maxLayer,
                  'Notes': Notes}
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # timer
    import time
    t0 = time.time()
    t = t0

    # include modules
    import os
    import shutil
    from chromedriver_binary.utils import get_chromedriver_path
    from selenium.webdriver import Chrome, ChromeOptions
    from selenium import webdriver
    from webdriver_auto_update import check_driver
    if not check_driver(get_chromedriver_path()):
        driver = webdriver.Chrome()
        driver.get("http://www.python.org")

    options = ChromeOptions()
    options.binary_location = shutil.which('chrome')

    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=2000x2000")

    web_driver = Chrome(executable_path=os.path.join(get_chromedriver_path(), 'chromedriver'),
                        options=options)

    from PIL import Image
    import img2pdf
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': fontSize})
    import pandas as pd
    from bokeh.plotting import figure, output_notebook, output_file, save
    import bokeh
    from bokeh.io import export_png
    output_notebook()
    parameters = pd.DataFrame.from_dict(parameters, orient = 'index')

    # functions
    def mkdir(AP):
        folder = os.path.exists(AP)
        if not folder:
            os.makedirs(AP)

    if timeFlag == 1: tt = t; t = time.time(); print('Configuration:', t-tt, 's')  # time checkpoint



    # Create a Client to visit neuPrint
    c = Client('neuprint.janelia.org',
               dataset = database, # select dataset v1.0.1 or v1.1
               token   = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImtybGVuZzEyMTg0QGdtYWlsLmNvbSIsImxldmVsIjoibm9hdXRoIiwiaW1hZ2UtdXJsIjoiaHR0cHM6Ly9saDYuZ29vZ2xldXNlcmNvbnRlbnQuY29tLy1VX3VwaU9BUXRPOC9BQUFBQUFBQUFBSS9BQUFBQUFBQUFBQS9BTVp1dWNuWEIxbkJHVjVBRFZEVlB2N0RyZVZoWnpZb0xBL3Bob3RvLmpwZz9zej01MD9zej01MCIsImV4cCI6MTc4MDIyMjU2Mn0.qFNak5jXGNuHmAhtt7ZAesANhmkxDz_kDiObEW5uErc')
    c.fetch_version()

    if timeFlag == 1: tt = t; t = time.time(); print('Logged in:', t-tt, 's')  # time checkpoint

    NotFound = pd.DataFrame.from_dict({'Results': ['Not Found']})

    for g in range(len(source_Neurons)):
        if len(filename_custom) > g:
            if len(filename_custom[g]) != 0:
                filename = filename_custom[g]
            else:
                filename = filenames[g]
        else:
            filename = filenames[g]

        print('Processing:', source_Neurons[g], 'to', target_Neurons[g])

        if type(source_Neurons[g]) == list:
            if type(source_Neurons[g][0]) == int:
                criteria_source = NeuronCriteria(bodyId=source_Neurons[g])
                print('source bodyId etc:', source_Neurons[g])
            elif str(source_Neurons[g][0])[-1] == '*':
                criteria_source = NeuronCriteria(type = source_Neurons[g], regex=True)
                print('source instance etc:', source_Neurons[g][:-2])
            else:
                criteria_source = NeuronCriteria(type=source_Neurons[g])
                print('source type etc:', source_Neurons[g])
        elif type(source_Neurons[g]) == int:
            criteria_source = NeuronCriteria(bodyId=source_Neurons[g])
            print('source bodyId:', source_Neurons[g])
        elif str(source_Neurons[g])[-1] == '*':
            criteria_source = NeuronCriteria(type=source_Neurons[g], regex=True)
            print('source instance:', source_Neurons[g][:-2])
        else:
            criteria_source = NeuronCriteria(type=source_Neurons[g])
            print('source type:', source_Neurons[g])

        if type(target_Neurons[g]) == list:
            if type(target_Neurons[g][0]) == int:
                criteria_target = NeuronCriteria(bodyId=target_Neurons[g])
                print('target bodyId etc:', target_Neurons[g])
            elif str(target_Neurons[g][0])[-1] == '*':
                criteria_target = NeuronCriteria(type=target_Neurons[g], regex=True)
                print('target instance:', target_Neurons[g][:-2])
            else:
                criteria_target = NeuronCriteria(type=target_Neurons[g])
                print('target type:', target_Neurons[g])
        elif type(target_Neurons[g]) == int:
            criteria_target = NeuronCriteria(bodyId=target_Neurons[g])
            print('target bodyId etc:', target_Neurons[g])
        elif str(target_Neurons[g])[-1] == '*':
            criteria_target = NeuronCriteria(type=target_Neurons[g], regex=True)
            print('target instance:', target_Neurons[g][:-2])
        else:
            criteria_target = NeuronCriteria(type=target_Neurons[g])
            print('target type:', target_Neurons[g])

        if directory[0] == '/':
            if directory[-1] != '/':
                directory = directory + '/'
            path = directory + filename + '/' # for MacOS, if Windows, '\\'
        else:
            if directory[-1] != '\\':
                directory = directory + '\\'
            path = directory + filename + '\\' # for Windows
        mkdir(path) # make a folder

        # fetch source and target neurons
        neuron_source_df, ROI_counts_source_df = fetch_neurons(criteria_source)
        neuron_target_df, ROI_counts_target_df = fetch_neurons(criteria_target)
        if timeFlag == 1: tt = t; t = time.time(); print('Fetched Source&Target Neurons:', t-tt, 's')  # time checkpoint
        source_ID = neuron_source_df.bodyId.tolist()
        target_ID = neuron_target_df.bodyId.tolist()

        # save in excel
        writer_basic = pd.ExcelWriter(path + '_Paths_' + filename + '.xlsx')
        parameters.to_excel(writer_basic, encoding='utf-8',sheet_name=filename + '_parameters')  # input parameters for filtering
        png_path_conn = list()
        path_all = pd.DataFrame()
        path_class = pd.DataFrame()
        path_inter_all = pd.DataFrame()
        path_inter = [pd.DataFrame()] * (maxLayer + 2) # save paths that have 'i' inter-layer(s), including 0 layer, and save paths that have more than 'maxLayer' inter-layers in path_inter[maxLayer+1]
        # find path
        for i, source_i in enumerate(source_ID):
            for j, target_j in enumerate(target_ID):
                print('\rLoading... source', str(i + 1) + '/' + str(len(source_ID)), 'target',
                      str(j + 1) + '/' + str(len(target_ID)),
                      str(round(100 * (i * len(target_ID) + j + 1) / (len(source_ID) * len(target_ID)), 2)) + '%', end='')
                path_ij = fetch_shortest_paths(upstream_bodyId = source_i,
                                               downstream_bodyId = target_j,
                                               min_weight = minSnpFind,
                                               intermediate_criteria = None,
                                               timeout = maxTime)
                # export to excel
                path_ij.to_excel(writer_basic, encoding ='utf-8', sheet_name =str(source_i) + '_to_' + str(target_j))
                path_all = path_all.append(path_ij, ignore_index=True)
                path_num = path_ij.path.max()
                if path_num >= 0:
                    for k in range(path_num+1):
                        path_ij_types = path_ij[path_ij.path == k].type.tolist()
                        path_ij_bodyIds = path_ij[path_ij.path == k].bodyId.tolist()
                        path_ij_weight = path_ij[path_ij.path == k].weight.tolist()
                        path_ij_min_weight = min(path_ij_weight[1:]) # min weight is the minimal weight through the path.
                        type_vector = str()
                        bodyId_vector = str()
                        layerNum = len(path_ij_types)
                        for l in range(layerNum): # including source and target
                            type_vector = type_vector + '->' + str(path_ij_types[l])
                        for l in range(1,layerNum-1): # excluding source and target
                            bodyId_vector = bodyId_vector + '->' + str(path_ij_bodyIds[l])
                        type_vector = type_vector[2:]
                        bodyId_vector = bodyId_vector + '->'
                        path_ij_class = pd.DataFrame({'path': k,
                                                      'bodyId': bodyId_vector,
                                                      'type': type_vector,
                                                      'interlayer': layerNum-2,
                                                      'min_weight': path_ij_min_weight}, # min weight is the minimal weight through the path.
                                                      index = [1])

                        # if (type_vector.find('DPM') == -1) & (type_vector.find('APL') == -1):
                        #     path_class = path_class.append(path_ij_class)
                        #     print('\n\n\033[33m  Some paths have been excluded  \033[0m\n\n')
                        # if type_vector.find('FB') != -1:
                        #     path_class = path_class.append(path_ij_class)
                        #     print('\n\n\033[33m  Some paths have been excluded  \033[0m\n\n')
                        path_class = path_class.append(path_ij_class)
                    # strip source and target neurons in path
                    for l in range(path_num+1):
                        path_ij_l = path_ij[path_ij.path == l][1:-1]
                        path_inter_all = path_inter_all.append(path_ij_l)
                        path_length = len(path_ij_l)
                        if path_length == 0:
                            path_inter[0] = path_inter[0].append(path_ij[path_ij.path == l])  # not stripped
                        elif path_length <= maxLayer:
                            path_inter[path_length] = path_inter[path_length].append(path_ij_l)
                        else:
                            path_inter[maxLayer+1] = path_inter[maxLayer+1].append(path_ij_l)
        writer_basic.save()
        if timeFlag == 1: tt = t; t = time.time(); print('\nPaths successfully saved:', t - tt, 's')  # time checkpoint

        writer_summary = pd.ExcelWriter(path + '_Summary_' + filename + '.xlsx')
        parameters.to_excel(writer_summary, encoding='utf-8', sheet_name=filename + '_parameters')  # input parameters for filtering
        neuron_source_df.to_excel(writer_summary, encoding ='utf-8', sheet_name ='source_neurons')
        neuron_target_df.to_excel(writer_summary, encoding ='utf-8', sheet_name ='target_neurons')
        path_class.to_excel(writer_summary, encoding ='utf-8', sheet_name ='path_concat_class')
        path_all.to_excel(writer_summary, encoding ='utf-8', sheet_name ='All_paths_concat') # concatenate all paths in one sheet, for debugging
        path_inter_all.to_excel(writer_summary, encoding ='utf-8', sheet_name ='all_inter_neurons') # all interneurons along all paths

        if path_all.empty:
            NotFound.to_excel(writer_summary, encoding='utf-8', sheet_name='DataNotFound')
        else:
            # count specific paths appeared
            path_type_counts = path_class['type'].value_counts()
            path_neuron_counts = path_class['bodyId'].value_counts()
            # add the number of interlayer to path_type_counts and path_neuron_counts
            path_interlayer = []
            path_weight = []
            for i in path_type_counts.index:
                path_interlayer.append(path_class[path_class.type == i].interlayer.unique()[0])
                path_weight.append(sum(path_class[path_class.type == i].min_weight.tolist()))
            path_type_counts = pd.DataFrame(path_type_counts)
            path_type_counts['interlayer'] = path_interlayer
            path_type_counts['weight'] = path_weight
            path_type_counts = path_type_counts.sort_values(by = ['interlayer','weight','type'], axis = 0, ascending = [True,False,False])

            path_interlayer = []
            path_type_t = [] # add type info to neuron
            path_weight = []
            for i in path_neuron_counts.index:
                path_interlayer.append(path_class[path_class.bodyId == i].interlayer.unique()[0])
                path_type_t.append(path_class[path_class.bodyId == i].type.unique().tolist())
                path_weight.append(sum(path_class[path_class.bodyId == i].min_weight.tolist()))
            path_neuron_counts = pd.DataFrame(path_neuron_counts)
            path_neuron_counts['interlayer'] = path_interlayer
            path_neuron_counts['weight'] = path_weight
            path_neuron_counts['type'] = path_type_t
            path_neuron_counts = path_neuron_counts.sort_values(by = ['interlayer','weight','bodyId'], axis = 0, ascending = [True,False,False])

            path_type_counts.to_excel(writer_summary, encoding='utf-8', sheet_name='path_type_counts')
            path_neuron_counts.to_excel(writer_summary, encoding='utf-8', sheet_name='path_neuron_counts')



            for i, inter_layers in enumerate(path_inter):
                if i <= maxLayer:
                    inter_layers.to_excel(writer_summary, encoding ='utf-8', sheet_name = str(i) + '_interlayer')
                else:
                    inter_layers.to_excel(writer_summary, encoding='utf-8', sheet_name=str(i) + '_and_more_interlayer')

            type_counts_all = path_all['type'].value_counts() # frequency of neurons along the path, type, including target and source
            neuron_counts_all = path_all['bodyId'].value_counts() # frequency of neurons along the path, body id, including target and source
            inter_type_counts_all = path_inter_all['type'].value_counts() # frequency of inter-neuron types, based on neuron type, excluding target and source
            inter_neuron_counts_all = path_inter_all['bodyId'].value_counts() # frequency of interneurons, based on body id, excluding target and source

            inter_type_counts = [pd.DataFrame()] * (maxLayer + 2) # I've forgot why it's 2, but it works.
            inter_neuron_counts = [pd.DataFrame()] * (maxLayer + 2)
            for i in range(1, maxLayer + 2):
                if not path_inter[i].empty:
                    inter_type_counts[i] = path_inter[i]['type'].value_counts()
                    inter_neuron_counts[i] = path_inter[i]['bodyId'].value_counts()

                    inter_types_t = []
                    # add neuron type to inter_neuron_counts[i]
                    for j in inter_neuron_counts[i].index:
                        inter_types_t.append(path_inter[i][path_inter[i].bodyId == j].type.unique()[0])
                    inter_neuron_counts[i] = pd.DataFrame(inter_neuron_counts[i])
                    inter_neuron_counts[i]['type'] = inter_types_t

                    if i <= maxLayer:
                        inter_type_counts[i].to_excel(writer_summary, encoding='utf-8',sheet_name=str(i)+'_inter_type_counts')
                        inter_neuron_counts[i].to_excel(writer_summary, encoding='utf-8',sheet_name=str(i)+'_inter_neuron_counts')
                        inter_type_counts_layers = [pd.DataFrame()] * i
                        inter_neuron_counts_layers = [pd.DataFrame()] * i # in paths having the same number of layers, calculate weights of neurons in each layer
                        for j in range(i):
                            inter_type_counts_layers[j] = path_inter[i]['type'][j::i].value_counts()
                            inter_neuron_counts_layers[j] = path_inter[i]['bodyId'][j::i].value_counts()

                            inter_types_L_t = []
                            # add neuron type to inter_neuron_counts[i]
                            for k in inter_neuron_counts_layers[j].index:
                                inter_types_L_t.append(path_inter[i][path_inter[i].bodyId == k].type.unique()[0])
                            inter_neuron_counts_layers[j] = pd.DataFrame(inter_neuron_counts_layers[j])
                            inter_neuron_counts_layers[j]['type'] = inter_types_L_t

                            inter_type_counts_layers[j].to_excel(writer_summary, encoding='utf-8', sheet_name=str(i)+'_inter_type_counts_L'+str(j+1))
                            inter_neuron_counts_layers[j].to_excel(writer_summary, encoding='utf-8', sheet_name=str(i)+'_inter_neuron_counts_L'+str(j+1))
                    else:
                        inter_type_counts[i].to_excel(writer_summary, encoding='utf-8', sheet_name=str(i) + '_and_more_inter_type_counts')
                        inter_neuron_counts[i].to_excel(writer_summary, encoding='utf-8', sheet_name=str(i) + '_and_more_inter_neuron_counts')

            # add neuron type to inter_neuron_counts
            inter_types = []
            for i in inter_neuron_counts_all.index:
                inter_types.append(path_inter_all[path_inter_all.bodyId == i].type.unique()[0])
            inter_neuron_counts_all = pd.DataFrame(inter_neuron_counts_all)
            inter_neuron_counts_all['type'] = inter_types

            type_counts_all.to_excel(writer_summary, encoding ='utf-8', sheet_name ='type_counts')
            neuron_counts_all.to_excel(writer_summary, encoding ='utf-8', sheet_name ='neuron_counts')
            inter_type_counts_all.to_excel(writer_summary, encoding ='utf-8', sheet_name ='inter_type_counts')
            inter_neuron_counts_all.to_excel(writer_summary, encoding ='utf-8', sheet_name ='inter_neuron_counts')

        writer_summary.save()

        if timeFlag == 1: tt = t; t = time.time(); print('Summary successfully saved:', t - tt, 's')  # time checkpoint
        print()




        # visualization
        if visualFlag == 1:
            for i, source_i in enumerate(source_ID):
                for j, target_j in enumerate(target_ID):
                    path_ij = fetch_shortest_paths(upstream_bodyId = source_i,
                                                   downstream_bodyId = target_j,
                                                   min_weight = minSnpFind,
                                                   intermediate_criteria = None,
                                                   timeout = maxTime)
                    # plotting
                    s_source = c.fetch_skeleton(body = source_i, format = 'pandas')
                    s_source['bodyId'] = source_i
                    s_source['color'] = bokeh.palettes.Spectral[6][0]
                    s_target = c.fetch_skeleton(body = target_j, format = 'pandas')
                    s_target['bodyId'] = target_j
                    s_target['color'] = bokeh.palettes.Spectral[6][5]

                    path_num = path_ij.path.max()
                    if path_num > maxPathNum - 1: path_num =  maxPathNum - 1;
                    if path_num >= 0:
                        for l in range(path_num+1):
                            path_l = path_ij[path_ij.path == l]
                            all_neurons_id = path_l.bodyId.tolist()
                            all_neurons_type = path_l.type.tolist()

                            p = figure(width=int(figureSize[0] * dpi), height=int(figureSize[1] * dpi),
                                       title='XZ projection of path_' + str(l) +  ' from ' + str(all_neurons_id[0]) + ' to ' + str(all_neurons_id[-1]),
                                       x_range=(0, max_X),
                                       y_range=(max_Y, 0))
                            p.title.text_font_size = str(int(fontSize * 1.5)) + 'pt'
                            # plot skeletons
                            for k, neuron_k in enumerate(all_neurons_id):
                                if k == 0:
                                    s = s_source
                                elif k == len(all_neurons_id)-1:
                                    s = s_target
                                else:
                                    s = c.fetch_skeleton(body = neuron_k, format = 'pandas')
                                    s['bodyId'] = neuron_k
                                    s['color'] = bokeh.palettes.Spectral[6][k]
                                segs = s.merge(s, 'inner',
                                               left_on=['bodyId', 'rowId'],
                                               right_on=['bodyId', 'link'],
                                               suffixes=['_child', '_parent'])
                                p.segment(x0='x_child', x1='x_parent',
                                          y0='z_child', y1='z_parent',
                                          color='color_child',
                                          alpha=0.6,
                                          source=segs,
                                          legend_label='layer_' + str(k) + '_' + str(neuron_k) + '_' + str(all_neurons_type[k]))
                            # plot synapses
                            for k, neuron_k in enumerate(all_neurons_id):
                                if k > 0:
                                    snp_conn = fetch_synapse_connections(source_criteria = NeuronCriteria(bodyId = all_neurons_id[k-1]),
                                                                         target_criteria = NeuronCriteria(bodyId = all_neurons_id[k]),
                                                                         min_total_weight = minSnpFind)
                                    p.scatter(snp_conn['x_pre'], snp_conn['z_pre'], alpha=0.8, size=5, color=bokeh.palettes.Dark2[8][k % 8],
                                              legend_label='synapse between layer_' + str(k-1) + '_' + str(k))
                            p.legend.label_text_font_size = str(int(fontSize * 1.5)) + 'pt'
                            output_file(path + 'path_' + str(l) + '_' + str(all_neurons_id[0]) + 'to' + str(all_neurons_id[-1]) + '.html')  # skeleton of neurons of interest with pre-synapses (to downstream)
                            save(p)
                            if PNG_export == 1:
                                export_png(p, filename = path + 'path_' + str(l) + '_' + str(all_neurons_id[0]) + 'to' + str(all_neurons_id[-1]) + '.png')
                                png_path_conn.append(path + 'path_' + str(l) + '_' + str(all_neurons_id[0]) + 'to' + str(all_neurons_id[-1]) + '.png')
                    if timeFlag == 1: tt = t; t = time.time(); print('Figure successfully saved:', t - tt, 's ---',
                                                                     'source: '+str(i+1)+'/'+str(len(source_ID)),
                                                                     'target: '+str(j+1)+'/'+str(len(target_ID)))  # time checkpoint
            if len(png_path_conn) > 0:
                for img_path in png_path_conn:
                    img = Image.open(img_path).convert('RGB')
                    img.save(img_path)
                with open(path + '_FigSum_paths.pdf', 'wb') as f_paths:
                    f_paths.write(img2pdf.convert(png_path_conn))
            print()

    print()

    # timer
    t = time.time() - t0
    print('Elapsed ', t, ' s')

findpath()
# if __name__ == '__main__':
#     while True :
#         try:
#             findpath()
#         except:
#             print('\n\n\033[33m  error occured and try a new loop  \033[0m\n\n')
#             continue