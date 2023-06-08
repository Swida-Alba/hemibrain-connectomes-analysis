# To visit neuPrint
# Created on Jul, 21, 2020 by KRL
# Last modified on Sep, 26, 2020 by KRL
# E-mail: krleng@pku.edu.cn
#
#
# V1.1     Jul, 24, 2020, by KRL
# Inherited all approaches in V1.0.2
# Plotting connection between each pair of neurons and filtering based on ROI of great weight
# V1.1.1   Sep, 16, 2020, by KRL
# Inherited all properties in V1.0.5 which is independent of anaconda installation
# Adds neuron type information in neuton-neuron connections.
# Adds progress bar.
# V1.1.2   Sep, 22, 2020, by KRL
# Improves performance
# Adds a switch for neuron-neuron connection output, so, it can cover all the functions of FindNeuron V1.0 series.
# Adjusts the plotting order between neurons
# V1.1.3   Sep, 26, 2020, by KRL
# Optimizes input parameters and tells the input type, body ID, or instance automatically.
# Fixes bugs that cannot process adjacent neurons whose type containing apostrofo
# Adds switches for minimal output
# Fixes filename bugs that name upstream neurons as downstream neurons [Facepalm...]
# V1.1.4   Jan, 11, 2021, by KRL
# Fixes errors emerging when find no upstream or downstream types that meet the criteria
# Adds the plots of each neuron in focused neuron set.

# If in Pycharm: (art)
# close Preferences -> Inspections -> Python -> PEP 8 naming convention violation
# close Preferences -> Inspections -> Spelling -> Typo



################################ parameters ##########################################
# appoint the neuron of interest and input parameters
import pandas as pd
import matplotlib.pyplot as plt
database = 'hemibrain:v1.2.1'
directory = R'C:\Users\krlen\Desktop' # absolute directory to save all results
token = '' # please provide your own neuprint token, which can be found at https://neuprint.janelia.org/account
requiredNeurons = ['aMe12']
filename_custom = []

Notes = ''

minSnpFind = 30  # minimal synapse counts for screening upstream and downstream neurons, min synapse number between sets, not neuron-to-neuron synapse number.
minSnpPlot = 10  # minimal synapse counts for plotting skeletons alone, >= minSnpFind
topTypeUp = 10  # the top n of upstream neuron types
topTypeDown = 10 # the top n of downstream neuron types
maxUp = 10  # maximal number of upstream neurons for skeleton sketching
maxDown = 10 # maximal number of downstream neurons for skeleton sketching

timeFlag = 1  # if 1, print time checkpoints, else, nothing, for improving performance and debugging
singleFlag = 1 # if 1, print single neurons. requires visualFlag = 1
upFlag = 0 # if 1, search for upstream neurons. requires ssFlag = 1
downFlag = 0 # if 1, search for downstream neurons. requires ssFlag = 1
ssFlag = 1 # if 1, export skeletons and synapses.
visualFlag = 1  # if 1, print figures. required ssFlag = 1 in part "Synapse" and "Skeleton"
typeFlag = 0 # if 1, export adjacent type information in detail. required visualFlag = 1
N_N_conn = 0 # if 1, export neuron to neuron connections in detail. required visualFlag = 1
PNG_export = 1 # if 1, export skeleton results to png, else only save in html. required for pdf concat

# import matplotlib
# matplotlib.use('TkAgg') # for MAC OS only, if not, comment it (I don't know what if run in Windows)
import datetime
curr_time = datetime.datetime.now()
import warnings
warnings.filterwarnings('ignore')
########################################################################################
# figure properties
fontSize = 14
max_Y = 40000 # the z range of brain to print, not y
min_Y = 0
max_X = 50000 # the x range of brain to print
min_X = 0
figHeight = 9.6 # inch; the bigger, the clearer the figure, the slower the process.
figWidth = figHeight * ((max_X-min_X) / (max_Y-min_Y))
figureSize = [figWidth, figHeight] # inch. first argument is width(x), second argument is height(y)
dpi = 144 # dots per inch, intensity of pixels/dots; no less than 72, more than 300 is not recommended.


parameters = {'dataset': database,
              'date': curr_time,
              'min_synapses': minSnpFind,
              'min_synapses_plotted': minSnpPlot,
              'num_of_top_upstream_types': topTypeUp,
              'num_of_top_downstream_types': topTypeDown,
              'max_upstream_neuron_number': maxUp,
              'max_downstream_neuron_number': maxDown,
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

web_driver = Chrome(executable_path=os.path.join(get_chromedriver_path(), 'chromedriver'), options=options)

from PIL import Image
import img2pdf
plt.rcParams.update({'font.size': fontSize})
from neuprint import *
from bokeh.plotting import figure, output_notebook, output_file, save
import bokeh
from bokeh.io import export_png
parameters = pd.DataFrame.from_dict(parameters, orient = 'index')

# functions
def mkdir(AP):
    folder = os.path.exists(AP)
    if not folder:
        os.makedirs(AP)

if timeFlag == 1: tt = t; t = time.time(); print('Configuration:', t-tt, 's')  # time checkpoint

fig = plt.figure()

# Create a Client to visit neuPrint
c = Client('neuprint.janelia.org',
           dataset = database, # select dataset v1.0.1 or v1.1
           token   = token)
c.fetch_version()
if timeFlag == 1: tt = t; t = time.time(); print('Logged in:', t-tt, 's')  # time checkpoint

# find neurons
for g, required_i in enumerate(requiredNeurons):
    png_general = list()
    print()
    if type(required_i) == list:
        if type(required_i[0]) == int:
            criteria = NeuronCriteria(bodyId = required_i)
            filename = str(required_i[0]) + '_etc'
            print('Processing required bodyId:', required_i)
        elif str(required_i[0])[-1] == '*':
            criteria = NeuronCriteria(type = required_i, regex = True)
            filename = required_i[0][:-2] + '_etc'
            print('Processing required instance:', required_i[0][:-2])
        else:
            criteria = NeuronCriteria(type = required_i)
            filename = str(required_i[0][:]) + '_etc'
            print('Processing required type:', required_i)
    elif type(required_i) == int:
        criteria = NeuronCriteria(bodyId=required_i)
        filename = str(required_i)
        print('Processing required bodyId:', required_i)
    elif str(required_i)[-1] == '*':
        criteria = NeuronCriteria(type = required_i, regex=True)
        filename = required_i[:-2]
        print('Processing required instance:', required_i[:-2])
    else:
        criteria = NeuronCriteria(type = required_i)
        filename = str(required_i)
        print('Processing required type:', required_i)
    # link to NeuronCriteria Documentation: https://connectome-neuprint.github.io/neuprint-python/docs/neuroncriteria.html
    # you can custom the neuron criteria here

    if len(filename_custom) > g:
        if len(filename_custom[g]) != 0:
            filename = filename_custom[g]
    print('Specified name:', filename)

    if directory[0] == '/':
        if directory[-1] != '/':
            directory = directory + '/'
        path = directory + filename + '/' # for MacOS, if Windows, '\\'
        path_info = path + 'info' + '/'
    else:
        if directory[-1] != '\\':
            directory = directory + '\\'
        path = directory + filename + '\\' # for Windows
        path_info = path + 'info' + '\\'
    mkdir(path) # make a folder
    mkdir(path_info)

    # fetch neurons of interest
    neuron_df, ROI_counts_df = fetch_neurons(criteria)  # return neuron properties and per-ROI synapse counts of neurons that meet criteria
    if timeFlag == 1: tt = t; t = time.time(); print('Fetched Neurons:', t-tt, 's')  # time checkpoint

    # fetch inputs and outputs of neurons of interest
    if upFlag == 1:
        neuron_input,  ROI_counts_input = fetch_adjacencies(targets = criteria, min_total_weight = minSnpFind)  # inputs
        mergeProperties_input = merge_neuron_properties(neuron_input, ROI_counts_input, ['type', 'instance'])  # merge input neurons and their properties
        mergeProperties_input = mergeProperties_input.sort_values('weight', ascending = False)
        ROI_counts_input = ROI_counts_input.sort_values('weight', ascending = False)
        if timeFlag == 1: tt = t; t = time.time(); print('Fetched Inputs:', t - tt, 's')  # time checkpoint
    if downFlag == 1:
        neuron_output, ROI_counts_output = fetch_adjacencies(sources = criteria, min_total_weight = minSnpFind)  # outputs
        mergeProperties_output = merge_neuron_properties(neuron_output, ROI_counts_output, ['type', 'instance'])  # merge output neurons and their properties
        mergeProperties_output = mergeProperties_output.sort_values('weight', ascending = False)
        ROI_counts_output = ROI_counts_output.sort_values('weight', ascending = False)
        if timeFlag == 1: tt = t; t = time.time(); print('Fetched Outputs:', t-tt, 's')  # time checkpoint

    # # connection matrix
    # matrix_input = connection_table_to_matrix(mergeProperties_input, 'bodyId', sort_by = 'type')
    # mat_input = matrix_input.iloc
    # matrix_output = connection_table_to_matrix(mergeProperties_output, 'bodyId', sort_by = 'type')
    # mat_output = matrix_output.iloc
    # print(mat_input[:10, :10])
    # print(mat_output)
    # pickle.dump(mat_input, open('/Users/apple/Desktop/mat_input.pkl','wb'))

    # Synapse
    if ssFlag == 1:
        # link to SynapseCriteria documentation: https://connectome-neuprint.github.io/neuprint-python/docs/synapsecriteria.html
        snp_pre = fetch_synapses(criteria, SynapseCriteria(type = 'pre')) # outputs
        snp_post = fetch_synapses(criteria, SynapseCriteria(type = 'post')) # inputs

        if visualFlag == 1:
            # plot pre-synapse
            figPre = plt.figure(figsize = figureSize, dpi = dpi)
            plt.ylim([min_Y,max_Y])
            plt.xlim([min_X,max_X])
            plt.gca().invert_yaxis()
            plt.scatter(snp_pre['x'], snp_pre['z'], alpha = 0.2, s = 3, c = color[1], label = 'pre')
            plt.legend()
            plt.title('XZ projection of ' + filename + '\'s pre-synapses (output)')
            plt.xlabel('X direction')
            plt.ylabel('Z direction')
            filename_temp = path_info + filename + '_snpPre.png'
            plt.savefig(filename_temp)
            png_general.append(filename_temp)

            # plot post-synapse
            figPost = plt.figure(figsize = figureSize, dpi = dpi)
            plt.ylim([min_Y,max_Y])
            plt.xlim([min_X,max_X])
            plt.gca().invert_yaxis()
            plt.scatter(snp_post['x'], snp_post['z'], alpha = 0.2, s = 3, c = color[2], label = 'post')
            plt.legend()
            plt.title('XZ projection of ' + filename + '\'s post-synapses (input)')
            plt.xlabel('X direction')
            plt.ylabel('Z direction')
            filename_temp = path_info + filename + '_snpPost.png'
            plt.savefig(filename_temp)
            png_general.append(filename_temp)

            # plot pre- and post-synapses
            figSnp = plt.figure(figsize = figureSize, dpi = dpi)
            plt.ylim([min_Y,max_Y])
            plt.xlim([min_X,max_X])
            plt.gca().invert_yaxis()
            plt.scatter(snp_pre['x'], snp_pre['z'], alpha = 0.2, s = 3, c = color[1], label = 'pre')
            plt.scatter(snp_post['x'], snp_post['z'], alpha = 0.2, s = 3, c = color[2], label = 'post')
            plt.legend()
            plt.title('XZ projection of ' + filename + '\'s synapses')
            plt.xlabel('X direction')
            plt.ylabel('Z direction')
            filename_temp = path_info + filename + '_snp.png'
            plt.savefig(filename_temp)
            png_general.append(filename_temp)

        if timeFlag == 1: tt = t; t = time.time(); print('Fetched Synapses:', t-tt, 's')  # time checkpoint

        # Synapse Connection, this part is slow
        # outputs
        if downFlag == 1:
            synPre_conn = fetch_synapse_connections(source_criteria = criteria, synapse_criteria = SynapseCriteria(type = 'pre', primary_only = True), min_total_weight = minSnpFind)
            # link to fetch_synapse_connections documentation: https://connectome-neuprint.github.io/neuprint-python/docs/queries.html#neuprint.queries.fetch_synapse_connections
            body_post = synPre_conn['bodyId_post'].unique()
            if len(body_post) > 0:
                postNeuron, _ = fetch_neurons(body_post)
                synPre_conn = merge_neuron_properties(postNeuron, synPre_conn, 'type')
                topNumPre_counts = synPre_conn['type_post'].value_counts().head(topTypeDown)
                print('Top post-synaptic neuron types:')
                print(topNumPre_counts)

                # plotting
                if visualFlag == 1:
                    if topTypeDown % 20 == 0:
                        colormap_pre = dict(zip(topNumPre_counts.index, bokeh.palettes.Category20[20][:20]))
                    else:
                        colormap_pre = dict(zip(topNumPre_counts.index, bokeh.palettes.Category20[20][:(topTypeDown % 20)]))
                    points_pre = synPre_conn.query('type_post in @topNumPre_counts.index').copy()
                    points_pre['color'] = points_pre['type_post'].map(colormap_pre)
                    figTopPost = plt.figure(figsize=figureSize, dpi=dpi)
                    plt.ylim([min_Y,max_Y])
                    plt.xlim([min_X,max_X])
                    plt.gca().invert_yaxis()
                    plt.scatter(points_pre['x_post'], points_pre['z_post'], color = points_pre['color'], alpha = 0.5, s = 3)
                    markers_pre = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in colormap_pre.values()] # MAGIC!
                    plt.legend(markers_pre, colormap_pre.keys(), numpoints=1)
                    plt.title('Top-' + str(topTypeDown) + ' post-synaptic neuron types (outputs) of ' + filename)
                    plt.xlabel('X direction')
                    plt.ylabel('Z direction')
                    filename_temp = path_info + filename + '_TopSnpPre.png'
                    plt.savefig(filename_temp)
                    png_general.append(filename_temp)

            if timeFlag == 1: tt = t; t = time.time(); print('Fetched Post-Synaptic Connections:', t-tt, 's')  # time checkpoint

        # inputs
        if upFlag == 1:
            synPost_conn = fetch_synapse_connections(target_criteria = criteria, synapse_criteria = SynapseCriteria(type = 'post', primary_only = True), min_total_weight = minSnpFind)
            body_pre = synPost_conn['bodyId_pre'].unique()
            if len(body_pre) > 0:
                preNeuron, _ = fetch_neurons(body_pre)
                synPost_conn = merge_neuron_properties(preNeuron, synPost_conn, 'type')
                topNumPost_counts = synPost_conn['type_pre'].value_counts().head(topTypeUp)
                print('Top pre-synaptic neuron types:')
                print(topNumPost_counts)

                # plotting
                if visualFlag == 1:
                    if topTypeUp % 20 == 0:
                        colormap_post = dict(zip(topNumPost_counts.index, bokeh.palettes.Category20[20][:20]))
                    else:
                        colormap_post = dict(zip(topNumPost_counts.index, bokeh.palettes.Category20[20][:(topTypeUp % 20)]))
                    points_post = synPost_conn.query('type_pre in @topNumPost_counts.index').copy()
                    points_post['color'] = points_post['type_pre'].map(colormap_post)
                    figTopPre = plt.figure(figsize=figureSize, dpi=dpi)
                    plt.ylim([min_Y,max_Y])
                    plt.xlim([min_X,max_X])
                    plt.gca().invert_yaxis()
                    plt.scatter(points_post['x_pre'], points_post['z_pre'], color=points_post['color'], alpha = 0.5, s = 3)
                    markers_post = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in colormap_post.values()]
                    plt.legend(markers_post, colormap_post.keys(), numpoints=1)
                    plt.title('Top-' + str(topTypeUp) + ' pre-synaptic neuron types (inputs) of ' + filename)
                    plt.xlabel('X direction')
                    plt.ylabel('Z direction')
                    filename_temp = path_info + filename + '_TopSnpPost.png'
                    plt.savefig(filename_temp)
                    png_general.append(filename_temp)

            if timeFlag == 1: tt = t; t = time.time(); print('Fetched Pre-Synaptic Connections:', t-tt, 's')  # time checkpoint

        # Skeleton
        if visualFlag == 1:
            # Skeletons of focused neurons themselves
            p = figure(width=int(figureSize[0]*dpi), height=int(figureSize[1]*dpi),
                       title = 'XZ projection of '+filename,
                       x_range=(min_X, max_X),
                       y_range=(max_Y, min_Y)) # flip axis
            p.title.text_font_size = str(int(fontSize*1.5))+'pt'
            # p.y_range.flipped = True
            for j, bodyId in enumerate(list(neuron_df['bodyId'])):
                s = c.fetch_skeleton(body=bodyId, format='pandas')
                s['bodyId'] = bodyId
                s['color'] = bokeh.palettes.Category20b[20][j % 20]
                # print(s)
                segs = s.merge(s, 'inner',
                               left_on=['bodyId', 'rowId'],
                               right_on=['bodyId', 'link'],
                               suffixes=['_child', '_parent'])
                p.segment(x0='x_child', x1='x_parent',
                          y0='z_child', y1='z_parent',
                          color='color_child',
                          alpha=0.8,
                          source=segs,
                          legend_label=str(bodyId))

            # p.scatter(snp_pre['x'], snp_pre['z'], alpha=0.2, size=3, color=color[1], legend_label='pre(output)')
            # p.scatter(snp_post['x'], snp_post['z'], alpha=0.2, size=3, color=color[2], legend_label='post(input)')
            p.legend.label_text_font_size = str(int(fontSize*1.5)) + 'pt'
            filename_temp = path_info + filename + '_skeleton'
            output_file(filename_temp+'.html')  # skeleton of neurons of interest with pre-synapses (to downstream)
            save(p)
            if timeFlag == 1: tt = t; t = time.time(); print('Fetched Skeletons:', t-tt, 's')  # time checkpoint
            if PNG_export == 1:
                # print('Exporting Skeleonts...')
                export_png(p, filename=filename_temp+'.png')
                png_general.append(filename_temp+'.png')
                # if timeFlag == 1: tt = t; t = time.time(); print('Exported Skeletons:', t-tt, 's')  # time checkpoint

            # Skeletons of downstream neurons
            if downFlag == 1:
                if len(body_post) > 0:
                    p = figure(width=int(figureSize[0]*dpi), height=int(figureSize[1]*dpi),
                               title = 'XZ projection of '+filename+'\'s top-' + str(maxDown) + ' downstream neurons',
                               x_range=(min_X, max_X),
                               y_range=(max_Y, min_Y))
                    p.title.text_font_size = str(int(fontSize * 1.5)) + 'pt'
                    # p.y_range.flipped = True
                    synPre_simple_conn = fetch_simple_connections(upstream_criteria=criteria, min_weight=minSnpFind)  # sorted by weight
                    for j, bodyId in enumerate(synPre_simple_conn['bodyId_post'].unique()[:maxDown]):
                        s = c.fetch_skeleton(body=bodyId, format='pandas')
                        s['bodyId'] = bodyId
                        s['color'] = bokeh.palettes.Category20b[20][j % 20]
                        down_type_t = synPre_simple_conn[synPre_simple_conn.bodyId_post == bodyId].type_post.unique()[0]
                        segs = s.merge(s, 'inner',
                                       left_on=['bodyId', 'rowId'],
                                       right_on=['bodyId', 'link'],
                                       suffixes=['_child', '_parent'])
                        p.segment(x0='x_child', x1='x_parent',
                                  y0='z_child', y1='z_parent',
                                  color='color_child',
                                  alpha=0.8,
                                  source=segs,
                                  legend_label='post_'+str(bodyId)+'_'+str(down_type_t))
                    p.scatter(points_pre['x_post'], points_pre['z_post'], alpha = 0.5, color=points_pre['color'])
                    p.legend.label_text_font_size = str(int(fontSize * 1.5)) + 'pt'
                    filename_temp = path_info + filename + '_skeleton_downstream'
                    output_file(filename_temp+'.html')  # skeleton of neurons of interest with pre-synapses (to downstream)
                    save(p)
                    if timeFlag == 1: tt = t; t = time.time(); print('Fetched Downstream Skeletons:', t-tt, 's')  # time checkpoint
                    if PNG_export == 1:
                        # print('Exporting Downstream Skeleonts...')
                        export_png(p, filename=filename_temp+'.png')
                        png_general.append(filename_temp+'.png')
                        # if timeFlag == 1: tt = t; t = time.time(); print('Exported Downstream Skeletons:', t-tt, 's')  # time checkpoint

            # Skeleton of upstream neurons
            if upFlag == 1:
                if len(body_pre) > 0:
                    p = figure(width=int(figureSize[0]*dpi), height=int(figureSize[1]*dpi),
                               title = 'XZ projection of '+filename+'\'s top-' + str(maxUp) + ' upstream neurons',
                               x_range=(min_X, max_X),
                               y_range=(max_Y, min_Y))
                    p.title.text_font_size = str(int(fontSize * 1.5)) + 'pt'
                    # p.y_range.flipped = True
                    synPost_simple_conn = fetch_simple_connections(downstream_criteria=criteria, min_weight=minSnpFind)  # sorted by weight
                    for j, bodyId in enumerate(synPost_simple_conn['bodyId_pre'].unique()[:maxUp]):  # plot top-weight connections
                        s = c.fetch_skeleton(body=bodyId, format='pandas')
                        s['bodyId'] = bodyId
                        s['color'] = bokeh.palettes.Category20b[20][j % 20]
                        up_type_t = synPost_simple_conn[synPost_simple_conn.bodyId_pre == bodyId].type_pre.unique()[0]
                        segs = s.merge(s, 'inner',
                                       left_on=['bodyId', 'rowId'],
                                       right_on=['bodyId', 'link'],
                                       suffixes=['_child', '_parent'])
                        p.segment(x0='x_child', x1='x_parent',
                                  y0='z_child', y1='z_parent',
                                  color='color_child',
                                  alpha=0.8,
                                  source=segs,
                                  legend_label='pre_' + str(bodyId) + '_' + str(up_type_t))
                    p.scatter(points_post['x_pre'], points_post['z_pre'], alpha = 0.5, color = points_post['color'])
                    p.legend.label_text_font_size = str(int(fontSize * 1.5)) + 'pt'
                    filename_temp = path_info + filename + '_skeleton_upstream'
                    output_file(filename_temp+'.html')  # skeleton of neurons of interest with pre-synapses (to downstream)
                    save(p)
                    if timeFlag == 1: tt = t; t = time.time(); print('Fetched Upstream Skeletons:', t-tt, 's')  # time checkpoint
                    if PNG_export == 1:
                        # print('Exporting Upstream Skeleonts...')
                        export_png(p, filename=filename_temp+'.png')
                        png_general.append(filename_temp+'.png')
                        # if timeFlag == 1: tt = t; t = time.time(); print('Exported Upstream Skeletons:', t-tt, 's')  # time checkpoint

    # save table data in an .xlsx file
    writer_basic = pd.ExcelWriter(path + filename + '.xlsx')
    parameters.to_excel(writer_basic, encoding ='utf-8', sheet_name =filename + '_parameters') # input parameters for filtering

    neuron_df.to_excel(writer_basic, encoding ='utf-8', sheet_name =filename + '_neurons') # found neurons of interest
    ROI_counts_df.to_excel(writer_basic, encoding ='utf-8', sheet_name =filename + '_ROI_counts')

    if upFlag == 1:
        # if len(body_pre) > 0:
        #     neuron_input.to_excel(writer_basic, encoding ='utf-8', sheet_name =filename + '_inputs(post)') # upstream neurons
        #     ROI_counts_input.to_excel(writer_basic, encoding ='utf-8', sheet_name =filename + '_in_ROI_counts')
        #     mergeProperties_input.to_excel(writer_basic, encoding ='utf-8', sheet_name =filename + '_in_merge')
        neuron_input.to_excel(writer_basic, encoding='utf-8', sheet_name=filename + '_inputs(post)')  # upstream neurons
        ROI_counts_input.to_excel(writer_basic, encoding ='utf-8', sheet_name =filename + '_in_ROI_counts')
        mergeProperties_input.to_excel(writer_basic, encoding ='utf-8', sheet_name =filename + '_in_merge')
    if downFlag == 1:
        # if len(body_post) > 0:
        #     neuron_output.to_excel(writer_basic, encoding ='utf-8', sheet_name =filename + '_outputs(pre)') # downstream neurons
        #     ROI_counts_output.to_excel(writer_basic, encoding ='utf-8', sheet_name =filename + '_out_ROI_counts')
        #     mergeProperties_output.to_excel(writer_basic, encoding ='utf-8', sheet_name =filename + '_out_merge')
        neuron_output.to_excel(writer_basic, encoding='utf-8',sheet_name=filename + '_outputs(pre)')  # downstream neurons
        ROI_counts_output.to_excel(writer_basic, encoding='utf-8', sheet_name=filename + '_out_ROI_counts')
        mergeProperties_output.to_excel(writer_basic, encoding='utf-8', sheet_name=filename + '_out_merge')

    if ssFlag == 1:
        if upFlag == 1:
            if len(body_pre) > 0:
                # snp_post_1 = snp_post.iloc[0:1048000,:]
                # snp_post_2 = snp_post.iloc[1048000:, :]
                # snp_post_1.to_excel(writer_basic, encoding ='utf-8', sheet_name =filename + '_snpPost(inputs)_1')
                # snp_post_2.to_excel(writer_basic, encoding='utf-8', sheet_name=filename + '_snpPost(inputs)_2')
                snp_post.to_excel(writer_basic, encoding ='utf-8', sheet_name =filename + '_snpPost(inputs)') # synapses formed with upstream neurons
                topNumPost_counts.to_excel(writer_basic, encoding='utf-8', sheet_name=filename + '_TopSnpPost(inputs)')  # predominant upstream neuron types
        if downFlag == 1:
            if len(body_post) > 0:
                snp_pre.to_excel(writer_basic, encoding ='utf-8', sheet_name =filename + '_snpPre(outputs)') # synapses formed with downstream neurons
                topNumPre_counts.to_excel(writer_basic, encoding='utf-8', sheet_name=filename + '_TopSnpPre(outputs)')  # predominant downstream neuron types

    writer_basic.save()
    if timeFlag == 1: tt = t; t = time.time(); print('Writing File:', t-tt, 's')  # time checkpoint

    # concatenate png figures and save in a pdf file
    if len(png_general) > 0:
        for img_path in png_general:
            img = Image.open(img_path).convert('RGB')
            img.save(img_path)
        with open(path + filename + '_Abstract.pdf', 'wb') as f_sum:
            f_sum.write(img2pdf.convert(png_general))


    if visualFlag == 1:
        neuron_df = neuron_df.sort_values('type', ascending=True)
        neuron_ID = list(neuron_df['bodyId'])
        # Single neurons that meet the search criteria
        if singleFlag == 1:
            png_requiredNeurons = []
            if directory[0] == '/':
                newpath = path + 'single_neuorns' + '/'  # for MacOS, if Windows, '\\'
            else:
                newpath = path + 'single_neurons' + '\\'  # for Windows
            mkdir(newpath)  # make a folder
            for i, neuron_i in enumerate(neuron_ID):
                type_t = neuron_df[neuron_df.bodyId == neuron_i].type.unique()[0]
                instance_t = neuron_df[neuron_df.bodyId == neuron_i].instance.unique()[0]
                if instance_t is None:
                    filename_t = str(neuron_i) + '_None'
                else:
                    filename_t = str(neuron_i) + '_' + instance_t
                p = figure(width=int(figureSize[0] * dpi), height=int(figureSize[1] * dpi),
                           title='XZ projection of ' + filename_t,
                           x_range=(min_X, max_X),
                           y_range=(max_Y, min_Y))  # flip axis
                p.title.text_font_size = str(int(fontSize * 1.5)) + 'pt'

                s = c.fetch_skeleton(body=neuron_i, format='pandas')
                s['bodyId'] = neuron_i
                s['color'] = bokeh.palettes.Category20b[20][0]
                # print(s)
                segs = s.merge(s, 'inner',
                               left_on=['bodyId', 'rowId'],
                               right_on=['bodyId', 'link'],
                               suffixes=['_child', '_parent'])
                p.segment(x0='x_child', x1='x_parent',
                          y0='z_child', y1='z_parent',
                          color='color_child',
                          alpha=0.8,
                          source=segs,
                          legend_label=str(filename_t))

                p.legend.label_text_font_size = str(int(fontSize * 1.5)) + 'pt'
                filename_temp = newpath + str(neuron_i) + '_skeleton'
                output_file(filename_temp + '.html')  # skeleton of neurons of interest with pre-synapses (to downstream)
                save(p)
                print('\rFetched Skeletons of ', filename_t,
                      str(round(100 * (i + 1) / (len(neuron_ID)), 2)) + '%',
                      end='')
                if PNG_export == 1:
                    # print('Exporting Skeleonts...')
                    export_png(p, filename=filename_temp + '.png')
                    png_requiredNeurons.append(filename_temp + '.png')
                    # if timeFlag == 1: tt = t; t = time.time(); print('Exported Skeletons:', t-tt, 's')  # time checkpoint
            print()
            if len(png_requiredNeurons) > 0:
                for img_path in png_requiredNeurons:
                    img = Image.open(img_path).convert('RGB')
                    img.save(img_path)
                with open(path + '_' + filename + '_FigSum.pdf', 'wb') as f_requiredNeurons:
                    f_requiredNeurons.write(img2pdf.convert(png_requiredNeurons))


        # Adjacent Neuron Types of Great Weight
        if typeFlag == 1:
            if (downFlag == 1) and (len(body_post) > 0):
                topOutput_types = list(topNumPre_counts.keys())
            if (upFlag == 1) and (len(body_pre) > 0):
                topInput_types = list(topNumPost_counts.keys())

            # downstream neuron types
            if (downFlag == 1) and (len(body_post) > 0):
                png_downType = list()
                if directory[0] == '/':
                    newpath = path + 'output_type' + '/'  # for MacOS, if Windows, '\\'
                else:
                    newpath = path + 'output_type' + '\\'  # for Windows
                mkdir(newpath)  # make a folder
                for i,type_i in enumerate(topOutput_types):
                    # type_i_ids = synPre_conn[synPre_conn.type_post == type_i].bodyId_post.unique().tolist() # bodyIds of type_i neurons.
                    # out_cri = NeuronCriteria(bodyId = type_i_ids)
                    out_cri = type_i
                    out_conn = fetch_synapse_connections(source_criteria = criteria, target_criteria = out_cri, min_total_weight = minSnpFind) # fetch synapses
                    # fetch skeletons
                    p = figure(width=int(figureSize[0]*dpi), height=int(figureSize[1]*dpi),
                               title = 'XZ projection of '+filename+'_to_'+type_i,
                               x_range=(min_X, max_X),
                               y_range=(max_Y, min_Y))
                    p.title.text_font_size = str(int(fontSize * 1.5)) + 'pt'
                    # p.y_range.flipped = True
                    # plot pre-synaptic neurons
                    for j, bodyId in enumerate(out_conn['bodyId_pre'].unique()):
                        s = c.fetch_skeleton(body=bodyId, format='pandas')
                        s['bodyId'] = bodyId
                        s['color'] = bokeh.palettes.Blues[j % 7 + 3][0]
                        # print(s)
                        segs = s.merge(s, 'inner',
                                       left_on=['bodyId', 'rowId'],
                                       right_on=['bodyId', 'link'],
                                       suffixes=['_child', '_parent'])
                        p.segment(x0='x_child', x1='x_parent',
                                  y0='z_child', y1='z_parent',
                                  color='color_child',
                                  alpha=1,
                                  source=segs,
                                  legend_label='pre_'+str(bodyId))
                    # plot post-synaptic neurons
                    for j, bodyId in enumerate(out_conn['bodyId_post'].unique()):
                        s = c.fetch_skeleton(body=bodyId, format='pandas')
                        s['bodyId'] = bodyId
                        s['color'] = bokeh.palettes.Oranges[j % 7 + 3][0]
                        # print(s)
                        segs = s.merge(s, 'inner',
                                       left_on=['bodyId', 'rowId'],
                                       right_on=['bodyId', 'link'],
                                       suffixes=['_child', '_parent'])
                        p.segment(x0='x_child', x1='x_parent',
                                  y0='z_child', y1='z_parent',
                                  color='color_child',
                                  alpha=0.4,
                                  source=segs,
                                  legend_label='post_' + str(bodyId))
                    # plot synapses
                    p.scatter(out_conn['x_pre'], out_conn['z_pre'], alpha=0.8, size=3, color=bokeh.palettes.OrRd9[0], legend_label='synapse(output)')
                    p.legend.label_text_font_size = str(int(fontSize * 1.5)) + 'pt'
                    if type_i.find('/') != -1:
                        type_i = type_i.replace('/','-')
                    output_file(newpath + str(i+1) + '_' + type_i + '_skeleton.html')  # skeleton of neurons of interest with pre-synapses (to downstream)
                    save(p)
                    if timeFlag == 1: tt = t; t = time.time(); print('Fetched', type_i, 'Skeletons (output):', t-tt, 's ---', 'type: '+str(i+1)+'/'+str(len(topOutput_types)))  # time checkpoint
                    if PNG_export == 1:
                        # print('Exporting', type_i, 'Skeleonts (output)...')
                        png_downType.append(newpath + str(i+1) + '_' + type_i + '_skeleton.png')
                        export_png(p, filename = newpath + str(i+1) + '_' + type_i + '_skeleton.png')
                        # if timeFlag == 1:  tt = t; t = time.time(); print('Exported', type_i, 'Skeleonts (output):', t-tt, 's')  # time checkpoint
                if len(png_downType) > 0:
                    for img_path in png_downType:
                        img = Image.open(img_path).convert('RGB')
                        img.save(img_path)
                    with open(path + filename + '_downstreamType.pdf', 'wb') as f_downType:
                        f_downType.write(img2pdf.convert(png_downType))

            # upstream neuron types
            if (upFlag == 1) and (len(body_pre) > 0):
                png_upType = list()
                if directory[0] == '/':
                    newpath = path + 'input_type' + '/'  # for MacOS, if Windows, '\\'
                else:
                    newpath = path + 'input_type' + '\\'  # for Windows
                mkdir(newpath)  # make a folder
                for i, type_i in enumerate(topInput_types):
                    # type_i_ids = synPost_conn[synPost_conn.type_pre == type_i].bodyId_pre.unique().tolist()
                    # in_cri = NeuronCriteria(bodyId = type_i_ids)
                    in_cri = type_i
                    in_conn = fetch_synapse_connections(source_criteria = in_cri, target_criteria = criteria, min_total_weight = minSnpFind) # fetch synapses
                    # fetch skeletons
                    p = figure(width=int(figureSize[0]*dpi), height=int(figureSize[1]*dpi),
                               title = 'XZ projection of '+type_i+'_to_'+filename,
                               x_range=(min_X, max_X),
                               y_range=(max_Y, min_Y))
                    p.title.text_font_size = str(int(fontSize * 1.5)) + 'pt'
                    # p.y_range.flipped = True
                    # plot post-synaptic neurons
                    for j, bodyId in enumerate(in_conn['bodyId_post'].unique()):
                        s = c.fetch_skeleton(body=bodyId, format='pandas')
                        s['bodyId'] = bodyId
                        s['color'] = bokeh.palettes.Blues[j % 7 + 3][0]
                        segs = s.merge(s, 'inner',
                                       left_on=['bodyId', 'rowId'],
                                       right_on=['bodyId', 'link'],
                                       suffixes=['_child', '_parent'])
                        p.segment(x0='x_child', x1='x_parent',
                                  y0='z_child', y1='z_parent',
                                  color='color_child',
                                  alpha=0.4,
                                  source=segs,
                                  legend_label='post_'+str(bodyId))
                    # plot pre-synaptic neurons
                    for j, bodyId in enumerate(in_conn['bodyId_pre'].unique()):
                        s = c.fetch_skeleton(body=bodyId, format='pandas')
                        s['bodyId'] = bodyId
                        s['color'] = bokeh.palettes.Oranges[j % 7 + 3][0]
                        segs = s.merge(s, 'inner',
                                       left_on=['bodyId', 'rowId'],
                                       right_on=['bodyId', 'link'],
                                       suffixes=['_child', '_parent'])
                        p.segment(x0='x_child', x1='x_parent',
                                  y0='z_child', y1='z_parent',
                                  color='color_child',
                                  alpha=1,
                                  source=segs,
                                  legend_label='pre_'+str(bodyId))
                    # plot synapses
                    p.scatter(in_conn['x_post'], in_conn['z_post'], alpha=0.8, size=3, color=bokeh.palettes.OrRd9[0], legend_label='synapse(input)')
                    p.legend.label_text_font_size = str(int(fontSize * 1.5)) + 'pt'
                    if type_i.find('/') != -1:
                        type_i = type_i.replace('/','-')
                    output_file(newpath + str(i+1) + '_' + type_i + '_skeleton.html')  # skeleton of neurons of interest with pre-synapses (to downstream)
                    save(p)
                    if timeFlag == 1: tt = t; t = time.time(); print('Fetched', type_i, 'Skeletons (input):', t-tt, 's --- ', 'type: '+str(i+1)+'/'+str(len(topInput_types)))  # time checkpoint
                    if PNG_export == 1:
                        # print('Exporting', type_i, 'Skeleonts (input)...')
                        export_png(p, filename = newpath + str(i+1) + '_' + type_i + '_skeleton.png')
                        png_upType.append(newpath + str(i+1) + '_' + type_i + '_skeleton.png')
                        # if timeFlag == 1: tt = t; t = time.time(); print('Exported', type_i, 'Skeleonts (input):', t-tt, 's')  # time checkpoint
                if len(png_upType) > 0:
                    for img_path in png_upType:
                        img = Image.open(img_path).convert('RGB')
                        img.save(img_path)
                    with open(path + filename + '_upstreamType.pdf', 'wb') as f_upType:
                        f_upType.write(img2pdf.convert(png_upType))

        # Adjacent Neurons of Great Weight -- Synapses between Two Neurons
        if N_N_conn == 1:
            # focused neurons and downstream connections
            if (downFlag == 1) and (len(body_post) > 0):
                png_downNeuron = list()
                if directory[0] == '/':
                    newpath = path + 'output_neuron' + '/'  # for MacOS, if Windows, '\\'
                else:
                    newpath = path + 'output_neuron' + '\\'  # for Windows
                mkdir(newpath)  # make a folder
                for neuron_count, neuron_i in enumerate(neuron_ID):
                    simple_conn = fetch_simple_connections(upstream_criteria = NeuronCriteria(bodyId = neuron_i),
                                                           min_weight = minSnpPlot)
                    s_source = c.fetch_skeleton(body=neuron_i, format='pandas')
                    s_source['bodyId'] = neuron_i
                    s_source['color'] = bokeh.palettes.Blues[3][0]

                    down_ID = list(simple_conn['bodyId_post'].unique())
                    for down_count, down_i in enumerate(down_ID):
                        down_conn = fetch_synapse_connections(source_criteria = NeuronCriteria(bodyId = neuron_i),
                                                              target_criteria = NeuronCriteria(bodyId = down_i))
                        down_type = simple_conn[simple_conn.bodyId_post == down_i].type_post.tolist()
                        # fetch skeletons
                        p = figure(width=int(figureSize[0]*dpi), height=int(figureSize[1]*dpi),
                                   title='XZ projection of ' + str(neuron_i) + '_to_' + str(down_i) + '_(' + str(down_type[0]) + ')',
                                   x_range=(min_X, max_X),
                                   y_range=(max_Y, min_Y))
                        p.title.text_font_size = str(int(fontSize * 1.5)) + 'pt'
                        # p.y_range.flipped = True
                        # plot pre-synaptic neurons
                        s = s_source
                        segs = s.merge(s, 'inner',
                                       left_on=['bodyId', 'rowId'],
                                       right_on=['bodyId', 'link'],
                                       suffixes=['_child', '_parent'])
                        p.segment(x0='x_child', x1='x_parent',
                                  y0='z_child', y1='z_parent',
                                  color='color_child',
                                  alpha=1,
                                  source=segs,
                                  legend_label='pre_' + str(neuron_i))
                        # plot post-synaptic neurons
                        s = c.fetch_skeleton(body=down_i, format='pandas')
                        s['bodyId'] = down_i
                        s['color'] = bokeh.palettes.Oranges[3][0]
                        segs = s.merge(s, 'inner',
                                       left_on=['bodyId', 'rowId'],
                                       right_on=['bodyId', 'link'],
                                       suffixes=['_child', '_parent'])
                        p.segment(x0='x_child', x1='x_parent',
                                  y0='z_child', y1='z_parent',
                                  color='color_child',
                                  alpha=0.4,
                                  source=segs,
                                  legend_label='post_' + str(down_i))
                        # plot synapses
                        p.scatter(down_conn['x_pre'], down_conn['z_pre'], alpha=0.8, size=3, color=bokeh.palettes.OrRd9[0],
                                  legend_label='synapse(output)')
                        p.legend.label_text_font_size = str(int(fontSize * 1.5)) + 'pt'
                        output_file(newpath + str(neuron_i) + 'to' + str(down_i) + '_skeleton.html')  # skeleton of neurons of interest with pre-synapses (to downstream)
                        save(p)
                        if timeFlag == 1: tt = t; t = time.time(); print('Fetched', str(neuron_i),'to', str(down_i), 'Skeletons (output):', t-tt, 's ---',
                                                                         'neuron: '+str(neuron_count+1)+'/'+str(len(neuron_ID)),
                                                                         'downstream: '+str(down_count+1)+'/'+str(len(down_ID)))
                        if PNG_export == 1:
                            # print('Exporting', str(neuron_i), 'to', str(down_i), 'Skeletons (output)...')
                            export_png(p, filename =newpath + str(neuron_i) + 'to' + str(down_i) + '_skeleton.png')
                            png_downNeuron.append(newpath + str(neuron_i) + 'to' + str(down_i) + '_skeleton.png')
                            # if timeFlag == 1: tt = t; t = time.time(); print('Exported', str(neuron_i), 'to', str(down_i), 'Skeleonts (output):', t-tt, 's')  # time checkpoint
                if len(png_downNeuron) > 0:
                    for img_path in png_downNeuron:
                        img = Image.open(img_path).convert('RGB')
                        img.save(img_path)
                    with open(path + filename + '_downstreamNeuron.pdf', 'wb') as f_downNeuron:
                        f_downNeuron.write(img2pdf.convert(png_downNeuron))

            # focused neurons and upstream connections
            if (upFlag == 1) and (len(body_pre) > 0):
                png_upNeuron = list()
                if directory[0] == '/':
                    newpath = path + 'input_neuron' + '/'  # for MacOS, if Windows, '\\'
                else:
                    newpath = path + 'input_neuron' + '\\'  # for Windows
                mkdir(newpath)  # make a folder
                for neuron_count, neuron_i in enumerate(neuron_ID):
                    simple_conn = fetch_simple_connections(downstream_criteria=NeuronCriteria(bodyId=neuron_i),
                                                           min_weight=minSnpPlot)
                    s_target = c.fetch_skeleton(body=neuron_i, format='pandas')
                    s_target['bodyId'] = neuron_i
                    s_target['color'] = bokeh.palettes.Blues[3][0]

                    up_ID = list(simple_conn['bodyId_pre'].unique())
                    for up_count, up_i in enumerate(up_ID):
                        up_conn = fetch_synapse_connections(target_criteria=NeuronCriteria(bodyId=neuron_i),
                                                            source_criteria=NeuronCriteria(bodyId=up_i))
                        up_type = simple_conn[simple_conn.bodyId_pre == up_i].type_pre.tolist()
                        # fetch skeletons
                        p = figure(width=int(figureSize[0]*dpi), height=int(figureSize[1]*dpi),
                                   title='XZ projection of ' + str(up_i) + '_(' + str(up_type[0]) + ')' + '_to_' + str(neuron_i),
                                   x_range=(min_X, max_X),
                                   y_range=(max_Y, min_Y))
                        p.title.text_font_size = str(int(fontSize * 1.5)) + 'pt'
                        # p.y_range.flipped = True
                        # plot post-synaptic neurons
                        s = s_target
                        segs = s.merge(s, 'inner',
                                       left_on=['bodyId', 'rowId'],
                                       right_on=['bodyId', 'link'],
                                       suffixes=['_child', '_parent'])
                        p.segment(x0='x_child', x1='x_parent',
                                  y0='z_child', y1='z_parent',
                                  color='color_child',
                                  alpha=0.4,
                                  source=segs,
                                  legend_label='post_' + str(neuron_i))
                        # plot pre-synaptic neurons
                        s = c.fetch_skeleton(body=up_i, format='pandas')
                        s['bodyId'] = up_i
                        s['color'] = bokeh.palettes.Oranges[3][0]
                        segs = s.merge(s, 'inner',
                                       left_on=['bodyId', 'rowId'],
                                       right_on=['bodyId', 'link'],
                                       suffixes=['_child', '_parent'])
                        p.segment(x0='x_child', x1='x_parent',
                                  y0='z_child', y1='z_parent',
                                  color='color_child',
                                  alpha=1,
                                  source=segs,
                                  legend_label='pre_' + str(up_i))
                        # plot synapses
                        p.scatter(up_conn['x_post'], up_conn['z_post'], alpha=0.8, size=3, color=bokeh.palettes.OrRd9[0],
                                  legend_label='synapse(input)')
                        p.legend.label_text_font_size = str(int(fontSize * 1.5)) + 'pt'
                        output_file(newpath + str(up_i) + 'to' + str(neuron_i) + '_skeleton.html')  # skeleton of neurons of interest with pre-synapses (to downstream)
                        save(p)
                        if timeFlag == 1: tt = t; t = time.time(); print('Fetched', str(up_i),'to', str(neuron_i), 'Skeletons (input):', t-tt, 's ---',
                                                                         'neuron: '+str(neuron_count + 1)+'/'+str(len(neuron_ID)),
                                                                         'upstream: '+str(up_count+1)+'/'+str(len(up_ID)))

                        if PNG_export == 1:
                            # print('Exporting', str(up_i),'to', str(neuron_i), 'Skeleonts (input)...')
                            export_png(p, filename=newpath + str(up_i) + 'to' + str(neuron_i) + '_skeleton.png')
                            png_upNeuron.append(newpath + str(up_i) + 'to' + str(neuron_i) + '_skeleton.png')
                            # if timeFlag == 1: tt = t; t = time.time(); print('Exported', str(up_i),'to', str(neuron_i), 'Skeleonts (input):', t-tt, 's')  # time checkpoint
                if len(png_upNeuron) > 0:
                    for img_path in png_upNeuron:
                        img = Image.open(img_path).convert('RGB')
                        img.save(img_path)
                    with open(path + filename + '_upstreamNeuron.pdf', 'wb') as f_upNeuron:
                        f_upNeuron.write(img2pdf.convert(png_upNeuron))
    print()

# timer
t = time.time() - t0
print('\nElapsed ', t, ' s')
