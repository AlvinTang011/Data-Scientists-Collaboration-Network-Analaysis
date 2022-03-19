import requests, random, tqdm, os
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy.linalg
from pyvis.network import Network
from itertools import cycle, zip_longest
random.seed(42)

def get_links(node_url, pid_list):
    '''node_url: url of node person
    returns: set of people linked to person
    '''
    #website = requests.get(node_url)
    # Extract data using beautiful soup
    
    if not node_url.endswith('.html'):
        url_xml = requests.get(node_url).url.replace('.html','.xml')
    else: 
        url_xml = node_url.replace('.html','.xml')
    
    
    xml = requests.get(url_xml)
    
    xml_content = xml.content
    
    # Drop the data itself
    if xml.status_code != 200:
        return
    
    pid = '/'.join(xml.url.split('/')[-2:]).replace('.xml','')
    if pid in pid_list:
        return

    pid_list.append(pid)
    
    
    years = []
    soup = BeautifulSoup(xml_content, "xml")
    #name = soup.find('dblpperson')['name']

    articles = soup.find_all('r')
    coauthor_pidList = []
    # Extract 'coauthor_pid' and 'author_pid'
    for article in articles:
        article = article.contents[0]
        article_authors = article.find_all('author')
        #return coauthor pid
        for article_author in article_authors:
            coauthor_pid = article_author['pid']
            if coauthor_pid == pid:
                continue
            coauthor_pidList.append(coauthor_pid)
            years.append(article.find('year').text)
            
    colinks = ','.join(coauthor_pidList)
    years = ','.join(years)
    
    return dict(name=pid, links=colinks, years=years)

def clean_collaboration_network():
    df = pd.read_excel('./Input/DataScientists.xls')

    # Filter out those with same links as they would be the same person
    df = df.drop_duplicates(subset = ['dblp'], keep='last').sort_values(by=['name']).reset_index(drop=True)



    clean_df = dict(pid=[],
                    country=[],
                    institution=[],
                    coauthor_pid=[],
                    expertise=[],
                    years=[])

    pid_list = []

    with tqdm.tqdm(total=len(df)) as pbar:
        for _, row in df.iterrows():
            _, country, institution, node_url, _ = row.values
            data = get_links(node_url, pid_list)
            if not data:
                pbar.update(1)
                continue
            clean_df['pid'].append(data['name'])
            clean_df['coauthor_pid'].append(data['links'])
            clean_df['country'].append(country)
            clean_df['years'].append(data['years'])
            clean_df['institution'].append(institution)
            clean_df['expertise'].append(random.randint(1, 10))
            pbar.update(1)

    cleaned_df = pd.DataFrame.from_dict(clean_df)
    cleaned_df.to_csv('./Results/clean.csv', index=False)


    
# # Graph

# Formulating the graph from the given network

def formulate_graph(filteredNetwork_NoNAN, pidWithNoCoauthor):
    G = nx.Graph()
    G = nx.from_pandas_edgelist(filteredNetwork_NoNAN, source = 'pid', target = 'coauthor_pid')
    pidNoCoauthor = [row[1] for row in pidWithNoCoauthor.itertuples()]
    
    #add back nodes that do not have neighbours that were previousl removed
    for lonelyPID in pidNoCoauthor:
        G.add_node(lonelyPID)
        
    return G

# # Graph Properties

# ### 1. Network Measures and Metrics

# Simple Properties of Collaboration Network

def print_simple_properties(G):
    print('Info: ', nx.info(G))
    print('Density: ', nx.density(G))

    numNodes = nx.number_of_nodes(G)
    print('Total number of nodes of Network: ', numNodes)
    numEdges = nx.number_of_edges(G)
    print('Total number of edges of Network: ', numEdges)
    
    #Average Degree - 2 * total edges / nodes
    aveDegree = (2 * numEdges) / numNodes
    print('Average Degree of Network', aveDegree)
    
    print("Average clustering coefficient rounded to 3 sf: ", round(nx.average_clustering(G), 3))


# Diameter and Average Shortest Path Length of Collaboration Network

def display_subgraph_ave_shortest_path_and_diameter(G):
    counter = 1
    for con_Comp in (G.subgraph(comp).copy() for comp in nx.connected_components(G)):
        print("Sub Graph ", counter)
        print("Average shortest path length: ", round(nx.average_shortest_path_length(con_Comp), 2))
        print("Diameter: ", nx.diameter(con_Comp), "\n")
        counter += 1



# Average Clustering Coefficient of Collaboration Network

# ### 2. Degree Analysis of Collaboration Network

# The functions below will generate three things
# 1. The subgraph of connected components of the collaboration network
# 2. The degree-rank plot of the collaboration network
# 3. The degree histogram of the collaboration network
# 4. The degree distribution of the collaboration network

def plot_degree_analysis_graph(
        G,
        figName="Degree of Collaboration Network",
        saveName='./Results/Degree_Analysis.png',
        setTitle = 'Connected components of Collaboration Network'):
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure(figName, figsize=(20, 15))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    print('Info of Gcc')
    print(nx.info(Gcc))
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title(setTitle)
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")

    fig.tight_layout()
    fig.savefig (saveName,facecolor = 'w', bbox_inches='tight', dpi=300)
    #plt.show()
    return


def plot_degree_distribution(
        graph,
        figName= "Collaboration Network Degree Distribution " ,
        saveName="./Results/Degree_Distribution.png"):
    degs = {}
    for n in graph.nodes() :
        deg = graph.degree( n )
        if deg not in degs:
            degs[deg] = 0
        degs[deg] += 1
    items = sorted(degs.items())
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    
    ax.set_xscale('log')
    ax.set_yscale( 'log')
    plt.title(figName)
    ax.plot([ k for (k , v ) in items ] , [ v for (k,v) in items ])
    fig.savefig(saveName,bbox_inches='tight', dpi=150)
    


def plot_degree_histogram(
        G,
        normalized=True,
        figName='\nDegree Distribution Of Collaboration Network (log-log scale)',
        saveName="./Results/Degree_Distribution_Log_Log_Scale.png"):
    aux_y = nx.degree_histogram(G)
    
    aux_x = np.arange(0,len(aux_y)).tolist()
    
    n_nodes = G.number_of_nodes()
    
    if normalized:
        for i in range(len(aux_y)):
            aux_y[i] = aux_y[i]/n_nodes
            
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    plt.title(figName)
    plt.xlabel('Degree\n(log scale)')
    plt.ylabel('Number of Nodes\n(log scale)')
    plt.xscale("log")
    plt.yscale("log")
    ax.plot(aux_x, aux_y, 'o')
    fig.savefig (saveName,bbox_inches='tight', dpi=150)
    
    return



# ### 3. Betweenness Centrality of Collaboration Network

def betweenness_centrality_analysis(
        G,
        figName="Betweenness Centrality of Collaboration Network",
        saveName="./Results/Betweenness_Centrality.png"):
    # largest connected component
    components = nx.connected_components(G)
    largest_component = max(components, key=len)
    H = G.subgraph(largest_component)

    # compute centrality
    centrality = nx.betweenness_centrality(H, k=10, endpoints=True)

    # compute community structure
    lpc = nx.community.label_propagation_communities(H)
    community_index = {n: i for i, com in enumerate(lpc) for n in com}

    #### draw graph ####
    fig, ax = plt.subplots(figsize=(20, 15))

    pos = nx.spring_layout(H, k=0.15, seed=4572321)
    node_color = [community_index[n] for n in H]
    node_size = [v * 20000 for v in centrality.values()]
    nx.draw_networkx(
        H,
        pos=pos,
        with_labels=False,
        node_color=node_color,
        node_size=node_size,
        edge_color="gainsboro",
        alpha=0.4,
    )

    # Title/legend
    font = {"color": "b", "fontweight": "bold", "fontsize": 20}
    ax.set_title(figName, font)
    # Change font color for legend
    font["color"] = "r"

    ax.text(
        0.80,
        0.10,
        "node color = community structure",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )
    ax.text(
        0.80,
        0.06,
        "node size = betweenness centrality",
        horizontalalignment="center",
        transform=ax.transAxes,
        fontdict=font,
    )

    # Resize figure for label readibility
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    fig.savefig (saveName, facecolor = 'w',bbox_inches='tight', dpi=300)
    #plt.show()



# ### 4. Eigenvalues of Collaboration Network

def eigenvalues_analysis(
        G,
        figName='Eigenvalues of Collaboration Network',
        saveName="./Results/Eigenvalues_Graph.png"):
    L = nx.normalized_laplacian_matrix(G)
    e = numpy.linalg.eigvals(L.A)
    #print("Largest eigenvalue:", max(e))
    #print("Smallest eigenvalue:", min(e))

    fig = plt.figure(figsize=(15,10))
    plt.hist(e, bins=100)  # histogram with 100 bins
    plt.xlim(0, 2)  # eigenvalues between 0 and 2

    # Title
    plt.title(figName)
    plt.xlabel('Eigenvalue')
    plt.ylabel('Number of Nodes')

    fig.savefig (saveName,facecolor = 'w',bbox_inches='tight', dpi=300)
    #plt.show()



def draw(G, pos, measures, figName = 'Closeness Centrality of Collaboration Network', saveName="./Results/Closeness_Centrality.png"):
    fig = plt.figure(figsize=(15,10))
    nodes = nx.draw_networkx_nodes(G, pos, node_size=250, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    # labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos)
    
    #plt.figure(figsize=(20,15))
    plt.title(figName)
    plt.colorbar(nodes)
    plt.axis('off')
    
    fig.savefig (saveName,facecolor = 'w',bbox_inches='tight', dpi=300)
    #plt.show()
    



def evaluate_nodes_with_special_properties(G):
    degree_centrality = nx.degree_centrality(G)
    print('Below values are rounded to 3SF\n')
    
    {k: v for k, v in sorted(degree_centrality.items(), key=lambda item: item[1])}
    highestD = max(list(degree_centrality.values()))
    max_DC = list(degree_centrality.keys())[list(degree_centrality.values()).index(highestD)]
    print("PID with highest degree centrality: \n", max_DC, " with value: ", round(degree_centrality[max_DC], 5), "\n")

    betweenness_centrality = nx.betweenness_centrality(G)
    {k: v for k, v in sorted(betweenness_centrality.items(), key=lambda item: item[1])}
    highestB = max(list(betweenness_centrality.values()))
    max_BC = list(betweenness_centrality.keys())[list(betweenness_centrality.values()).index(highestB)]
    
    print("PID with highest betweenness centrality: \n", max_BC, " with value: ", round(betweenness_centrality[max_BC], 5), "\n")

    eigen_centrality = nx.eigenvector_centrality(G)
    {k: v for k, v in sorted(eigen_centrality.items(), key=lambda item: item[1])}
    highestE = max(list(eigen_centrality.values()))
    max_EC = list(eigen_centrality.keys())[list(eigen_centrality.values()).index(highestE)]
    
    print("PID with highest eigenvector centrality: \n", max_EC, " with value: ",round(eigen_centrality[max_EC], 5), "\n")

    closeness_centrality = nx.closeness_centrality(G)
    {k: v for k, v in sorted(closeness_centrality.items(), key=lambda item: item[1])}
    highestC = max(list(closeness_centrality.values()))
    #print(closeness_centrality)
    max_CC = list(closeness_centrality.keys())[list(closeness_centrality.values()).index(highestC)]

    # Included to display the closeness centrality properties of the 4 next largest closeness centrality
    #second_CC = sorted(list(closeness_centrality.values()), reverse = True)[1]
    #second_CC = list(closeness_centrality.keys())[list(closeness_centrality.values()).index(second_CC)]

    #third_CC = sorted(list(closeness_centrality.values()), reverse = True)[2]
    #third_CC = list(closeness_centrality.keys())[list(closeness_centrality.values()).index(third_CC)]

    #forth_CC = sorted(list(closeness_centrality.values()), reverse = True)[3]
    #forth_CC = list(closeness_centrality.keys())[list(closeness_centrality.values()).index(forth_CC)]

    #fifthCC = sorted(list(closeness_centrality.values()), reverse = True)[4]
    #fifthCC = list(closeness_centrality.keys())[list(closeness_centrality.values()).index(fifthCC)]
    
    print("PID with highest closeness centrality: \n", max_CC, " with value: ", round(closeness_centrality[max_CC],5), "\n")
    #print("PID with second closeness centrality: \n", second_CC, " with value: ", round(closeness_centrality[second_CC],5), "\n")
    #print("PID with third closeness centrality: \n", third_CC, " with value: ", round(closeness_centrality[third_CC],5), "\n")
    #print("PID with forth closeness centrality: \n", forth_CC, " with value: ", round(closeness_centrality[forth_CC],5), "\n")
    #print("PID with fifth closeness centrality: \n", fifthCC, " with value: ", round(closeness_centrality[fifthCC],5), "\n")




def obtain_collaboration_network(filteredNetwork_NoNAN, pidWithNoCoauthor):
    Graph_G = formulate_graph(filteredNetwork_NoNAN, pidWithNoCoauthor)

    # Visualizing the graph

    net = Network(notebook = False)
    net.from_nx(Graph_G)
    net.show('./Results/networkvisualized.html')

    print_simple_properties(Graph_G)

    display_subgraph_ave_shortest_path_and_diameter(Graph_G)

    plot_degree_analysis_graph(Graph_G)

    plot_degree_distribution(Graph_G)

    plot_degree_histogram(Graph_G)

    betweenness_centrality_analysis(Graph_G)

    eigenvalues_analysis(Graph_G)

    draw(Graph_G, nx.spring_layout(Graph_G, seed=675), nx.closeness_centrality(Graph_G))

    evaluate_nodes_with_special_properties(Graph_G)


# # Evolution of Collaboration Network and its Properties Over Time
def collaboration_network_evolution_over_time(cleaned_df, filteredNetwork):
    # get range of years papers were published
    n_years = set()
    for _, row in cleaned_df.iterrows():
        year = row.values[-1]
        year = list(set(year.split(',')))
        for y in year:
            n_years.add(y)
    n_years = sorted(str(x) for x in list(n_years))

    for col in filteredNetwork:
        filteredNetwork[col] = filteredNetwork[col].astype(str)

    # draw and save network for each discrete year
    os.makedirs('./Results/years', exist_ok=True)
    for y in n_years:
        sub_df = filteredNetwork[filteredNetwork['years'] == y]
        if len(sub_df) == 0:
            continue
        G = nx.Graph()
        G = nx.from_pandas_edgelist(sub_df, source='pid', target='coauthor_pid')
        net = Network(notebook = True)
        net.from_nx(G)
        net.show(f'./Results/years/{int(y)}.html')

    # draw and save network for cumulative years
    os.makedirs('./Results/cumulative_years', exist_ok=True)
    def _cumulative(year, years):
            year = int(year)
            years = [int(x) for x in years.split(',')]
            if len(years) == 0:
                return year >= years[0]
            return any(year >= y for y in years)
    for y in n_years:
        # publish any paper at or before year y
        sub_df = filteredNetwork[filteredNetwork['years'].apply(lambda x: _cumulative(y, x))]
        if len(sub_df) == 0:
            continue
        G = nx.Graph()
        G = nx.from_pandas_edgelist(sub_df, source='pid', target='coauthor_pid')
        net = Network(notebook = True)
        net.from_nx(G)
        net.show(f'./Results/cumulative_years/{int(y)}.html')

    # do network analysis for each discrete year
    folders = ['degree_analysis',
            'degree_distribution',
            'degree_histogram',
            'betweenness_centrality',
            'eigenvalues',
            'closeness']
    folders = ['Results/' + x for x in folders]
    for f in folders:
        os.makedirs(f, exist_ok=True)
    for y in n_years:
        print(f'Obtaining details for Collaboraition Network in year {y}')
        sub_df = filteredNetwork[filteredNetwork['years'] == y]
        if len(sub_df) == 0:
            continue
        G = nx.Graph()
        G = nx.from_pandas_edgelist(sub_df, source='pid', target='coauthor_pid')
        
        try:
            savefile_name = f'{y}.png'
            plot_degree_analysis_graph(
                G, 
                setTitle=f'\nConnected Components of Collaboration Network in year {y}',
                figName =f'\nConnected Components of Collaboration Network in year {y}',
                saveName = os.path.join(folders[0], savefile_name))
            plot_degree_distribution(
                G, 
                figName= f"\nDegree Distribution of Collaboration Network in year {y} " ,
                saveName = os.path.join(folders[1], savefile_name))
            plot_degree_histogram(
                G,
                figName=f'\nDegree Histogram of Collaboration Network in year {y} (log-log scale)',
                saveName=os.path.join(folders[2], savefile_name))
            betweenness_centrality_analysis(
                G, 
                figName=f"\nBetweenness Centrality of Collaboration Network in year {y}",
                saveName = os.path.join(folders[3], savefile_name))
            eigenvalues_analysis(
                G, 
                figName=f'\nEigenvalues of Collaboration Network in year {y}',
                saveName = os.path.join(folders[4], savefile_name))
            draw(G,
                nx.spring_layout(G, seed=675),
                nx.closeness_centrality(G),
                figName = f'\nCloseness Centrality of Collaboration Network in year {y}',
                saveName = os.path.join(folders[5], savefile_name))
            plt.close('all')
        except ValueError: # network too small
            pass

    # ## Average Degree over time
    plt.style.use('seaborn-whitegrid')

    averageDegrees = []; xAxisYears = []
    for y in n_years:
        sub_df = filteredNetwork[filteredNetwork['years'] == y]
        if len(sub_df) == 0:
            continue
        G = nx.Graph()
        G = nx.from_pandas_edgelist(sub_df, source='pid', target='coauthor_pid')
        averageDegrees.append(2 * nx.number_of_edges(G) / nx.number_of_nodes(G))
        xAxisYears.append(y)
    plt.figure(figsize=(10, 6))
    plt.title('Average Degree Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Degree')
    plt.xticks(rotation=90)
    plt.plot(xAxisYears, averageDegrees)
    plt.savefig('./Results/Average_Degree_Over_Time.png', dpi=160)

# # Random Network and its Comparison with Collaboration Network

# ### 1. Generating Random Network
def obtain_random_network(filteredNetwork_NoNAN, pidWithNoCoauthor):
    # Obtain original graph
    G = formulate_graph(filteredNetwork_NoNAN, pidWithNoCoauthor)
    numNodes = nx.number_of_nodes(G)
    numEdges = nx.number_of_edges(G)
    #Average Degree - 2 * total edges / nodes
    aveDegree = (2 * numEdges) / numNodes

    # Generate Random Network
    print('Generating Random Network')
    randomG = nx.fast_gnp_random_graph(numNodes, aveDegree/(numNodes-1))
    print(nx.number_of_nodes(randomG))
    print('Average Degree:', 2 * nx.number_of_edges(randomG)/nx.number_of_nodes(randomG))


    print_simple_properties(randomG)

    display_subgraph_ave_shortest_path_and_diameter(randomG)

    plot_degree_analysis_graph(
        randomG,
        setTitle='Connected Components of Random Network',
        figName ='Connected Components of Random Network',
        saveName = './Results/random_network/Random Network Degree Analysis.png'
        )

    plot_degree_distribution(
        randomG,
        figName='Degree Distribution of Random Network',
        saveName = './Results/random_network/Random Network Degree Distribution.png'
        )

    plot_degree_histogram(
        randomG,
        figName='Degree Histogram of Random Network',
        saveName = './Results/random_network/Random Network Degree Histogram.png'
        )

    betweenness_centrality_analysis(
        randomG,
        figName='Betweenness Centrality of Random Network',
        saveName = './Results/random_network/Random Network Betweenness Centrality.png'
        )

    eigenvalues_analysis(
        randomG,
        figName='Eigenvalues of Random Network',
        saveName = './Results/random_network/Random Network Eigenvalues.png'
        )

    draw(randomG, 
    nx.spring_layout(randomG, seed=675), 
    nx.closeness_centrality(randomG), 
    figName = 'Closeness Centrality of Random Network',
    saveName = './Results/random_network/Random Network Closeness Centrality.png')

    print('Random network\'s nodes with special properties')
    evaluate_nodes_with_special_properties(randomG)

# # Algorithm for Transformed Network

# ### Defining functions for transforming network

# Finding network max degree
def find_max_degrees(G):
    degrees = [val for (node, val) in G.degree()]
    return max(degrees)

# Utility needed
def zip_cycle(*iterables, empty_default=None):
    cycles = [cycle(i) for i in iterables]
    for _ in zip_longest(*iterables):
        yield tuple(next(i, empty_default) for i in cycles)

# Validate user input
def valid_degree(degree, G):
    try:
        maxDegree = int(degree)
        if maxDegree > find_max_degrees(G) or maxDegree < 0:
            print('Current network graph has maximum degree of ', find_max_degrees(G))
            print('Please input valid range from 1 to maximum degree')
            return True
        elif maxDegree == 0:
            print('Max Degree set to 0')
            return False
        else:
            print('Max Degree set to: ', maxDegree)
            return False
    except:
        print('Please Input an Integer')
        return True
    return False

def transform_network_algorithm(max_degree, G):

    if max_degree == 0:
        G.clear()
        return
    
    counter = 0
    before_transformDF = pd.read_csv('./Results/clean.csv')

    df = before_transformDF[['pid', 'coauthor_pid']]
    lst_col = 'coauthor_pid'
    df1 = df.assign(**{lst_col: (df[lst_col]).str.split(',')})
    # Separate the coauthor_pid
    collabNetwork = df1.explode('coauthor_pid', ignore_index = True)


    # Obtain list of existing pid
    existingNames = df[['pid']]
    existNames = [row[1] for row in existingNames.itertuples()]

    # Remove names that do not exist there
    filteredNetwork_NoNAN = collabNetwork[collabNetwork['coauthor_pid'].isin(existNames)]
    nodes_net = filteredNetwork_NoNAN.drop_duplicates(subset = ['pid'], keep='first', inplace = False, ignore_index = True)

    # Initalize weight for diversity
    countryDF = before_transformDF.groupby(['country'])['pid'].count().reset_index().rename(columns = {'pid': 'count'})
    expertiseDF = before_transformDF.groupby(['expertise'])['pid'].count().reset_index().rename(columns = {'pid': 'count'})
    institutionDF = before_transformDF.groupby(['institution'])['pid'].count().reset_index().rename(columns = {'pid': 'count'})

    # Calculate individual weights based on country, institution and expertise
    for country, institution, expertise in zip_cycle(countryDF['country'], institutionDF['institution'], expertiseDF['expertise']):
        before_transformDF.loc[before_transformDF['country'] == country, 'ctry_sum'] = countryDF.loc[countryDF['country'] == country, 'count'].iloc[0]
        before_transformDF.loc[before_transformDF['institution'] == institution, 'inst_sum'] = (institutionDF.loc[institutionDF['institution'] == institution, 'count'].iloc[0])
        before_transformDF.loc[before_transformDF['expertise'] == expertise, 'exp_sum'] = expertiseDF.loc[expertiseDF['expertise'] == expertise, 'count'].iloc[0]

    before_transformDF['inst_sum'].fillna(0, inplace = True)

    # Weight allocation to preserve diversity
    # institute = 50 [emphasised to be removed first]
    # country = 1
    # expertise = 1
    before_transformDF['weight'] = before_transformDF['ctry_sum'] + 50*before_transformDF['inst_sum'] + before_transformDF['exp_sum']

    for pid_transformNet in before_transformDF['pid']:
        nodes_net.loc[nodes_net['pid'] == pid_transformNet, 'weight'] = before_transformDF.loc[before_transformDF['pid'] == pid_transformNet, 'weight'].iloc[0]


    # Set the weights for nodes present in network accordingly
    nodes_net = nodes_net.sort_values(by='weight', ascending = False)

    nodes_removed_from_graph = []
    # we need to check maxdegree < find max every remove
    while max_degree < find_max_degrees(G):
        #calculate weight after removal

        #pop the highest weight node
        pid = nodes_net['pid'].iloc[0]
        nodes_net = nodes_net.iloc[1: , :]

        # For loop to get the full PID list that has all the affected nodes
        # Country, Institute, and Expertise for PID
        pid_ctry = before_transformDF.loc[before_transformDF['pid'] == pid, 'country'].iloc[0]
        pid_inst = before_transformDF.loc[before_transformDF['pid'] == pid, 'institution'].iloc[0]
        pid_exp = before_transformDF.loc[before_transformDF['pid'] == pid, 'expertise'].iloc[0]

        # Obtain affected PID that will have weight changes
        pid_affected = []
        pid_affected_I = []
        # PID affected by country
        pid_ctry_DF = before_transformDF.loc[before_transformDF['country'] == pid_ctry]
        pid_ctry = pid_ctry_DF['pid']
        for pid_aff_C in pid_ctry:
            pid_affected.append(pid_aff_C)
        # PID affected by institute
        pid_inst_DF = before_transformDF.loc[before_transformDF['institution'] == pid_inst]
        pid_inst = pid_inst_DF['pid']
        for pid_aff_I in pid_inst:
            pid_affected_I.append(pid_aff_I)
        # PID affected by expertise
        pid_exp_DF = before_transformDF.loc[before_transformDF['expertise'] == pid_exp]
        pid_exp = pid_exp_DF['pid']
        for pid_aff_E in pid_exp:
            pid_affected.append(pid_aff_E)

        # Decrease weight according to the removed PID
        i = 0
        while i < len(pid_affected):
            nodes_net.loc[nodes_net['pid'] == pid_affected[i], 'weight'] -= 1
            i += 1
        i_inst = 0
        while i_inst < len(pid_affected_I):
            nodes_net.loc[nodes_net['pid'] == pid_affected_I[i_inst], 'weight'] -= 50
            i_inst += 1

        # Print for clarity of number of nodes removed
        counter += 1
        print('Removing node', counter, 'with pid:', pid)

        # Removing selected node from graph while keeping a track of what is removed
        G.remove_node(pid)
        nodes_removed_from_graph.append(pid)


    # Get countries/expertise/institutions
    after_transformDF = before_transformDF
    for remove_pid in nodes_removed_from_graph:
        index_pid = before_transformDF[before_transformDF['pid'] == remove_pid].index
        after_transformDF.drop(index_pid, inplace = True)
    
    

    countryDF_transformed = after_transformDF.groupby(['country'])['pid'].count().reset_index().rename(columns = {'pid': 'count_after'})
    expertiseDF_transformed = after_transformDF.groupby(['expertise'])['pid'].count().reset_index().rename(columns = {'pid': 'count_after'})
    institutionDF_transformed = after_transformDF.groupby(['institution'])['pid'].count().reset_index().rename(columns = {'pid': 'count_after'})


    countries_comparison = pd.merge(countryDF, countryDF_transformed, on="country", how = 'outer')
    expertise_comparison = pd.merge(expertiseDF, expertiseDF_transformed, on="expertise", how = 'outer')
    institution_comparison = pd.merge(institutionDF, institutionDF_transformed, on="institution", how = 'outer')
 
    countries_comparison.plot(x="country", y=["count", "count_after"], kind="bar").get_figure().savefig(
        './Results/transformed_network/Countries before and after comparison.png',bbox_inches='tight', dpi=300)

    expertise_comparison.plot(x="expertise", y=["count", "count_after"], kind="bar").get_figure().savefig(
        './Results/transformed_network/Expertise before and after comparison.png',bbox_inches='tight', dpi=300)


    to_bin = sorted(institution_comparison.institution.values)
    bin_index = len(to_bin) // 10
    institution_comparison.institution = institution_comparison.institution.apply (
        lambda x: f'Bin {str(to_bin.index(x) // bin_index). zfill(2)}'
    )
    institution_comparison = institution_comparison.groupby('institution').count().reset_index()
    institution_comparison.plot(x = 'institution', y = ['count', 'count_after'], kind = 'bar').get_figure().savefig(
        './Results/transformed_network/Institution before and after comparison.png',
        bbox_inches = 'tight', dpi = 300
    )
    
        
    return nodes_removed_from_graph

# ### Function calls
def obtain_transform_network(filteredNetwork_NoNAN, pidWithNoCoauthor):
    # Obtain back original network
    G = formulate_graph(filteredNetwork_NoNAN, pidWithNoCoauthor)

    print('Formulating transformed network')
    print('Current network graph has maximum degree of ', find_max_degrees(G))
    # User Input
    input_validation = True
    while input_validation:
        user_max_degree = input("Enter your desired maximum degree for transform network: ")
        input_validation = valid_degree(user_max_degree, G)
        continue
        
    # Transform network
    nodes_removed = transform_network_algorithm(int(user_max_degree), G)

    # Print output for user
    try:
        print('Network has been transformed with a maximum degree of:', find_max_degrees(G))
    except:
        print('Network is cleared, degree is 0')


    # Visualize transformed network

    net = Network(notebook = True)
    net.from_nx(G)

    net.show('./Results/transformednetworkvisualized.html')

    print('Generating Transformed Network properties')

    print_simple_properties(G)

    display_subgraph_ave_shortest_path_and_diameter(G)

    plot_degree_analysis_graph(
        G,
        setTitle='Connected Components of Transformed Network',
        saveName = './Results/transformed_network/Transformed Network Degree Analysis.png'
        )

    plot_degree_distribution(
        G,
        figName='Degree Distribution of Transformed Network',
        saveName = './Results/transformed_network/Transformed Network Degree Distribution.png'
        )

    plot_degree_histogram(
        G,
        figName='Degree Histogram of Transformed Network',
        saveName = './Results/transformed_network/Transformed Network Degree Histogram.png'
        )

    betweenness_centrality_analysis(
        G,
        figName='Betweenness Centrality of Transformed Network',
        saveName = './Results/transformed_network/Transformed Network Betweenness Centrality.png'
        )

    eigenvalues_analysis(
        G,
        figName='Eigenvalues of Transformed Network',
        saveName = './Results/transformed_network/Transformed Network Eigenvalues.png'
        )

    draw(G, nx.spring_layout(G, seed=675), nx.closeness_centrality(G), 
    figName = 'Closeness Centrality of Transformed Network',
    saveName = './Results/transformed_network/Transformed Network Closeness Centrality.png')

    print('Transformed network\'s nodes with special properties')
    evaluate_nodes_with_special_properties(G)

    print('Transformed Network properties have been obtained')


def main():
    # part 1
    clean_collaboration_network()

    print('Data cleaned')

    cleaned_df = pd.read_csv('./Results/clean.csv')

    df = cleaned_df[['pid', 'coauthor_pid', 'years']]
    df1 = df.assign(**{'coauthor_pid': (df['coauthor_pid']).str.split(','),
                    'years': (df['years']).str.split(',')})
    # Separate the coauthor_pid and years in pairs
    collabNetwork = df1.explode(['coauthor_pid', 'years'], ignore_index = True)

    # Obtain list of existing authors (pid)
    existingNames = df[['pid', 'years']]
    existNames = [row[1] for row in existingNames.itertuples()]

    # Remove coauthors that are not authors
    filteredNetwork_NoNAN = collabNetwork[collabNetwork['coauthor_pid'].isin(existNames)]

    # Obtain back pid that was removed due to having no coauthor from pid
    pidWithNoCoauthor = existingNames[~existingNames['pid'].isin(filteredNetwork_NoNAN['pid'])]

    filteredNetwork = filteredNetwork_NoNAN.append(pidWithNoCoauthor, ignore_index = False)

    filteredNetwork = filteredNetwork.drop_duplicates(keep='first', inplace = False, ignore_index = True)

    filteredNetwork.to_csv('./Results/filtered.csv', index=False)

    print('Obtianing Collaboration Network')
    obtain_collaboration_network(filteredNetwork_NoNAN, pidWithNoCoauthor)

    # part 2
    print('Obtaining evolution over time')
    collaboration_network_evolution_over_time(cleaned_df, filteredNetwork)

    # part 3
    obtain_random_network(filteredNetwork_NoNAN, pidWithNoCoauthor)

    # part 4
    obtain_transform_network(filteredNetwork_NoNAN, pidWithNoCoauthor)

    print('Ending program')

main()