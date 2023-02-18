
import pandas as pd
import numpy as np
from PIL import ImageTk,Image
from PIL.Image import Resampling

from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import messagebox
import networkx as nx


class Vertex:
    ''' This Class is to save vertices of graph this will have key as Line Name
    index for that index of station in array
    and name of station
    '''
    def __init__(self,key,nameOfStation,ind):
        self.key=key
        self.index=ind
        self.nameOfStation=nameOfStation

class Edges:
    '''
     This class is for Edges of Graph
     where source station and destination station will be saved
     and also the cost/time of travelling through those stations
    '''
    def __init__(self,key,origin,destination,cost):
        self.key = key

        self.sourceVertex=origin
        self.destinationVertex=destination
        self.cost=cost

class AdjecencyMatrix:
    '''
    This class is used to store our graph in computer
    so to achieve this we will use adjacency matrix

    '''
    def __init__(self, edges, vertices):
        #edges of graph
        self.edges = edges
        #vertices of graph
        self.vertices = vertices
        #unique vertices because adjacency matrix should only have on vertex for each station
        self.uniqueVertices = list(self.getUniqueVertices(self.vertices))
        #returns Adjacency Matrix
        self.adjacencyMatrix = self.getam()
        #this function will initialize adjacency matrix so that it will place cost ,edge,vertices to a 2d array
        self.populateAdjacencyMatrix()


    def getam(self):
        ''' this function returns adjacency matrix'''
        #list to store adjacency matrix data
        am = []
        numberOfVertices=len(self.vertices)
        print(numberOfVertices,'total vertices')
        #uniqueVetices=self.getUniqueVertices(self.vertices)
        numberofUniqueVertices=len(self.uniqueVertices)
        print(numberofUniqueVertices,'unique vertices')

        # this loop will create a 2d array of list according to number of unique vertices
        i = 0
        while i < numberofUniqueVertices:
            l2 = []
            j = 0
            while j < numberofUniqueVertices:
                l2.append([None])
                j = j+1
            am.append(l2)
            i=i+1
        # this will return a adjecency matrix
        return am
    def getIndexOfVertices(self,nameofStation):
        ''' This Function will return index of vertex '''
        ind=self.uniqueVertices.index(nameofStation)
        return ind

    def getUniqueVertices(self,vertices):
        ''' this function will return stations names of vertices '''
        unlist={ver.nameOfStation for ver in vertices}
        return unlist

    def populateAdjacencyMatrix(self):
        """ This Function is where we will place the cost ,edges,vertices to our graph """
        #iterate through each connected edges and place source station name,line,and cost
        for eachConnectedEdge in self.edges:
            #print(eachConnectedEdge.key,'in populate')
            self.adjacencyMatrix[self.getIndexOfVertices(eachConnectedEdge.sourceVertex.nameOfStation)][self.getIndexOfVertices(eachConnectedEdge.destinationVertex.nameOfStation)]=\
                [eachConnectedEdge.sourceVertex.key,eachConnectedEdge.sourceVertex.nameOfStation,eachConnectedEdge.destinationVertex.key,
                 eachConnectedEdge.destinationVertex.nameOfStation,eachConnectedEdge.cost]
        # same here but index will be reversed to place [1,5] is having same data as [5,1]
        for eachConnectedEdge in self.edges:
            #print(eachConnectedEdge.key,'in populate')
            self.adjacencyMatrix[self.getIndexOfVertices(eachConnectedEdge.destinationVertex.nameOfStation)][self.getIndexOfVertices(eachConnectedEdge.sourceVertex.nameOfStation)] = \
                [eachConnectedEdge.destinationVertex.key, eachConnectedEdge.destinationVertex.nameOfStation,
                 eachConnectedEdge.sourceVertex.key, eachConnectedEdge.sourceVertex.nameOfStation,
                 eachConnectedEdge.cost]

    def printadmatrix(self):
        #print('dimension of adjecencymatrix')
        print(self.adjacencyMatrix)


class FileReader:
    ''' This Function will read our data file '''
    def __init__(self,filename):
        self.filename = filename
        self.dataframe = self.getDataFrame()


    def getDataFrame(self):
        """ This function will return a pandas data frame after reading file"""
        headers = ['lines', 'source', 'dest', 'cost']
        df = pd.read_excel(self.filename, header=None)
        df.columns = headers
        df=df.reset_index()
        return df


def getDesiredVertex(lines,stationName,vertices):
    """ This function will return desired vertex for line ,station name"""

    #check in each vertices
    for ver in vertices:

        if ver.key==lines and ver.nameOfStation==stationName:
            #print(ver.key,ver.nameOfStation)

            return ver


class DijekstraAlgorithem:
    """This Function will implement Dijkstra Algorithm on our Adjacency Matrix to get shortest path and time"""
    def __init__(self,graph,source,dist,numberOfVerticesInAM,vertices):
        self.graph=graph
        self.source = source
        self.distination = dist
        self.numberofVerticesInAM = numberOfVerticesInAM
        self.vertices = vertices
        self.infinityNum=999999
        self.distance=[]
        self.visited=[False]*numberOfVerticesInAM
        self.parent=[]
        self.sourceIndex=-9
        self.distinationIndex=-9
        self.pathDict = {}
        self.graphTask3=nx.Graph()

    def getMinimumOfVertices(self,distVertices,visitedVertices):
        """This function will return the minimum vertex which is having low cost/time  """
        index=0
        minimumCost= self.infinityNum
        for i in range(self.numberofVerticesInAM):
            if( not visitedVertices[i] and distVertices[i]<minimumCost):
                minimumCost = distVertices[i]
                index=i

        return index
    def initializeMatrices(self):
        """ this function will be our initializer for our declared variables """
        #first our distance will be infinity
        self.distance=[self.infinityNum]*self.numberofVerticesInAM
        #index of source vertex
        self.sourceIndex=self.vertices.index(self.source)
        #distination index
        self.distinationIndex=self.vertices.index(self.distination)
        self.distance[self.sourceIndex]=0
        #add i for parent array
        for i in range(self.numberofVerticesInAM):
            self.parent.append(i)

    def getNearestMinimumNode(self):
        """This will return Nearest minimum node for our dijkstra algorithm"""
        minimumValue=self.infinityNum
        minimumNode=0
        for i in range(self.numberofVerticesInAM):
            if (not self.visited[i] and self.distance[i] < minimumValue):
                minimumValue=self.distance[i]
                minimumNode=i
        return minimumNode

    def dijkstra_algorithm(self):
        """ This Function implements our dijkstra algorithm"""
        #iterate every vertices
        for i in range(self.numberofVerticesInAM):
            #get minimum node
            nearestNode=self.getNearestMinimumNode()
            #mark this visted
            self.visited[nearestNode] = True
            #this loop will check if there is cost between any node which is minimum
            c=0
            for adjecentNode in range(self.numberofVerticesInAM):

                if (self.graph[nearestNode][adjecentNode] != [None] ):
                    getCost=self.graph[nearestNode][adjecentNode]

                    realCost=int(getCost[-1])

                # this is dijekstra algorithm main scenario where we will update distance if it is minimum
                if (self.graph[nearestNode][adjecentNode] != [None] and self.distance[adjecentNode]> self.distance[nearestNode] + realCost):
                    self.distance[adjecentNode]=self.distance[nearestNode] + realCost
                    self.parent[adjecentNode]=nearestNode
                    self.graphTask3.add_edge(i, c)
                c=c+1
    def displayPath(self):
        """Prints the path array"""

        #print ("Node \t \t \t Cost \t \t \t Path")
        for i in range(self.numberofVerticesInAM):
            #print( i, '\t \t \t ',self.distance[i],'\t \t \t  ',i)

            parnode=self.parent[i]
            keysrcdis = str(i)
            parentnodesnam = []
            parentnodesnam.append(self.vertices[i])
            while parnode!= self.sourceIndex:

                #print("<----",parnode,end="  ",)
                parentnodesnam.append(self.vertices[parnode])
                parnode=self.parent[parnode]

            #print("\n")
            self.pathDict[keysrcdis] = parentnodesnam
        #print(self.pathDict)
    def getPathFromSourceToIndex(self):
        """Returns the Path from Source to Destination"""
        return self.pathDict[str(self.distinationIndex)]
    def getTimeConsumedFromSourceToDistination(self):
        """Returns total time/cost"""
        return self.distance[self.distinationIndex]

    def createHistogramOfAllStationsPairs(self):
        """This creates a histogram of all pairs of station cost """

        nameofsrcanddistination=[]
        costsrcdist=[]

        for i in range(self.numberofVerticesInAM):
            print(i)

            for j in range(self.numberofVerticesInAM):

                chlist=[self.vertices[i],'to',self.vertices[j]]
                blist=[self.vertices[j],'to',self.vertices[i]]
                if i != j and chlist not in nameofsrcanddistination and blist not in nameofsrcanddistination:
                    nameofsrcanddistination.append([self.vertices[i],'to',self.vertices[j]])
                    dijekstra = DijekstraAlgorithem(self.graph, self.vertices[i],
                                                    self.vertices[j], self.numberofVerticesInAM,
                                                    self.vertices)
                    dijekstra.initializeMatrices()
                    dijekstra.dijkstra_algorithm()
                    costb = dijekstra.getTimeConsumedFromSourceToDistination()
                    costsrcdist.append(costb)

        '''for i in range(self.numberofVerticesInAM):
            nameofsrcanddistination.append(self.vertices[i])

        costsrcdist=np.array(self.distance)
        '''
        #print('all values of cost ',costsrcdist)
        figure,axis=plt.subplots(1,1)
        axis.hist(costsrcdist)
        axis.set_title("Histogram Of Time Between All Different Pair of Stations ")
        axis.set_ylabel('Time ')

        '''rects=axis.patches
        labels=[x for x in nameofsrcanddistination]
        for rect,label in zip(rects,labels):
            height=rect.get_height()
            axis.text(rect.get_x()+rect.get_width()/2,height+0.1,label,ha='center',va='bottom')
        '''
        plt.savefig('histogram.png')


        '''lenthofHelper=np.arange(len(nameofsrcanddistination))
        plt.bar(lenthofHelper,costsrcdist)
        plt.xticks(ticks=lenthofHelper,labels=nameofsrcanddistination,rotation=45)
        plt.xlabel(' Number of connection source to destination')
        plt.ylabel('Time taken')
        plt.title('Histogram Of Time Between All Different Pair of Stations From Source')
        plt.savefig('histogram.png')
        return 0'''

    def task3augmentingtask1(self):
        """ This will return all the bridges of graph from networkx library"""
        Task3=nx.algorithms.bridges(self.graphTask3)
        #print(self.graphTask3.matrix)

        bridgesofgraph=Task3
        answersbrdiges=[]
        for eachbridge in bridgesofgraph:
            #print(eachbridge)
            answersbrdiges.append([self.vertices[eachbridge[0]],self.vertices[eachbridge[1]]])

        return answersbrdiges

class HelpGovernmetByRemovingStations:
    """This class will return all those stations which can be removed and still will have another path to reach destinaton"""

    def __init__(self,numberofedges,edges,vertices):
        self.setOfStations=set()
        self.parentNodes=[i for i in range(numberofedges)]
        self.edges=edges
        self.vertices = vertices
        self.indexed_edges=self.set_edges_indexes()
        self.numberOfEdges=numberofedges


    def find_parent(self,node):
        """This funciotn returns the parent of current node """
        #print("in PArent efdf ",node)
        if node==self.parentNodes[node]:
            return node
        self.parentNodes[node]=self.find_parent(self.parentNodes[node])
        return self.parentNodes[node]

    def find_extra_stations(self):
        """This Function will return all the stations which can be closed """
        #traverse for each edge
        for edg in self.indexed_edges:
            #print('edge',edg)
            #find parents for u and v
            parent_of_v=self.find_parent(edg[0])
            parent_of_u=self.find_parent(edg[1])
            #check if parent of v is same as parent of u
            if parent_of_v==parent_of_u:
                #if edg first is less then edg second place it first
                if (edg[0]<edg[1]):
                    #create a tuple
                    tuples=(edg[0],edg[1])
                else:
                    tuples=(edg[1],edg[0])
                #add to the set of staions
                self.setOfStations.add(tuples)
            else :
                self.parentNodes[parent_of_v]=parent_of_u
        return self.setOfStations
    def set_edges_indexes(self):
        """this class will assign a index for each edge """
        indexedges=[]
        for eachedge in self.edges:
            src_index=self.vertices.index(eachedge.sourceVertex.nameOfStation)
            dis_index=self.vertices.index(eachedge.destinationVertex.nameOfStation)
            indexedges.append([src_index,dis_index])
        return indexedges

    def get_can_discard_stations(self):
        """This will return a string array of names of those stations which can be closed """
        answer=[]
        for each_value in self.setOfStations:
            src=self.vertices[each_value[0]]
            dis=self.vertices[each_value[1]]
            answer.append([src,dis])
        return answer


def my_magic_output(data):
    """This function is used for console printing """
    maymagic_string = ''
    i = 0
    chk = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130]
    for i in range(len(data)):
        if i in chk:
            maymagic_string += ' \n  '
        maymagic_string += str(data[i]) + '   ,   '
    return maymagic_string

def call_this_function_for_console_outputs():
    """This function is used for Console outputs only """

    #create a FileReader Object and pass data file name
    filereader = FileReader('data.xlsx')
    #returns a dat frame
    dataFrame = filereader.getDataFrame()
    # print(dataFrame)

    # lets create our vertices first to create vertices from our file we have to drop those columns which are not having any destination and cost
    verticesDataFromDataFrame = dataFrame[dataFrame['dest'].isna()]
    verticesDataFromDataFrame = verticesDataFromDataFrame.reset_index()
    # Lets Create edges
    edgesDataFromDataFrame = dataFrame[~dataFrame['dest'].isna()]
    edgesDataFromDataFrame = edgesDataFromDataFrame.reset_index()
    # print(edgesDataFromDataFrame)

    # lets create Vertices For out Graph
    vertices = []
    for index, row in verticesDataFromDataFrame.iterrows():
        vertices.append(Vertex(row['lines'], row['source'], index))
    # print(vertices)

    # lets create Edges from out data frame
    edges = []

    for index, row in edgesDataFromDataFrame.iterrows():
        source = getDesiredVertex(row['lines'], row['source'], vertices)

        destination = getDesiredVertex(row['lines'], row['dest'], vertices)

        edges.append(Edges(row['lines'], source, destination, row['cost']))
    # print(edges)

    # lets create our adjacency matrix for graph representation
    adjecencyMatrix = AdjecencyMatrix(edges, vertices)
    # print(adjecencyMatrix.printadmatrix())

    # lets see our dijkstra part
    numberOfuniqueVertices = len(adjecencyMatrix.uniqueVertices)
    sourceName = "Harrow & Wealdstone"
    distName = "Brixton"
    srcIndex = adjecencyMatrix.uniqueVertices.index(sourceName)
    ditIndex = adjecencyMatrix.uniqueVertices.index(distName)
    print("Source is  ", sourceName, " Distination is ", distName)
    print("Source is  ", adjecencyMatrix.uniqueVertices[srcIndex], " Distination is ",
          adjecencyMatrix.uniqueVertices[ditIndex])
    #create a dijkstra object and pass adjacency matrix msource ,destination and vertices as required
    dijkstra = DijekstraAlgorithem(adjecencyMatrix.adjacencyMatrix, adjecencyMatrix.uniqueVertices[srcIndex],
                                    adjecencyMatrix.uniqueVertices[ditIndex], numberOfuniqueVertices,
                                    adjecencyMatrix.uniqueVertices)
    #initilize data in dijkstra
    dijkstra.initializeMatrices()
    #creates shortest path and calculates time
    dijkstra.dijkstra_algorithm()

    dijkstra.displayPath()

    print('this is the path',dijkstra.getPathFromSourceToIndex())
    print('####################################################################################')
    print('this is the time taken form source to destination ', dijkstra.getTimeConsumedFromSourceToDistination())

    #dijkstra.createHistogramOfAllStationsPairs()
    print('####################################################################################')
    print('List OF Stations which can be discarded ')

    #Lets create HelpG...... class object to detect which stations can be closed
    gov_helper = HelpGovernmetByRemovingStations(len(edges), edges, adjecencyMatrix.uniqueVertices)
    gov_helper.find_extra_stations()
    print(gov_helper.get_can_discard_stations())
    print('####################################################################################')
    print()
    print('Task 3 is augmenting task 1 to find the all bridges of graph ')
    print(dijkstra.task3augmentingtask1())
    print()
    print('####################################################################################')


if __name__ == '__main__':

    #call_this_function_for_console_outputs()
    #read the file and return data frame
    filereader = FileReader('data.xlsx')
    dataFrame = filereader.getDataFrame()
    # print(dataFrame)

    # lets create our vertices first, to create vertices from our file we have to drop those columns which do not have any destinations and costs
    verticesDataFromDataFrame = dataFrame[dataFrame['dest'].isna()]
    verticesDataFromDataFrame = verticesDataFromDataFrame.reset_index()
    # lets create our edges
    edgesDataFromDataFrame = dataFrame[~dataFrame['dest'].isna()]
    edgesDataFromDataFrame = edgesDataFromDataFrame.reset_index()
    # print(edgesDataFromDataFrame)

    # lets create Vertices For out Graph
    vertices = []
    for index, row in verticesDataFromDataFrame.iterrows():
        vertices.append(Vertex(row['lines'], row['source'], index))
    # print(vertices)

    # lets create Edges from our data frame
    edges = []
    for index, row in edgesDataFromDataFrame.iterrows():
        source = getDesiredVertex(row['lines'], row['source'], vertices)

        destination = getDesiredVertex(row['lines'], row['dest'], vertices)

        edges.append(Edges(row['lines'], source, destination, row['cost']))
    # print(edges)
    #here for tkinter we are building a drop down option menu such that if a user selects Line it will only show those stations which present Line
    optionsLinesSource = verticesDataFromDataFrame['lines'].unique()
    optionsStationsSource = verticesDataFromDataFrame[verticesDataFromDataFrame['lines'] == optionsLinesSource[0]]
    optionsStationsSource = optionsStationsSource['source']

    optionsLinesDestination = verticesDataFromDataFrame['lines'].unique()
    optionsStationsDestination = verticesDataFromDataFrame[
        verticesDataFromDataFrame['lines'] == optionsLinesDestination[0]]
    optionsStationsDestination = optionsStationsDestination['source']
    adjacencyMatrix = AdjecencyMatrix(edges, vertices)
    # print(adjecencyMatrix.printadmatrix())
    route = 'select Source and Destination'
    gov_station_can_remove = 'Please run task 1 first '
    time = 'select source and destination'

    # this is the Gui Part using tkinter library
    root = tk.Tk()
    root.geometry("1200x1000")
    tk.Label(root, text='London Underground Tube Map', font="comicsansms 15 bold", pady=20).grid(row=1, column=4)

    # Label(root,text='London ',font="comicsansms 15 bold",pady=20).grid(row=1,column=1)
    tk.Label(root, text='Please Select Line and Starting Station ', font="comicsansms 9 bold", ).grid(row=5, column=3)
    tk.Label(root, text='Please Select Line and Destination Station ', font="comicsansms 9 bold", ).grid(row=5,
                                                                                                         column=8)
    #placing a drop down menu for source line this
    sourceStationLine = tk.StringVar(root)
    sourceStationLine.set('Line of Source')
    sourceStationLine.set(optionsLinesSource[0])
    sLineMenu = tk.OptionMenu(root, sourceStationLine, *optionsLinesSource)
    sLineMenu.grid(row=6, column=3)
    #placing a drop down menu for source station name
    sourceStationName = tk.StringVar(root)
    sourceStationName.set('Station Name')
    sourceStationName.set(optionsStationsSource[0])
    sStationNameMenu = tk.OptionMenu(root, sourceStationName, *optionsStationsSource)
    sStationNameMenu.grid(row=7, column=3)

    #placing destination line dropdown menu
    destinationStationLine = tk.StringVar(root)
    destinationStationLine.set('Line of Destination')
    destinationStationLine.set(optionsLinesDestination[0])
    dLineMenu = tk.OptionMenu(root, destinationStationLine, *optionsLinesDestination)
    dLineMenu.grid(row=6, column=8)
    #placing drop down menu destination stations names
    destinationStationName = tk.StringVar(root)
    destinationStationName.set('Station Name')
    destinationStationName.set(optionsStationsDestination[0])
    dStationNameMenu = tk.OptionMenu(root, destinationStationName, *optionsStationsDestination)
    dStationNameMenu.grid(row=7, column=8)

    #placing our histogram images which is stored
    task1image = (Image.open('histogram.png'))
    resized_img = task1image.resize((700, 300), Resampling.LANCZOS)
    task1histogram = ImageTk.PhotoImage(resized_img)
    task1histogramimg = tk.Label(image=task1histogram)
    task1histogramimg.grid(row=14, column=4,pady=20)



    def getSourceStationsFromDropDown(*args):
        """This function helps when user clicks on source station line this will only show all the stations present in that line """
        print('Stations Changes S', sourceStationLine.get())
        changesSlabel = sourceStationLine.get()
        sStationNameMenu["menu"].delete(0, 'end')
        optionsStationsSource = verticesDataFromDataFrame[verticesDataFromDataFrame['lines'] == changesSlabel]
        optionsStationsSource = optionsStationsSource['source']
        for ss in optionsStationsSource:
            sStationNameMenu['menu'].add_command(label=ss, command=tk._setit(sourceStationName, ss))


    sourceStationLine.trace('w', getSourceStationsFromDropDown)


    def getDestinationStationsFromDropDown(*args):
        """This function helps when user clicks on destination station line this will only show all the stations present in that line"""
        print('Stations Changes S', destinationStationLine.get())
        changesSlabel = destinationStationLine.get()
        dStationNameMenu["menu"].delete(0, 'end')
        optionsStationsDestination = verticesDataFromDataFrame[verticesDataFromDataFrame['lines'] == changesSlabel]
        optionsStationsDestination = optionsStationsDestination['source']
        for ss in optionsStationsDestination:
            dStationNameMenu['menu'].add_command(label=ss, command=tk._setit(destinationStationName, ss))


    destinationStationLine.trace('w', getDestinationStationsFromDropDown)


    def getPathTime():
        """this functions calls to get shortest path"""
        print(sourceStationLine.get(), ' to ', destinationStationLine.get())
        print(sourceStationName.get(), ' to ', destinationStationName.get())
        # lets create our adjacency matrix for graph representation



        # lets see our dijkstra part
        numberOfuniqueVertices = len(adjacencyMatrix.uniqueVertices)

        sourceName = sourceStationName.get()
        distName = destinationStationName.get()
        srcIndex = adjacencyMatrix.uniqueVertices.index(sourceName)
        ditIndex = adjacencyMatrix.uniqueVertices.index(distName)
        print("Source is  ", sourceName, " Destination is ", distName)
        print("Source is  ", adjacencyMatrix.uniqueVertices[srcIndex], " Destination is ",
              adjacencyMatrix.uniqueVertices[ditIndex])

        dijkstra = DijekstraAlgorithem(adjacencyMatrix.adjacencyMatrix, adjacencyMatrix.uniqueVertices[srcIndex],
                                        adjacencyMatrix.uniqueVertices[ditIndex], numberOfuniqueVertices,
                                        adjacencyMatrix.uniqueVertices)
        dijkstra.initializeMatrices()
        dijkstra.dijkstra_algorithm()
        dijkstra.displayPath()

        route = dijkstra.getPathFromSourceToIndex()
        time = dijkstra.getTimeConsumedFromSourceToDistination()
        route = route[::-1]
        print('this is the path', route)
        print('this is the time taken form source to destination ', time)
        #pathText.config(text='\n'.join(map(str, route)))
        #timeText.config(text=str(time))
        print('List OF Stations which can be discarded ')



        #dijkstra.createHistogramOfAllStationsPairs()
        pathmagic=my_magic_output(route)
        pathmagic+="\n \n \n "+'This is Total Time Taken   =     '+str(time)
        #this pop up will show the path in a message box
        messagebox.showinfo('Shortest Path and Time taken',pathmagic)

    def show_data_of_gov_helper():
        """This function displays all the stations which can be closed on a message box"""
        gov_helper = HelpGovernmetByRemovingStations(len(edges), edges, adjacencyMatrix.uniqueVertices)
        gov_helper.find_extra_stations()
        print(gov_helper.get_can_discard_stations())
        gvrom = gov_helper.get_can_discard_stations()

        my_magic_gov_data=my_magic_output(gvrom)
        gov_station_can_remove =my_magic_gov_data
        messagebox.showinfo("These Station Can Be Discarded ",gov_station_can_remove)

    def show_Data_of_Task3():
        """This function will show the bridges of graph """
        numberOfuniqueVertices = len(adjacencyMatrix.uniqueVertices)

        sourceName = "Harrow & Wealdstone"
        distName = "Harrow & Wealdstone"
        srcIndex = adjacencyMatrix.uniqueVertices.index(sourceName)
        ditIndex = adjacencyMatrix.uniqueVertices.index(distName)


        dijkstra = DijekstraAlgorithem(adjacencyMatrix.adjacencyMatrix, adjacencyMatrix.uniqueVertices[srcIndex],
                                        adjacencyMatrix.uniqueVertices[ditIndex], numberOfuniqueVertices,
                                        adjacencyMatrix.uniqueVertices)
        dijkstra.initializeMatrices()
        dijkstra.dijkstra_algorithm()
        dijkstra.displayPath()




        result=dijkstra.task3augmentingtask1()
        result=my_magic_output(result)



        messagebox.showinfo("Task 3 is Augmenting Task 1 To Find All Bridges of Graph",result)
    #button for shortest time and path
    buttonTask1 = tk.Button(root, text='Get Path and Time ', command=getPathTime)
    buttonTask1.grid(row=8, column=4)
    #button for closing the stations
    buttonTask2 = tk.Button(root, text='Click To See Which Stations Can Be Closed To Help Government', command=show_data_of_gov_helper)
    buttonTask2.grid(row=16, column=4,pady=20)
    #Button for showing bridges of graph
    buttonTask3 = tk.Button(root, text='Click To See Task 3 To Find Bridges Of Graph  ', command=show_Data_of_Task3)
    buttonTask3.grid(row=17, column=4,pady=20)

    root.mainloop()





