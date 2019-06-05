import pydot

from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.externals.six import StringIO

from IPython.display import Image  

def treeGraph(classifier, data_frame, labels):
        dot_data = StringIO()
        tree.export_graphviz(classifier, out_file=dot_data) 
        graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
        #graph[0].write_pdf('cancer.png')
        dot_data = StringIO()  
        export_graphviz(classifier, out_file=dot_data,    
                        feature_names=labels,  
                        class_names=data_frame['target_names'],  
                        filled=True, rounded=True,  
                        special_characters=True) 
        graph = pydot.graph_from_dot_data(dot_data.getvalue())  
        return Image(graph[0].create_png())