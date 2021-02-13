
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from seaborn.utils import relative_luminance
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import List 
from pathlib import Path

class ClusteringModel: 
    
    '''Provide 6 functionalities for unsupervised segmentation data return in png format. 
        KDE_plot (Kernel Density Estimate);
        Correlation_plot (Correlation Matrix);
        Elbow_plot (WCSS Vs Number of Clusters - for choosing optimal number of clusters);
        centroids (Customer groups information in dataframe format)
        PCA2_plot (Apply standard 2 components PCA plot)
    '''


    def __init__(self,dataframe: pd.DataFrame):
        self.dataframe = dataframe 
        self.scaler = StandardScaler()
        self.scaled_arr = self.scaler.fit_transform(self.dataframe)


    @staticmethod
    def _save_to_bytes_image(plot: plt):
        bytes_image = io.BytesIO()
        plot.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        return bytes_image

    @staticmethod
    def _save_to_png_image(plot:plt, filename):
        # write to disk isntead - this is used as alternative route for a different use case 
        filedir = Path(__file__).parent
        relative_path = 'results'
        save_location = (filedir / relative_path / filename ).resolve()
        plot.savefig(save_location, format='png')
        

    def KDE_plot(self): 
        n = len(self.dataframe.columns)
        plt.figure(figsize=(10,50))
        for i in range(len(self.dataframe.columns)):
            plt.subplot(n, 1, i+1)
            sns.distplot(self.dataframe[self.dataframe.columns[i]], kde_kws={"color": "b", "lw": 3, "label": "KDE"}, hist_kws={"color": "g"})
            plt.title(self.dataframe.columns[i])
        plt.tight_layout()
        self._save_to_png_image(plt,"KDEPlot.png")
        return self._save_to_bytes_image(plt)

    def correlation_plot(self):
        correlations = self.dataframe.corr()
        f, ax = plt.subplots(figsize = (20, 20))
        sns.heatmap(correlations, annot = True)
        self._save_to_png_image(plt,"CorrelationPlot.png")
        return self._save_to_bytes_image(plt)

    def elbow_plot(self, clusters_scores: List[int] = []):
        if not  clusters_scores:
            clusters_scores = self._generate_clusters_score()
        plt.figure(figsize=(15,15))
        plt.plot(clusters_scores, 'bx-')
        plt.title('Optimize the number of clusters')
        plt.xlabel('Clusters')
        plt.ylabel('Scores') 
        self._save_to_png_image(plt,"ElbowPlot.png")
        return self._save_to_bytes_image(plt)

    def _generate_clusters_score(self, no_of_clusters_list: List[int] = []):
        clusters_scores = []
        if not no_of_clusters_list:
            # choosing arbitary number of clusters to expiriment as default 
            no_of_clusters_list = list(range(1,21))
        for i in no_of_clusters_list:
            kmeans = self._generate_KMeans(i)
            clusters_scores.append(kmeans.inertia_)
        return clusters_scores 

    def _generate_KMeans(self, no_of_clusters: int) -> pd.DataFrame:
        _kmeans = KMeans(no_of_clusters)
        _kmeans.fit(self.scaled_arr)
        return _kmeans 

    def centroids(self, no_of_clusters: int = 8) -> pd.DataFrame :
        _kmeans = self._generate_KMeans(no_of_clusters)
        cluster_centers = pd.DataFrame(data = _kmeans.cluster_centers_,columns = [self.dataframe.columns])
        cluster_centers = self.scaler.inverse_transform(cluster_centers) #inverse can only be called only after the data is fitted.
        return pd.DataFrame(data = cluster_centers, columns = [self.dataframe.columns])

    def _generate_PCA(self, no_of_clusters: int, no_of_pca: int):
        '''2-Components PCA plot chosen as this is the standard norm and based on the nature of marketing csv.'''
        kmeans = self._generate_KMeans(no_of_clusters)
        principal_comp = PCA(no_of_pca).fit_transform(self.scaled_arr)
        pca_df = pd.DataFrame(data = principal_comp, columns =['PCA1','PCA2'])
        pca_df = pd.concat([pca_df,pd.DataFrame({'cluster': kmeans.labels_})], axis = 1)
        return pca_df

    def PCA2_plot(self, no_of_clusters = 8):
        '''if no_of_clusters is not specified, number 8 will be chosen. See sklearn doc. '''
        pca_df = self._generate_PCA(no_of_clusters, no_of_pca =2)
        plt.figure(figsize=(10,10))
        ax = sns.scatterplot(x="PCA1", y="PCA2", hue = "cluster", data = pca_df, palette =['red','green','blue','pink','yellow','gray','purple', 'black'])
        self._save_to_png_image(plt,"PCAPlot.png")
        return self._save_to_bytes_image(plt)

