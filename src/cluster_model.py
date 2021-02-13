
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import logging 
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path 
from abc import ABC,abstractmethod


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
    def _save_to_bytes_image(plt_: plt):
        bytes_image = io.BytesIO()
        plt_.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        return bytes_image

    def KDE_plot(self): 
        n = len(self.dataframe.columns)
        plt.figure(figsize=(10,50))
        for i in range(len(self.dataframe.columns)):
            plt.subplot(n, 1, i+1)
            sns.distplot(self.dataframe[self.dataframe.columns[i]], kde_kws={"color": "b", "lw": 3, "label": "KDE"}, hist_kws={"color": "g"})
            plt.title(self.dataframe.columns[i])
        plt.tight_layout()
        return self._save_to_bytes_image(plt)

    def correlation_plot(self):
        correlations = self.dataframe.corr()
        f, ax = plt.subplots(figsize = (20, 20))
        sns.heatmap(correlations, annot = True)
        return self._save_to_bytes_image(plt)

    def elbow_plot(self, clusters_scores: list[int] = []):
        if not  clusters_scores:
            clusters_scores = self._generate_clusters_score()
        plt.plot(clusters_scores, 'bx-')
        plt.title('Optimize the number of clusters')
        plt.xlabel('Clusters')
        plt.ylabel('Scores') 
        return self._save_to_bytes_image(plt)

    def _generate_clusters_score(self, no_of_clusters_list: list[int] = []):
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
        cluster_centers = self.scaler.inverse_transform(cluster_centers)
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
        return self._save_to_bytes_image(plt)


class ICleaner(ABC):

    @abstractmethod
    def clean_data():
        pass

class MarketingCSVCleaner(ICleaner):

    def __init__(self):
        self.logger = self._initLogger()

    def clean_data(self, dataframe: pd.DataFrame):
        dataframe = self._drop_CUST_ID(dataframe)
        # The two columns below do not contribute the model for this kind of dataset 
        dataframe = self._set_null_to_mean(dataframe,columns = ['MINIMUM_PAYMENTS','CREDIT_LIMIT'])

    def _set_null_to_mean(self, dataframe: pd.DataFrame , columns: list[str]):
        '''Mean is chosen based on the inherent nature of the expected data'''
        for column in columns: 
            if column in dataframe.columns:
                try:
                    dataframe.loc[(dataframe[column].isnull() == True ), column] = dataframe[column].mean()
                    
                except:
                    self.logger.exception("Column to set does not exists in the dataframe")
        return dataframe 

    def _drop_CUST_ID(self,dataframe) -> pd.DataFrame:
        '''CUST_ID do not provide any useful information for the model'''
        try: 
            dataframe.drop("CUST_ID", axis = 1, inplace= True)
        except: 
            self.logger.exception("Fail to drop CUST_ID. Warning: Column may not exists")
        return dataframe 
    
    def _initLogger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('model.log')
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')      
        file_handler.setFormatter(formatter)  
        logger.addHandler(file_handler) 
        return logger

