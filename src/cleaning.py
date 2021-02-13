
import logging 
import pandas as pd
from abc import ABC,abstractmethod


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

