import pandas as pd 
from pathlib import Path 
from cleaning import MarketingCSVCleaner 
from cluster_model import ClusteringModel


def run_sample(file_location = None): 
    if not file_location:
        filedir = Path(__file__).parent
        relative_path = 'tests/marketing_data.csv'
        file_location = (filedir / relative_path).resolve()
    df = pd.read_csv(file_location)
    cleaner = MarketingCSVCleaner()
    cleaner.clean_data(df)
    model = ClusteringModel(df)
    run_plots_on(model)

def run_plots_on(model: ClusteringModel):
    model.correlation_plot() 
    model.elbow_plot() 
    model.PCA2_plot() 
    model.KDE_plot() 
    
if __name__ == "__main__": 
    run_sample()