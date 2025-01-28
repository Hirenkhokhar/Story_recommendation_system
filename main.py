import numpy as np
import pandas as pd   
from surprise import SVD,Dataset, Reader
from surprise.accuracy import rmse  
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


df  = pd.read_csv("metadata.csv")  
df.drop(["updated_at","published_at"],axis = 1,inplace = True)  
df1 = pd.read_csv("user_interaction.csv")  
df1.drop(["updated_at"],axis = 1,inplace =True)

genre = df.groupby('pratilipi_id').agg({
    'category_name': lambda x: list(set(x)),  
    'author_id': 'first',              
    'reading_time': 'first'                 
}).reset_index()

genre['genre'] = genre['category_name']


df = df.merge(genre[['pratilipi_id', 'genre']], on='pratilipi_id', how='left')

df = df.drop_duplicates(subset='pratilipi_id').reset_index(drop=True)

df = df.drop(columns=['category_name'])

df5 = pd.merge(df,df1,on = "pratilipi_id")
   


interaction_data = df5[['user_id', 'pratilipi_id', 'read_percent']]

item_metadata = df5[['pratilipi_id', 'genre', 'author_id']]

reader = Reader(rating_scale=(0, 100))
data_surprise = Dataset.load_from_df(interaction_data, reader)

trainset, testset = train_test_split(data_surprise, test_size=0.25, random_state=42)

svd_model = SVD()

svd_model.fit(trainset)

svd_predictions = svd_model.test(testset)
svd_rmse = rmse(svd_predictions)
print(f"SVD RMSE: {svd_rmse}")     


item_metadata['genre'] = item_metadata['genre'].apply(tuple)
item_metadata= item_metadata.drop_duplicates()
item_metadata['genre'] = item_metadata['genre'].apply(list)
item_metadata['combined_features'] = item_metadata.apply(lambda x: f"{x['genre']} {x['author_id']}", axis=1)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(item_metadata['combined_features']) 

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_metadata['pratilipi_id'])}  

def get_content_based_recommendations(pratilipi_id, top_n=5):
    idx = item_id_to_index[pratilipi_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[2:top_n + 2] 
    recommended_items = [item_metadata.iloc[i[0]]['pratilipi_id'] for i in sim_scores]
    
    return recommended_items 


def hybrid_recommendations(user_id, top_n=5):
    cf_predictions = []
    for pratilipi_id in item_metadata['pratilipi_id']:
        pred = svd_model.predict(user_id, pratilipi_id)
        cf_predictions.append((pratilipi_id, pred.est))
 
    cf_predictions = sorted(cf_predictions, key=lambda x: x[1], reverse=True)[:top_n] 
    
    top_cf_item = cf_predictions[0][0]
    cb_recommendations = get_content_based_recommendations(top_cf_item, top_n=top_n)
    

    hybrid_recommendations = list(set([x[0] for x in cf_predictions] + cb_recommendations))[:top_n]
    return hybrid_recommendations

user_id = 5506791962844778
recommendations = hybrid_recommendations(user_id, top_n=5)
print(f"Hybrid Recommendations for user {user_id}: {recommendations}")  
