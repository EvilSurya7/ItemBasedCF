

import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)

ratings.head()

userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
userRatings.head()

correlationMatrix = userRatings.corr()
correlationMatrix.head()

correlationMatrix = userRatings.corr(method='pearson', min_periods=100)
correlationMatrix.head()

myRatings = userRatings.loc[0].dropna()
myRatings


simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print ("Adding sims for " + myRatings.index[i] + "...")
    
    sims = correlationMatrix[myRatings.index[i]].dropna()
    # Scale its similarity by how well I rated this movie
    sims = sims.map(lambda x: x * myRatings[i])
    # Add the score to the list of similarity candidates
    simCandidates = simCandidates.append(sims)
    
print ("sorting...")
simCandidates.sort_values(inplace = True, ascending = False)

simCandidates = simCandidates.groupby(simCandidates.index).sum()

simCandidates.sort_values(inplace = True, ascending = False)
simCandidates.head(10)

filterSim = simCandidates.drop(myRatings.index)
print(filterSim.head(10))