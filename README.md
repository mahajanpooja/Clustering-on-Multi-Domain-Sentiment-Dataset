# Clustering-on-Multi-Domain-Sentiment-Dataset
Applied different clustering methods (K-means, Hierarchical, DBSCAN) to find the best model with optimal parameters


1. Data Setup (Load and Preprocessing)
    
    1.1   The following steps were performed for data loading and preprocessing:
        
      *   Reading the dataset: The dataset was read using file handling techniques (open () function). Each line was split into texts which contained feature:<count> .... feature:<count> and #label#:<label> at the end of the line. Using separator of colon (:), feature and it’s count were separated into a dictionary D, containing feature as keys and it’s count as values. Using a condition (key!='#label#':) in an if-else loop, the features were assigned to the ‘text’ column of the ‘output’ array whereas the labels were assigned to the ‘label’ column of the array. The array was further converted into a dataframe, df.
  
      *   Tokenization and stemming: In order to break down the text into sentences and text and to construct a numerical reprsentation, we used tokenization and stemming using nltk library. Non-letter characters were removed using re (regular expressions) followed by stemming.
      *   Tf-idf vectorization: Next, we created a dataframe with tf-idf vectorized features. This method was chosen over CountVectorizer() or DictVectorizer() because some words which occur frequently in one document and less frequently in others teach the model about the importance of that word in the document penalizing common words. On the other hand, CountVectorizer() gives bias result after sorting out the frequency count of the words.

  2. K-means Clustering

      *   Training the dataset: Initially we trained the k-means model for certain values of k, using scikit learn which uses “Euclidean Distance” to cluster similar data points[2]. Using elbow method for parameter selection, we could identify the optimal value of clusters; however, this value of k is just for reference.
      
      *   Similarity measures: We took Euclidean and Cosine distances because they have an interrelated as explained further. Euclidean distance is calculated by taking the square root of square differences between the coordinates of a pair of data points as shown in equation (1) given below. Cosine distance: It measures cosine of the angle between two vectors as shown in equation (2)given by the following formula[36]. In this equation, theta (θ) reprsents the angle between two vectors and A, B are n dimensional. We can convert Euclidean distance as a proportionate measure of cosine distance using equation (3).


             𝐷𝑖𝑠𝑡𝑋𝑌 = 𝑚𝑎𝑥𝑘 𝑋𝑖𝑘 − 𝑋𝑗𝑘 ...........................................................................equation (1) 

             vectors. 𝜃 = 𝑎𝑟𝑐𝑐𝑜𝑠 𝐴. 𝐵 𝐴 𝐵........................................................................equation (2) 

             Euclidean vector(u,v)=2*(1-cosine_similarity(u,v).....................................equation(3)
  
      *   Evaluation methods: We have computed the Silhouette score and Davies Bouldin score for cluster evaluation. Different values were computed for each evaluation method based on the distance metric. The values of Silhouette score range from -1 to 1, with -1 being the worse (samples allocated to wrong cluster), 1 being the best (samples are clustered appropriately) and 0 indicating overlapping clusters [1]. Davies Bouldin Score is based on the ratio of distances within the cluster and distances between the cluster. The lower the value of this ratio (minimum value=0), the better the formation of the clusters [2].
     
      

#### K-means Scatter Plot: 
<p align="center">
  <img src="/images/K-means-scatterplot.png" width="500" alt="K-means Scatter Plot">
</p>                                                              
    
Similarity measure selection: Euclidean distance and Cosine distance. 
The optimal k selection: Scores for Euclidean metric are better than Cosine for both the evaluation methods, hence, k=3 (closer to k=4 as shown in elbow method below) 
    
#### Clustering Validation/Evaluation methods:
<p align="center">
  <img src="/images/clustering-evaluation.png" width="1000" alt="Clustering Validation/Evaluation">
</p> 
    
#### K-means Runtime:
 <p align="center">
  <img src="/images/kmeans-runtime.png" width="500" alt="Computational complexity">
</p>  
  
I personally, find Davies Bouldin score to be better than Silhouette Score because of the following reasons:

1. Mathematical interpretation: Davies Bouldin score is better because it tries to maximize the numerator (inter cluster distance) and decreases the denominator (intra cluster distance). On the other hand, Silhouette Coefficient is calculated using the average of distances of points within the cluster (intra-cluster distance) and the average distance of a nearest-cluster distance (b) for each data point.
  
2.    Computational complexity: The computational complexity of Silhouette is more than Davies Bouldin.
        
3.    Visual Representation: Silhouette score values irrespective of the distance metric, are closer to zero, indicating overlapping clusters. Compared to that, clusters evaluated by Davies Bouldin score show separated clusters.
    
#### K-means Silhouette Score:
 <p align="center">
  <img src="/images/kmeans-silhouette.png" width="500" alt="Computational complexity">
</p>  

#### K-means Davies Bouldin Score:
 <p align="center">
  <img src="/images/kmeans-davies-bouldin.png" width="500" alt="Computational complexity">
</p>  

  ```
  Result: Similarity measure: Euclidean | Optimal value of k = 3| Evaluation method: Davies Bouldin Score
  ```

  #### Hierarchical Clustering | Dendrogram: 
<p align="center">
  <img src="/images/dendrogram.png" width="500" alt="Dendrogram">
</p>   
 
  3. Hierarchical Clustering  
    
      *   Training the model: Initially we trained the model, by computing the distance matrix from the tfidf_matrix. Using the linkage () method in scipy library, all the three linkage methods ={single, complete, average} were passed and their cophenet coefficients were computed. Since the maximum value of cophenet coefficient is used to determine the best linkage method, we created a dendrogram for ‘average’ linkage method [3].
  
      *   Cluster evaluation: We have computed the Silhouette score (reason stated above in k-means) and Calinski Harabasz score for cluster evaluation. Different values were computed for each evaluation method based on the distance metric. Calinski Harabasz Score represents the degree of dispersion ie. how well the data points are spread within the cluster and how far the clusters are from each other. The denser clusters have high value of Calinski Harabasz Score with it’s values starting from zero and increasing without any upper limit.

      *    Similarity measure selection: Euclidean, cosine, Manhattan
    
      *    Best distance metrics: We get average linkage as the best method and optimal number of clusters k=4 from both the algorithms.
    
      *    Clustering Validation/Evaluation: The highest value of Silhouette Score amongst the nine value is 0.212377 for k=4. As per the table, the highest value of Calinski Harabasz Score amongst the nine values is 1330.39403572466 for k=4.
    
      *    Comparing Evaluation Methods: 
 
        I personally, find Calinski Harabasz score to be better than Silhouette Score because of the following reasons:
    
        1. Mathematical interpretation: Calinski Harabasz score is better because it tries to maximize the numerator (inter cluster dispersion) and decreases the denominator (intra cluster dispersion). On the other hand, Silhouette Coefficient is calculated using the average of distances of points within the cluster (intra-cluster distance) and the average distance of a nearest-cluster distance (b) for each data point.
    
        2. Computational complexity: The computational complexity of Silhouette is more than Calinski Harabasz.
    
        3. Visual Representation: Silhouette score values irrespective of the distance metric, are closer to zero, indicating overlapping clusters. Compared to that, clusters evaluated by Calinski Harabasz score show separated clusters.

#### Silhouette Score and Davies Bouldin Score:
 <p align="center">
  <img src="/images/hierarchical-silhouette.png" width="700" alt="Silhouette score and Davies Bouldin Score ">
</p>

#### Runtime for Silhouette Score:
 <p align="center">
  <img src="/images/runtime-silhouette.png" width="300" alt="Runtime for Silhouette score">
</p>

#### Runtime for Calinski Harabasz Score:
 <p align="center">
  <img src="/images/runtime-calinski.png" width="300" alt="Runtime for Davies Bouldin Score">
</p>
    
   ```
    Result: We get average linkage as the best method and optimal number of clusters k=4 from both the algorithms.
    Similarity measure: Cosine | Optimal value of k = 4| Best distance metrics: Average | Evaluation method: Calinski Harabasz Score
    
   ```
    
  4. DBSCAN
            
        *    The optimal values of model parameters: min_points=1000 and eps= 0.7930339190381753 (found by visualizing the maximum point of curvature)
    
        *    Similarity measure selection: Euclidean, Cosine, Manhattan
    
        *    Clustering Validation/Evaluation: We chose Silhouette Score as the evaluation method because it takes all the distance metrics (Euclidean, Cosine, Manhattan) into account (Table 9). Table 9: DBSCAN (Dataset 1)

#### Silhouette Score for DBSCAN:
 <p align="center">
  <img src="/images/dbscan.png" width="500" alt="Runtime for Davies Bouldin Score">
</p>

 ```
    Result: Similarity measure: Manhattan | Optimal value of k = 4| Evaluation method: Silhouette Score
```
    
  5. The Best Model

#### Best model for all the 3 algorithms:
 <p align="center">
  <img src="/images/best-model.png" width="900" alt="Best Model">
</p>    
  

### References:

[1]"sklearn.metrics.silhouette_score — scikit-learn 0.24.2 documentation", Scikit-learn.org, 2021. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html. [Accessed: 30- May- 2021].

[2]"sklearn.metrics.davies_bouldin_score — scikit-learn 0.24.2 documentation", Scikit-learn.org, 2021. [Online]. Available: https://scikit- learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html?highlight=davies%20bouldin%20score. [Accessed: 30- May- 2021].

[3]"scipy.cluster.hierarchy.cophenet — SciPy v0.16.1 Reference Guide", Docs.scipy.org, 2021. [Online]. Available: https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.cluster.hierarchy.cophenet.html. [Accessed: 30- May- 2021].

[4]"Hierarchical Clustering Model in 5 Steps with Python", Medium, 2021. [Online]. Available: https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318. [Accessed: 30- May- 2021].
  
[5]"tirthajyoti/Machine-Learning-with-Python", GitHub, 2021. [Online]. Available: https://github.com/tirthajyoti/Machine- Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/Clustering_metrics.ipynb. [Accessed: 30- May- 2021].

[6]"Cheat sheet for implementing 7 methods for selecting the optimal number of clusters in Python", Medium, 2021. [Online]. Available: https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal- number-of-clusters-in-python-898241e1d6ad. [Accessed: 30- May- 2021].

[7]"Clustering", Wilshireliu.com, 2021. [Online]. Available: https://www.wilshireliu.com/clustering. [Accessed: 30- May- 2021].

[8]W. criterion?, "What is an acceptable value of the Calinski & Harabasz (CH) criterion?", Cross Validated, 2021. [Online]. Available: https://stats.stackexchange.com/questions/52838/what-is-an-acceptable-value-of-the-calinski- harabasz-ch-criterion. [Accessed: 30- May- 2021].

[9]"DBSCAN Python Example: The Optimal Value For Epsilon (EPS)", Medium, 2021. [Online]. Available: https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python- example-3100091cfbc. [Accessed: 30- May- 2021].

   
