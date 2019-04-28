# Multinomial-Naive-Bayes
A simple multinomial naive bayes classifier that I made for an assignment in my machine learning paper at university.

I have a training set composed of a number of abstracts from various scentific research papers written on proteins. Each protein is found either in Archaea, Bacteria, Eukaryota, or Viri. 
So the class can take on {'Arachnae', 'Bacteria', 'Eukaryota', 'Viri'}
I have hard coded a multinomial naive bayes classifier that is able to predict the class from the abstract of unseen research papers.

The classifier has 96.666% accuracy on unseen training data.

I'm sure there are more pythonic/faster ways to do what I have done - especially the data cleaning and preparation part.
