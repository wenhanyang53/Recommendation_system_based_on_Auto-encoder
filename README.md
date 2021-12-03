## Recommendation system based on Auto-encoder

We created a simple deep learning algorithm to suggest the events that might interest user by using Tensorflow. The data is coming from Toulouse council. There is plenty of information in this data, but we only used the columns that can define the event such as Description, Category, Type and Theme. Based on the columns we chose, we one-hot coded all the features and trained a neural network model using TensorFlow. To get a suggestion, users must choose their interested events and we use the events as an input to our model and calculate the centroid of them and suggest which events have the shortest distance to it. This process is called “data based general recommendation”.

## Features

We only used a few features that could define the type of the event. The original data is in the repository.

![features](document/features.png 'features')

## Algorithm

Using Auto-encoder can provide the low dimension representation from the original features which can help us to choose the short distance between the events and the events that intreste the user.

![Auto-encoder](document/autoencoder.png 'Auto-encoder')

We used Tensorflow to develope the Auto-encoder and here is the result.

![Accuracy](document/accuracy.png 'Accuracy')

Noticed that we didn't have many data to train this neural network so the result wasn't as good as we expected but the recommendation performance was satisfying.

## Result

![Result](document/result.png 'Result')

We draw a graph that represents all the events and orange dots represent the events the user chose and red dot represents the centroid. To suggest the new events, we calculate the Euclidean distance between the events and centroid and we recommend the closet events to the user.

In this example, user has chosen “Musique”, “Nature et détente” and “Sports et loisirs” as type, “concert”, “Animations”, “Spectacle” etc as category and “Fleurs, plantes”, “Bio, fleurs, plantes” and “Musique classique” as the theme. The suggestion includes “Musique”, “Sports et loisirs”, “Concert”, “Animation” and “Bio, fleurs, plantes” and etc. It not only suggests the type, category and theme that the user chose, it also suggests more types of events that the user could be interested in. It is more likely that a user who chose to go to “Musique” and “Concert” would love to go to “Culturelle” events.

