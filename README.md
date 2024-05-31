# Machine-Learning-methods-for-Text-Classification-using-the-IMDB-corpus

This project was part of the TKO_7095 Introduction to Human Language Technology course at the Universiy of Turku.

Authored by :
- Hiba Daafane
- Parisa Piran

The project involved creating a machine learning-based method for a binary sentiment classification task using a Large Movie Review Dataset (IMDB). Specifically, the main tasks 
within this project included :  
 
- Selecting a text classification corpus to work on.
- Read the paper presenting the corpus as well as any other relevant published materials about the corpus to assure a good understanding of the data
- Identify what is the state-of-the-art performance for this corpus (i.e. the best published results).
- Write code to :
   - download the corpus
   - perform any necessary preprocessing
   - train a machine learning method on the training set, evaluating performance on the validation set
   - perform hyperparameter optimization
   - evaluate the final model on the test set
## Results and summary

### 1. Corpus insights

IMDB dataset is a Large Movie Review Dataset provided by Stanford. This dataset is quiet popular and is often used as a benchmark for sentiment analysis and text classification tasks. The dataset contains a collection of movie reviews, where each review is labeled with either a positive or negative sentiment.

It's important to not that the version that was used for this work is relatively large, containing about 25,000 movie reviews for training, and another 25,000 for testing. With an additional unlabeled data for use in unsupervised learning setting as well (which will be dismissed in this work). However, due to the limited computational power, we used only 20,000 instances for training and 5000 for validation and test.

### 2. Results

In this project, we used the pre-trained model "distilbert-base-uncased" for initializing both the tokenizer and the classifier. Training the classifier was a bit of a tricky task, we mainly faced two issues, the first issue was that we noticed that our initial values for the model's hyperparameters were causing the model to overfit, we ended up having very high values for accuracy, with a train loss that almost reaches 60%, and before going into the hyperparameter tuning, we tried different combinations of parameters until we got some satisfying results, where our model achieved a 92.6% accuracy, with 0.22 loss on the validation data. However, after the Hyperparameter optimization process, we achieved a best accuracy of 92.7%, a validation loss of 0.20, and a training loss of 0.13.

### 3. Relation to state of the art

The accuracy our model achieved surpasses all of the mentioned material we discussed earlier, however a deeper dive into the internet helped us realize that the highest recorded accuracy achieved with the imdb Dataset is about 96.21% by [XLNet](https://arxiv.org/abs/1906.08237v2). So taking into account the extremly limited ressources we had to work with, and the constant crashing of memory, we believe that the obtained results are quiet satisfactory, and our model sits at about the top 20 recorded benchmark models.

## Bonus Task 

### 1. Annotating out-of-domain documents

For this task we chose a collection of 100 review comments gathered from amazon, the comments come from a variety of products in order to inssure the diversity of the vocabulary used to describe the products (clothing items, shoes, electronics, cosmetics..). The data contains an almost equal portion of negative and positive reviews.

For the annotation process we used the [INCEpTION](https://inception-project.github.io/) open source annotation platform. This tool is a product of work of the same team that developped WebAnno, that introduces a bit more flexibility and some extra features. The annotation process was quiet simple, all we had to do was upload our plain text, that contained the unannotated corpora, and then select each document (comment) and choose the appropriate label (0 for negative, and 1 for positive). Lastly, we had the option to download our annotated data in a variety of formats, and we ended up choosing the CoNLL format.

### 2. Results

Our final evaluation on the annotated dataset is quiet disappointing, with such a good model we ended up getting only 53% accuracy, which is barely better than a random guesser.

This kind of result is not really surprising, because pre-trained models (like our model) are usually trained on specific datasets, which capture the characteristics and patterns of the training data. When evaluating the model on a new dataset from a different text domain, even if the task is the same, the model may struggle to generalize well due to the differences in language style, vocabulary, topic distribution, or sentiment expressions.

This would be the same as trying to use BERT for example directly on our movie reviews dataset, without any previous training. And that is why fine-tuning is important when it comes to language models.

Another thing that should be taken into account is the task of manual annotation, the web is full of all sorts of corpora, and expressions of all kinds, so choosing what documents to collect for a certain task can also play a big role in the final results.
