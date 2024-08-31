import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
#from nltk.tokenize import tokenize
st.sidebar.header("Emotion Detection on Text Data Application")
st.write("This particular app has been made in order to detect emotion on restaurant, hotel and ... reviews. This way the user can quickly obtain a summarization of the emotion on the reviews.")
st.markdown("***Please upload the text file on CSV format***")


#crypto_name = st.sidebar.selectbox("Select Emotion Detection Algorithm", ("Logistic Regression", "Neural Networks", "RNN"))
value_name = st.sidebar.selectbox("Select Review Type", ("Restaurant", "Hotel", "Airlines"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("MultinomialNB", "RandomForest", "MultilayerPerceptron"))


#detector_name = st.sidebar.selectbox("Detection Method", ("Logistic Regression", "Neural Networks", "RNN"))
st.title(value_name)
st.title(classifier_name)

st.set_option('deprecation.showPyplotGlobalUse', False)

data = st.sidebar.file_uploader("Upload a CSV file", type ='csv')
if data is not None:
    new_df = pd.read_csv(data, on_bad_lines='skip')

    st.sidebar.markdown("Uploaded File Preview:")
    st.sidebar.write(new_df)

    st.sidebar.write("File Size:")
    st.sidebar.write(new_df.size)
    st.sidebar.write("File Shape:")
    st.sidebar.write(new_df.shape)
    
    #new_df = new_df.rename(columns={ new_df.columns[0]: "Review" }, inplace = True)


    if value_name == 'Restaurant':
        if classifier_name =='MultinomialNB':
            df=pd.read_csv('Restaurant reviews.csv', encoding = "ISO-8859-1")
            df = df.drop(columns=["Restaurant","Reviewer","Metadata","Time","Pictures"])
            y = df["Rating"]
            X = df.drop(columns=["Rating"])
            y = y.replace({'Like':3})
            y.isnull().sum()
            y = pd.to_numeric(y)
            y = y.fillna(y.median())

            st.header('Exploratory Data Analysis')

            st.markdown('**Character Count: The count of characters for each review**')

            length = len(new_df['Review'][0])
            st.write(f'Length of a sample review: {length}')

            new_df['Length'] = new_df['Review'].str.len()
            st.write(new_df.head(10))

            new_df['Length'].plot(kind="line", title="Character Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Characters')
            st.pyplot()

            st.markdown('**Word Count: The cound of words for each review**')

            word_count = new_df['Review'][0].split()
            st.write(f'Word count in a sample review: {len(word_count)}')

            def word_count(review):
                review_list = review.split()
                return len(review_list)
            
            new_df['Word_count'] = new_df['Review'].apply(word_count)
            st.write(new_df.head(10))

            new_df['Word_count'].plot(kind="line", title="Word Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Words')
            st.pyplot()


            st.markdown('**Mean Word Length: average length of words per review**')
            new_df['mean_word_length'] = new_df['Review'].map(lambda rev: np.mean([len(word) for word in rev.split()]))
            st.write(new_df.head(10))

            new_df['mean_word_length'].plot(kind="line", title="Mean Word Length Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Mean Word Length')
            st.pyplot()

            #st.markdown('Mean sentence length: Average length of the sentences in the review')
            #st.write(np.mean([len(sent) for sent in tokenize.sent_tokenize(new_df['Review'][0])]))

            #new_df['mean_sent_length'] = new_df['Review'].map(lambda rev: np.mean([len(sent) for sent in tokenize.sent_tokenize(rev)]))
            #st.write(new_df.head(10))

            st.markdown('**Reviews Wordcloud**')
            #Constructing the wordcloud for all the data
            comment_words = ''
            stopwords = set(STOPWORDS)
        
        # iterate through the csv file
            for val in new_df.Review:
            
            # typecaste each val to string
                val = str(val)
        
            # split the value
                tokens = val.split()
            
            # Converts each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            
            comment_words += " ".join(tokens)+" "
        
            wordcloud = WordCloud(max_words=15156633, width = 800, height = 800,
                        background_color ='white',
                        stopwords = stopwords,
                        min_font_size = 3).generate(comment_words)
        
            # plot the WordCloud image                      
            plt.figure(figsize = (8, 8), facecolor = None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad = 0)
        
            plt.show()
            st.pyplot()

            for i in range(0,len(y)):
                if (y[i]== 5.):
                    y[i] = "Positive"
                elif (y[i]== 4.5):
                    y[i] = "Positive"
                elif (y[i]== 4.):
                    y[i] = "Positive"        
                elif (y[i] == 3.5):
                    y[i] = "Neutral"
                elif (y[i]== 3.):
                    y[i] = "Neutral" 
                else:
                    y[i] = "Negative"

            for i in range(0,len(y)):
                if (y[i] == 'Positive'):
                    y[i] = 2
                elif (y[i] == 'Neutral'):
                    y[i] = 1
                else:
                    y[i] = 0

            y = y.astype('int')


            import re
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem.porter import PorterStemmer
            ps = PorterStemmer()
            corpus = []
            for i in range(0, len(X)):
                review = re.sub('[^a-zA-Z]',' ', str(X['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus.append(review) 

            corpus_test = []
            for i in range(0, len(new_df)):
                review = re.sub('[^a-zA-Z]',' ', str(new_df['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus_test.append(review) 

            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features=9000) #After experimenting with 7500, 5000, 2500 ...9000 worked best.
            X = cv.fit_transform(corpus).toarray()

            from sklearn.naive_bayes import MultinomialNB
            clf = MultinomialNB()
            clf.fit(X, y)

            X_test = corpus_test
            #st.write(X_test)

            X_test = cv.transform(X_test)
            #X_test = X_test.toarray()

            y_pred = clf.predict(X_test)
            #st.write(y_pred)

            new_df['Classification Result'] = y_pred.tolist()

            st.markdown('**CSV file with metrics and classification result per review**')
            #new_df['Classification Result'] = new_df['Classification Result'].replace('2', 'Positive', inplace=True)
            
            new_df.loc[new_df['Classification Result'] == 2, 'Classification Result'] = 'Positive'
            new_df.loc[new_df['Classification Result'] == 0, 'Classification Result'] = 'Negative'
            new_df.loc[new_df['Classification Result'] == 1, 'Classification Result'] = 'Neutral'

            st.write(new_df)

            def convert_df(df):
                return df.to_csv().encode("utf-8")

            csv = convert_df(new_df)

            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="derived_dataframe.csv",
            mime="text/csv",
            )

            st.markdown('**Classification Results Summarization**')

            #new_df["Classification Result"].value_counts(dropna=True).plot(kind="pie")

            counts = new_df['Classification Result'].value_counts(dropna=True)
            counts.plot.pie(autopct='%1.1f%%', labels=['Positive', 'Negative', 'Neutral'], shadow=True)

            plt.title('Emotion Percentage of Reviews')

            plt.axis('equal')
            plt.show()
            st.pyplot()

        elif classifier_name == 'RandomForest':
            df=pd.read_csv('Restaurant reviews.csv', encoding = "ISO-8859-1")
            df = df.drop(columns=["Restaurant","Reviewer","Metadata","Time","Pictures"])
            y = df["Rating"]
            X = df.drop(columns=["Rating"])
            y = y.replace({'Like':3})
            y.isnull().sum()
            y = pd.to_numeric(y)
            y = y.fillna(y.median())

            st.header('Exploratory Data Analysis')

            st.markdown('**Character Count: The count of characters for each review**')

            length = len(new_df['Review'][0])
            st.write(f'Length of a sample review: {length}')

            new_df['Length'] = new_df['Review'].str.len()
            st.write(new_df.head(10))

            new_df['Length'].plot(kind="line", title="Character Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Characters')
            st.pyplot()

            st.markdown('**Word Count: The cound of words for each review**')

            word_count = new_df['Review'][0].split()
            st.write(f'Word count in a sample review: {len(word_count)}')

            def word_count(review):
                review_list = review.split()
                return len(review_list)
            
            new_df['Word_count'] = new_df['Review'].apply(word_count)
            st.write(new_df.head(10))

            new_df['Word_count'].plot(kind="line", title="Word Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Words')
            st.pyplot()


            st.markdown('**Mean Word Length: average length of words per review**')
            new_df['mean_word_length'] = new_df['Review'].map(lambda rev: np.mean([len(word) for word in rev.split()]))
            st.write(new_df.head(10))

            new_df['mean_word_length'].plot(kind="line", title="Mean Word Length Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Mean Word Length')
            st.pyplot()

            #st.markdown('Mean sentence length: Average length of the sentences in the review')
            #st.write(np.mean([len(sent) for sent in tokenize.sent_tokenize(new_df['Review'][0])]))

            #new_df['mean_sent_length'] = new_df['Review'].map(lambda rev: np.mean([len(sent) for sent in tokenize.sent_tokenize(rev)]))
            #st.write(new_df.head(10))

            st.markdown('**Reviews Wordcloud**')
            #Constructing the wordcloud for all the data
            comment_words = ''
            stopwords = set(STOPWORDS)
        
        # iterate through the csv file
            for val in new_df.Review:
            
            # typecaste each val to string
                val = str(val)
        
            # split the value
                tokens = val.split()
            
            # Converts each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            
            comment_words += " ".join(tokens)+" "
        
            wordcloud = WordCloud(max_words=15156633, width = 800, height = 800,
                        background_color ='white',
                        stopwords = stopwords,
                        min_font_size = 3).generate(comment_words)
        
            # plot the WordCloud image                      
            plt.figure(figsize = (8, 8), facecolor = None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad = 0)
        
            plt.show()
            st.pyplot()

            for i in range(0,len(y)):
                if (y[i]== 5.):
                    y[i] = "Positive"
                elif (y[i]== 4.5):
                    y[i] = "Positive"
                elif (y[i]== 4.):
                    y[i] = "Positive"        
                elif (y[i] == 3.5):
                    y[i] = "Neutral"
                elif (y[i]== 3.):
                    y[i] = "Neutral" 
                else:
                    y[i] = "Negative"

            for i in range(0,len(y)):
                if (y[i] == 'Positive'):
                    y[i] = 2
                elif (y[i] == 'Neutral'):
                    y[i] = 1
                else:
                    y[i] = 0

            y = y.astype('int')


            import re
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem.porter import PorterStemmer
            ps = PorterStemmer()
            corpus = []
            for i in range(0, len(X)):
                review = re.sub('[^a-zA-Z]',' ', str(X['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus.append(review) 

            corpus_test = []
            for i in range(0, len(new_df)):
                review = re.sub('[^a-zA-Z]',' ', str(new_df['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus_test.append(review) 

            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features=9000) #After experimenting with 7500, 5000, 2500 ...9000 worked best.
            X = cv.fit_transform(corpus).toarray()

            from sklearn.naive_bayes import MultinomialNB
            clf = RandomForestClassifier()
            clf.fit(X, y)

            X_test = corpus_test
            #st.write(X_test)

            X_test = cv.transform(X_test)
            #X_test = X_test.toarray()

            y_pred = clf.predict(X_test)
            #st.write(y_pred)

            new_df['Classification Result'] = y_pred.tolist()

            st.markdown('**CSV file with metrics and classification result per review**')
            #new_df['Classification Result'] = new_df['Classification Result'].replace('2', 'Positive', inplace=True)
            
            new_df.loc[new_df['Classification Result'] == 2, 'Classification Result'] = 'Positive'
            new_df.loc[new_df['Classification Result'] == 0, 'Classification Result'] = 'Negative'
            new_df.loc[new_df['Classification Result'] == 1, 'Classification Result'] = 'Neutral'

            st.write(new_df)

            def convert_df(df):
                return df.to_csv().encode("utf-8")

            csv = convert_df(new_df)

            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="derived_dataframe.csv",
            mime="text/csv",
            )

            st.markdown('**Classification Results Summarization**')

            #new_df["Classification Result"].value_counts(dropna=True).plot(kind="pie")

            counts = new_df['Classification Result'].value_counts(dropna=True)
            counts.plot.pie(autopct='%1.1f%%', labels=['Positive', 'Negative', 'Neutral'], shadow=True)

            plt.title('Emotion Percentage of Reviews')

            plt.axis('equal')
            plt.show()
            st.pyplot()

        else:
            df=pd.read_csv('Restaurant reviews.csv', encoding = "ISO-8859-1")
            df = df.drop(columns=["Restaurant","Reviewer","Metadata","Time","Pictures"])
            y = df["Rating"]
            X = df.drop(columns=["Rating"])
            y = y.replace({'Like':3})
            y.isnull().sum()
            y = pd.to_numeric(y)
            y = y.fillna(y.median())

            st.header('Exploratory Data Analysis')

            st.markdown('**Character Count: The count of characters for each review**')

            length = len(new_df['Review'][0])
            st.write(f'Length of a sample review: {length}')

            new_df['Length'] = new_df['Review'].str.len()
            st.write(new_df.head(10))

            new_df['Length'].plot(kind="line", title="Character Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Characters')
            st.pyplot()

            st.markdown('**Word Count: The cound of words for each review**')

            word_count = new_df['Review'][0].split()
            st.write(f'Word count in a sample review: {len(word_count)}')

            def word_count(review):
                review_list = review.split()
                return len(review_list)
            
            new_df['Word_count'] = new_df['Review'].apply(word_count)
            st.write(new_df.head(10))

            new_df['Word_count'].plot(kind="line", title="Word Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Words')
            st.pyplot()


            st.markdown('**Mean Word Length: average length of words per review**')
            new_df['mean_word_length'] = new_df['Review'].map(lambda rev: np.mean([len(word) for word in rev.split()]))
            st.write(new_df.head(10))

            new_df['mean_word_length'].plot(kind="line", title="Mean Word Length Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Mean Word Length')
            st.pyplot()

            #st.markdown('Mean sentence length: Average length of the sentences in the review')
            #st.write(np.mean([len(sent) for sent in tokenize.sent_tokenize(new_df['Review'][0])]))

            #new_df['mean_sent_length'] = new_df['Review'].map(lambda rev: np.mean([len(sent) for sent in tokenize.sent_tokenize(rev)]))
            #st.write(new_df.head(10))

            st.markdown('**Reviews Wordcloud**')
            #Constructing the wordcloud for all the data
            comment_words = ''
            stopwords = set(STOPWORDS)
        
        # iterate through the csv file
            for val in new_df.Review:
            
            # typecaste each val to string
                val = str(val)
        
            # split the value
                tokens = val.split()
            
            # Converts each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            
            comment_words += " ".join(tokens)+" "
        
            wordcloud = WordCloud(max_words=15156633, width = 800, height = 800,
                        background_color ='white',
                        stopwords = stopwords,
                        min_font_size = 3).generate(comment_words)
        
            # plot the WordCloud image                      
            plt.figure(figsize = (8, 8), facecolor = None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad = 0)
        
            plt.show()
            st.pyplot()

            for i in range(0,len(y)):
                if (y[i]== 5.):
                    y[i] = "Positive"
                elif (y[i]== 4.5):
                    y[i] = "Positive"
                elif (y[i]== 4.):
                    y[i] = "Positive"        
                elif (y[i] == 3.5):
                    y[i] = "Neutral"
                elif (y[i]== 3.):
                    y[i] = "Neutral" 
                else:
                    y[i] = "Negative"

            for i in range(0,len(y)):
                if (y[i] == 'Positive'):
                    y[i] = 2
                elif (y[i] == 'Neutral'):
                    y[i] = 1
                else:
                    y[i] = 0

            y = y.astype('int')


            import re
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem.porter import PorterStemmer
            ps = PorterStemmer()
            corpus = []
            for i in range(0, len(X)):
                review = re.sub('[^a-zA-Z]',' ', str(X['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus.append(review) 

            corpus_test = []
            for i in range(0, len(new_df)):
                review = re.sub('[^a-zA-Z]',' ', str(new_df['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus_test.append(review) 

            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features=9000) #After experimenting with 7500, 5000, 2500 ...9000 worked best.
            X = cv.fit_transform(corpus).toarray()

            from sklearn.naive_bayes import MultinomialNB
            clf = MLPClassifier()
            clf.fit(X, y)

            X_test = corpus_test
            #st.write(X_test)

            X_test = cv.transform(X_test)
            #X_test = X_test.toarray()

            y_pred = clf.predict(X_test)
            #st.write(y_pred)

            new_df['Classification Result'] = y_pred.tolist()

            st.markdown('**CSV file with metrics and classification result per review**')
            #new_df['Classification Result'] = new_df['Classification Result'].replace('2', 'Positive', inplace=True)
            
            new_df.loc[new_df['Classification Result'] == 2, 'Classification Result'] = 'Positive'
            new_df.loc[new_df['Classification Result'] == 0, 'Classification Result'] = 'Negative'
            new_df.loc[new_df['Classification Result'] == 1, 'Classification Result'] = 'Neutral'

            st.write(new_df)

            def convert_df(df):
                return df.to_csv().encode("utf-8")

            csv = convert_df(new_df)

            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="derived_dataframe.csv",
            mime="text/csv",
            )

            st.markdown('**Classification Results Summarization**')

            #new_df["Classification Result"].value_counts(dropna=True).plot(kind="pie")

            counts = new_df['Classification Result'].value_counts(dropna=True)
            counts.plot.pie(autopct='%1.1f%%', labels=['Positive', 'Negative', 'Neutral'], shadow=True)

            plt.title('Emotion Percentage of Reviews')

            plt.axis('equal')
            plt.show()
            st.pyplot()

    if value_name == 'Hotel':
        if classifier_name == 'MultinomialNB':
            df=pd.read_csv('tripadvisor_hotel_reviews.csv')
            y = df["Rating"]
            X = df.iloc[:,:-1]
            y = pd.to_numeric(y)
            y = y.fillna(y.median())

            st.header('Exploratory Data Analysis')

            st.markdown('**Character Count: The count of characters for each review**')

            length = len(new_df['Review'][0])
            st.write(f'Length of a sample review: {length}')

            new_df['Length'] = new_df['Review'].str.len()
            st.write(new_df.head(10))

            new_df['Length'].plot(kind="line", title="Character Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Characters')
            st.pyplot()

            st.markdown('**Word Count: The cound of words for each review**')

            word_count = new_df['Review'][0].split()
            st.write(f'Word count in a sample review: {len(word_count)}')

            def word_count(review):
                review_list = review.split()
                return len(review_list)
            
            new_df['Word_count'] = new_df['Review'].apply(word_count)
            st.write(new_df.head(10))

            new_df['Word_count'].plot(kind="line", title="Word Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Words')
            st.pyplot()


            st.markdown('**Mean Word Length: average length of words per review**')
            new_df['mean_word_length'] = new_df['Review'].map(lambda rev: np.mean([len(word) for word in rev.split()]))
            st.write(new_df.head(10))

            new_df['mean_word_length'].plot(kind="line", title="Mean Word Length Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Mean Word Length')
            st.pyplot()

            #st.markdown('Mean sentence length: Average length of the sentences in the review')
            #st.write(np.mean([len(sent) for sent in tokenize.sent_tokenize(new_df['Review'][0])]))

            #new_df['mean_sent_length'] = new_df['Review'].map(lambda rev: np.mean([len(sent) for sent in tokenize.sent_tokenize(rev)]))
            #st.write(new_df.head(10))

            st.markdown('**Reviews Wordcloud**')
            #Constructing the wordcloud for all the data
            comment_words = ''
            stopwords = set(STOPWORDS)
        
        # iterate through the csv file
            for val in new_df.Review:
            
            # typecaste each val to string
                val = str(val)
        
            # split the value
                tokens = val.split()
            
            # Converts each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            
            comment_words += " ".join(tokens)+" "
        
            wordcloud = WordCloud(max_words=15156633, width = 800, height = 800,
                        background_color ='white',
                        stopwords = stopwords,
                        min_font_size = 3).generate(comment_words)
        
            # plot the WordCloud image                      
            plt.figure(figsize = (8, 8), facecolor = None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad = 0)
        
            plt.show()
            st.pyplot()

            for i in range(0,len(y)):
                if (y[i]== 5.):
                    y[i] = "Positive"
                elif (y[i]== 4.5):
                    y[i] = "Positive"
                elif (y[i]== 4.):
                    y[i] = "Positive"        
                elif (y[i] == 3.5):
                    y[i] = "Neutral"
                elif (y[i]== 3.):
                    y[i] = "Neutral" 
                else:
                    y[i] = "Negative"

            for i in range(0,len(y)):
                if (y[i] == 'Positive'):
                    y[i] = 2
                elif (y[i] == 'Neutral'):
                    y[i] = 1
                else:
                    y[i] = 0

            y = y.astype('int')


            import re
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem.porter import PorterStemmer
            ps = PorterStemmer()
            corpus = []
            for i in range(0, len(X)):
                review = re.sub('[^a-zA-Z]',' ', str(X['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus.append(review) 

            corpus_test = []
            for i in range(0, len(new_df)):
                review = re.sub('[^a-zA-Z]',' ', str(new_df['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus_test.append(review) 

            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features=9000) #After experimenting with 7500, 5000, 2500 ...9000 worked best.
            X = cv.fit_transform(corpus).toarray()

            from sklearn.naive_bayes import MultinomialNB
            clf = MultinomialNB()
            clf.fit(X, y)

            X_test = corpus_test
            #st.write(X_test)

            X_test = cv.transform(X_test)
            #X_test = X_test.toarray()

            y_pred = clf.predict(X_test)
            #st.write(y_pred)

            new_df['Classification Result'] = y_pred.tolist()

            st.markdown('**CSV file with metrics and classification result per review**')
            #new_df['Classification Result'] = new_df['Classification Result'].replace('2', 'Positive', inplace=True)
            
            new_df.loc[new_df['Classification Result'] == 2, 'Classification Result'] = 'Positive'
            new_df.loc[new_df['Classification Result'] == 0, 'Classification Result'] = 'Negative'
            new_df.loc[new_df['Classification Result'] == 1, 'Classification Result'] = 'Neutral'

            st.write(new_df)

            def convert_df(df):
                return df.to_csv().encode("utf-8")

            csv = convert_df(new_df)

            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="derived_dataframe.csv",
            mime="text/csv",
            )

            st.markdown('**Classification Results Summarization**')

            #new_df["Classification Result"].value_counts(dropna=True).plot(kind="pie")

            counts = new_df['Classification Result'].value_counts(dropna=True)
            counts.plot.pie(autopct='%1.1f%%', labels=['Positive', 'Negative', 'Neutral'], shadow=True)

            plt.title('Emotion Percentage of Reviews')

            plt.axis('equal')
            plt.show()
            st.pyplot()
        
        elif classifier_name == 'RandomForest':

            df=pd.read_csv('tripadvisor_hotel_reviews.csv')
            y = df["Rating"]
            X = df.iloc[:,:-1]
            y = pd.to_numeric(y)
            y = y.fillna(y.median())

            st.header('Exploratory Data Analysis')

            st.markdown('**Character Count: The count of characters for each review**')

            length = len(new_df['Review'][0])
            st.write(f'Length of a sample review: {length}')

            new_df['Length'] = new_df['Review'].str.len()
            st.write(new_df.head(10))

            new_df['Length'].plot(kind="line", title="Character Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Characters')
            st.pyplot()

            st.markdown('**Word Count: The cound of words for each review**')

            word_count = new_df['Review'][0].split()
            st.write(f'Word count in a sample review: {len(word_count)}')

            def word_count(review):
                review_list = review.split()
                return len(review_list)
            
            new_df['Word_count'] = new_df['Review'].apply(word_count)
            st.write(new_df.head(10))

            new_df['Word_count'].plot(kind="line", title="Word Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Words')
            st.pyplot()


            st.markdown('**Mean Word Length: average length of words per review**')
            new_df['mean_word_length'] = new_df['Review'].map(lambda rev: np.mean([len(word) for word in rev.split()]))
            st.write(new_df.head(10))

            new_df['mean_word_length'].plot(kind="line", title="Mean Word Length Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Mean Word Length')
            st.pyplot()

            #st.markdown('Mean sentence length: Average length of the sentences in the review')
            #st.write(np.mean([len(sent) for sent in tokenize.sent_tokenize(new_df['Review'][0])]))

            #new_df['mean_sent_length'] = new_df['Review'].map(lambda rev: np.mean([len(sent) for sent in tokenize.sent_tokenize(rev)]))
            #st.write(new_df.head(10))

            st.markdown('**Reviews Wordcloud**')
            #Constructing the wordcloud for all the data
            comment_words = ''
            stopwords = set(STOPWORDS)
        
        # iterate through the csv file
            for val in new_df.Review:
            
            # typecaste each val to string
                val = str(val)
        
            # split the value
                tokens = val.split()
            
            # Converts each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            
            comment_words += " ".join(tokens)+" "
        
            wordcloud = WordCloud(max_words=15156633, width = 800, height = 800,
                        background_color ='white',
                        stopwords = stopwords,
                        min_font_size = 3).generate(comment_words)
        
            # plot the WordCloud image                      
            plt.figure(figsize = (8, 8), facecolor = None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad = 0)
        
            plt.show()
            st.pyplot()

            for i in range(0,len(y)):
                if (y[i]== 5.):
                    y[i] = "Positive"
                elif (y[i]== 4.5):
                    y[i] = "Positive"
                elif (y[i]== 4.):
                    y[i] = "Positive"        
                elif (y[i] == 3.5):
                    y[i] = "Neutral"
                elif (y[i]== 3.):
                    y[i] = "Neutral" 
                else:
                    y[i] = "Negative"

            for i in range(0,len(y)):
                if (y[i] == 'Positive'):
                    y[i] = 2
                elif (y[i] == 'Neutral'):
                    y[i] = 1
                else:
                    y[i] = 0

            y = y.astype('int')


            import re
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem.porter import PorterStemmer
            ps = PorterStemmer()
            corpus = []
            for i in range(0, len(X)):
                review = re.sub('[^a-zA-Z]',' ', str(X['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus.append(review) 

            corpus_test = []
            for i in range(0, len(new_df)):
                review = re.sub('[^a-zA-Z]',' ', str(new_df['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus_test.append(review) 

            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features=9000) #After experimenting with 7500, 5000, 2500 ...9000 worked best.
            X = cv.fit_transform(corpus).toarray()

            from sklearn.naive_bayes import MultinomialNB
            clf = RandomForestClassifier()
            clf.fit(X, y)

            X_test = corpus_test
            #st.write(X_test)

            X_test = cv.transform(X_test)
            #X_test = X_test.toarray()

            y_pred = clf.predict(X_test)
            #st.write(y_pred)

            new_df['Classification Result'] = y_pred.tolist()

            st.markdown('**CSV file with metrics and classification result per review**')
            #new_df['Classification Result'] = new_df['Classification Result'].replace('2', 'Positive', inplace=True)
            
            new_df.loc[new_df['Classification Result'] == 2, 'Classification Result'] = 'Positive'
            new_df.loc[new_df['Classification Result'] == 0, 'Classification Result'] = 'Negative'
            new_df.loc[new_df['Classification Result'] == 1, 'Classification Result'] = 'Neutral'

            st.write(new_df)

            def convert_df(df):
                return df.to_csv().encode("utf-8")

            csv = convert_df(new_df)

            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="derived_dataframe.csv",
            mime="text/csv",
            )

            st.markdown('**Classification Results Summarization**')

            #new_df["Classification Result"].value_counts(dropna=True).plot(kind="pie")

            counts = new_df['Classification Result'].value_counts(dropna=True)
            counts.plot.pie(autopct='%1.1f%%', labels=['Positive', 'Negative', 'Neutral'], shadow=True)

            plt.title('Emotion Percentage of Reviews')

            plt.axis('equal')
            plt.show()
            st.pyplot()

        else:
            df=pd.read_csv('tripadvisor_hotel_reviews.csv')
            y = df["Rating"]
            X = df.iloc[:,:-1]
            y = pd.to_numeric(y)
            y = y.fillna(y.median())

            st.header('Exploratory Data Analysis')

            st.markdown('**Character Count: The count of characters for each review**')

            length = len(new_df['Review'][0])
            st.write(f'Length of a sample review: {length}')

            new_df['Length'] = new_df['Review'].str.len()
            st.write(new_df.head(10))

            new_df['Length'].plot(kind="line", title="Character Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Characters')
            st.pyplot()

            st.markdown('**Word Count: The cound of words for each review**')

            word_count = new_df['Review'][0].split()
            st.write(f'Word count in a sample review: {len(word_count)}')

            def word_count(review):
                review_list = review.split()
                return len(review_list)
            
            new_df['Word_count'] = new_df['Review'].apply(word_count)
            st.write(new_df.head(10))

            new_df['Word_count'].plot(kind="line", title="Word Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Words')
            st.pyplot()


            st.markdown('**Mean Word Length: average length of words per review**')
            new_df['mean_word_length'] = new_df['Review'].map(lambda rev: np.mean([len(word) for word in rev.split()]))
            st.write(new_df.head(10))

            new_df['mean_word_length'].plot(kind="line", title="Mean Word Length Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Mean Word Length')
            st.pyplot()

            #st.markdown('Mean sentence length: Average length of the sentences in the review')
            #st.write(np.mean([len(sent) for sent in tokenize.sent_tokenize(new_df['Review'][0])]))

            #new_df['mean_sent_length'] = new_df['Review'].map(lambda rev: np.mean([len(sent) for sent in tokenize.sent_tokenize(rev)]))
            #st.write(new_df.head(10))

            st.markdown('**Reviews Wordcloud**')
            #Constructing the wordcloud for all the data
            comment_words = ''
            stopwords = set(STOPWORDS)
        
        # iterate through the csv file
            for val in new_df.Review:
            
            # typecaste each val to string
                val = str(val)
        
            # split the value
                tokens = val.split()
            
            # Converts each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            
            comment_words += " ".join(tokens)+" "
        
            wordcloud = WordCloud(max_words=15156633, width = 800, height = 800,
                        background_color ='white',
                        stopwords = stopwords,
                        min_font_size = 3).generate(comment_words)
        
            # plot the WordCloud image                      
            plt.figure(figsize = (8, 8), facecolor = None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad = 0)
        
            plt.show()
            st.pyplot()

            for i in range(0,len(y)):
                if (y[i]== 5.):
                    y[i] = "Positive"
                elif (y[i]== 4.5):
                    y[i] = "Positive"
                elif (y[i]== 4.):
                    y[i] = "Positive"        
                elif (y[i] == 3.5):
                    y[i] = "Neutral"
                elif (y[i]== 3.):
                    y[i] = "Neutral" 
                else:
                    y[i] = "Negative"

            for i in range(0,len(y)):
                if (y[i] == 'Positive'):
                    y[i] = 2
                elif (y[i] == 'Neutral'):
                    y[i] = 1
                else:
                    y[i] = 0

            y = y.astype('int')


            import re
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem.porter import PorterStemmer
            ps = PorterStemmer()
            corpus = []
            for i in range(0, len(X)):
                review = re.sub('[^a-zA-Z]',' ', str(X['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus.append(review) 

            corpus_test = []
            for i in range(0, len(new_df)):
                review = re.sub('[^a-zA-Z]',' ', str(new_df['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus_test.append(review) 

            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features=9000) #After experimenting with 7500, 5000, 2500 ...9000 worked best.
            X = cv.fit_transform(corpus).toarray()

            from sklearn.naive_bayes import MultinomialNB
            clf = MLPClassifier()
            clf.fit(X, y)

            X_test = corpus_test
            #st.write(X_test)

            X_test = cv.transform(X_test)
            #X_test = X_test.toarray()

            y_pred = clf.predict(X_test)
            #st.write(y_pred)

            new_df['Classification Result'] = y_pred.tolist()

            st.markdown('**CSV file with metrics and classification result per review**')
            #new_df['Classification Result'] = new_df['Classification Result'].replace('2', 'Positive', inplace=True)
            
            new_df.loc[new_df['Classification Result'] == 2, 'Classification Result'] = 'Positive'
            new_df.loc[new_df['Classification Result'] == 0, 'Classification Result'] = 'Negative'
            new_df.loc[new_df['Classification Result'] == 1, 'Classification Result'] = 'Neutral'

            st.write(new_df)

            def convert_df(df):
                return df.to_csv().encode("utf-8")

            csv = convert_df(new_df)

            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="derived_dataframe.csv",
            mime="text/csv",
            )

            st.markdown('**Classification Results Summarization**')

            #new_df["Classification Result"].value_counts(dropna=True).plot(kind="pie")

            counts = new_df['Classification Result'].value_counts(dropna=True)
            counts.plot.pie(autopct='%1.1f%%', labels=['Positive', 'Negative', 'Neutral'], shadow=True)

            plt.title('Emotion Percentage of Reviews')

            plt.axis('equal')
            plt.show()
            st.pyplot()

    if value_name == 'Airlines':
        if classifier_name == 'MultinomialNB':
            df=pd.read_csv('Airline Passenger Reviews.csv')
            df = df.rename(columns={"customer_review": "Review", "NPS Score": "Rating"})

            y = df["Rating"]

            for i in range(0,len(y)):
                if (y[i] == 'Promoter'):
                    y[i] = 2
                elif (y[i] == 'Passive'):
                    y[i] = 1
                else:
                    y[i] = 0

            y = y.astype('int')

            y = df["Rating"]
            X = df.iloc[:,:-1]
            y = y.fillna(y.median())

            st.header('Exploratory Data Analysis')

            st.markdown('**Character Count: The count of characters for each review**')

            length = len(new_df['Review'][0])
            st.write(f'Length of a sample review: {length}')

            new_df['Length'] = new_df['Review'].str.len()
            st.write(new_df.head(10))

            new_df['Length'].plot(kind="line", title="Character Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Characters')
            st.pyplot()

            st.markdown('**Word Count: The cound of words for each review**')

            word_count = new_df['Review'][0].split()
            st.write(f'Word count in a sample review: {len(word_count)}')

            def word_count(review):
                review_list = review.split()
                return len(review_list)
            
            new_df['Word_count'] = new_df['Review'].apply(word_count)
            st.write(new_df.head(10))

            new_df['Word_count'].plot(kind="line", title="Word Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Words')
            st.pyplot()


            st.markdown('**Mean Word Length: average length of words per review**')
            new_df['mean_word_length'] = new_df['Review'].map(lambda rev: np.mean([len(word) for word in rev.split()]))
            st.write(new_df.head(10))

            new_df['mean_word_length'].plot(kind="line", title="Mean Word Length Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Mean Word Length')
            st.pyplot()

            #st.markdown('Mean sentence length: Average length of the sentences in the review')
            #st.write(np.mean([len(sent) for sent in tokenize.sent_tokenize(new_df['Review'][0])]))

            #new_df['mean_sent_length'] = new_df['Review'].map(lambda rev: np.mean([len(sent) for sent in tokenize.sent_tokenize(rev)]))
            #st.write(new_df.head(10))

            st.markdown('**Reviews Wordcloud**')
            #Constructing the wordcloud for all the data
            comment_words = ''
            stopwords = set(STOPWORDS)
        
        # iterate through the csv file
            for val in new_df.Review:
            
            # typecaste each val to string
                val = str(val)
        
            # split the value
                tokens = val.split()
            
            # Converts each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            
            comment_words += " ".join(tokens)+" "
        
            wordcloud = WordCloud(max_words=15156633, width = 800, height = 800,
                        background_color ='white',
                        stopwords = stopwords,
                        min_font_size = 3).generate(comment_words)
        
            # plot the WordCloud image                      
            plt.figure(figsize = (8, 8), facecolor = None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad = 0)
        
            plt.show()
            st.pyplot()


            import re
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem.porter import PorterStemmer
            ps = PorterStemmer()
            corpus = []
            for i in range(0, len(X)):
                review = re.sub('[^a-zA-Z]',' ', str(X['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus.append(review) 

            corpus_test = []
            for i in range(0, len(new_df)):
                review = re.sub('[^a-zA-Z]',' ', str(new_df['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus_test.append(review) 

            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features=9000) #After experimenting with 7500, 5000, 2500 ...9000 worked best.
            X = cv.fit_transform(corpus).toarray()

            from sklearn.naive_bayes import MultinomialNB
            clf = MultinomialNB()
            clf.fit(X, y)

            X_test = corpus_test
            #st.write(X_test)

            X_test = cv.transform(X_test)
            #X_test = X_test.toarray()

            y_pred = clf.predict(X_test)
            #st.write(y_pred)

            new_df['Classification Result'] = y_pred.tolist()

            st.markdown('**CSV file with metrics and classification result per review**')
            #new_df['Classification Result'] = new_df['Classification Result'].replace('2', 'Positive', inplace=True)
            
            new_df.loc[new_df['Classification Result'] == 2, 'Classification Result'] = 'Positive'
            new_df.loc[new_df['Classification Result'] == 0, 'Classification Result'] = 'Negative'
            new_df.loc[new_df['Classification Result'] == 1, 'Classification Result'] = 'Neutral'

            st.write(new_df)

            def convert_df(df):
                return df.to_csv().encode("utf-8")

            csv = convert_df(new_df)

            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="derived_dataframe.csv",
            mime="text/csv",
            )

            st.markdown('**Classification Results Summarization**')

            #new_df["Classification Result"].value_counts(dropna=True).plot(kind="pie")

            counts = new_df['Classification Result'].value_counts(dropna=True)
            counts.plot.pie(autopct='%1.1f%%', labels=['Positive', 'Negative', 'Neutral'], shadow=True)

            plt.title('Emotion Percentage of Reviews')

            plt.axis('equal')
            plt.show()
            st.pyplot()

        if classifier_name == 'RandomForest':

            df=pd.read_csv('Airline Passenger Reviews.csv')
            df = df.rename(columns={"customer_review": "Review", "NPS Score": "Rating"})

            y = df["Rating"]

            for i in range(0,len(y)):
                if (y[i] == 'Promoter'):
                    y[i] = 2
                elif (y[i] == 'Passive'):
                    y[i] = 1
                else:
                    y[i] = 0

            y = y.astype('int')

            y = df["Rating"]
            X = df.iloc[:,:-1]
            y = y.fillna(y.median())

            st.header('Exploratory Data Analysis')

            st.markdown('**Character Count: The count of characters for each review**')

            length = len(new_df['Review'][0])
            st.write(f'Length of a sample review: {length}')

            new_df['Length'] = new_df['Review'].str.len()
            st.write(new_df.head(10))

            new_df['Length'].plot(kind="line", title="Character Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Characters')
            st.pyplot()

            st.markdown('**Word Count: The cound of words for each review**')

            word_count = new_df['Review'][0].split()
            st.write(f'Word count in a sample review: {len(word_count)}')

            def word_count(review):
                review_list = review.split()
                return len(review_list)
            
            new_df['Word_count'] = new_df['Review'].apply(word_count)
            st.write(new_df.head(10))

            new_df['Word_count'].plot(kind="line", title="Word Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Words')
            st.pyplot()


            st.markdown('**Mean Word Length: average length of words per review**')
            new_df['mean_word_length'] = new_df['Review'].map(lambda rev: np.mean([len(word) for word in rev.split()]))
            st.write(new_df.head(10))

            new_df['mean_word_length'].plot(kind="line", title="Mean Word Length Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Mean Word Length')
            st.pyplot()

            #st.markdown('Mean sentence length: Average length of the sentences in the review')
            #st.write(np.mean([len(sent) for sent in tokenize.sent_tokenize(new_df['Review'][0])]))

            #new_df['mean_sent_length'] = new_df['Review'].map(lambda rev: np.mean([len(sent) for sent in tokenize.sent_tokenize(rev)]))
            #st.write(new_df.head(10))

            st.markdown('**Reviews Wordcloud**')
            #Constructing the wordcloud for all the data
            comment_words = ''
            stopwords = set(STOPWORDS)
        
        # iterate through the csv file
            for val in new_df.Review:
            
            # typecaste each val to string
                val = str(val)
        
            # split the value
                tokens = val.split()
            
            # Converts each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            
            comment_words += " ".join(tokens)+" "
        
            wordcloud = WordCloud(max_words=15156633, width = 800, height = 800,
                        background_color ='white',
                        stopwords = stopwords,
                        min_font_size = 3).generate(comment_words)
        
            # plot the WordCloud image                      
            plt.figure(figsize = (8, 8), facecolor = None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad = 0)
        
            plt.show()
            st.pyplot()


            import re
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem.porter import PorterStemmer
            ps = PorterStemmer()
            corpus = []
            for i in range(0, len(X)):
                review = re.sub('[^a-zA-Z]',' ', str(X['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus.append(review) 

            corpus_test = []
            for i in range(0, len(new_df)):
                review = re.sub('[^a-zA-Z]',' ', str(new_df['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus_test.append(review) 

            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features=9000) #After experimenting with 7500, 5000, 2500 ...9000 worked best.
            X = cv.fit_transform(corpus).toarray()

            from sklearn.naive_bayes import MultinomialNB
            clf = RandomForestClassifier()
            clf.fit(X, y)

            X_test = corpus_test
            #st.write(X_test)

            X_test = cv.transform(X_test)
            #X_test = X_test.toarray()

            y_pred = clf.predict(X_test)
            #st.write(y_pred)

            new_df['Classification Result'] = y_pred.tolist()

            st.markdown('**CSV file with metrics and classification result per review**')
            #new_df['Classification Result'] = new_df['Classification Result'].replace('2', 'Positive', inplace=True)
            
            new_df.loc[new_df['Classification Result'] == 2, 'Classification Result'] = 'Positive'
            new_df.loc[new_df['Classification Result'] == 0, 'Classification Result'] = 'Negative'
            new_df.loc[new_df['Classification Result'] == 1, 'Classification Result'] = 'Neutral'

            st.write(new_df)

            def convert_df(df):
                return df.to_csv().encode("utf-8")

            csv = convert_df(new_df)

            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="derived_dataframe.csv",
            mime="text/csv",
            )

            st.markdown('**Classification Results Summarization**')

            #new_df["Classification Result"].value_counts(dropna=True).plot(kind="pie")

            counts = new_df['Classification Result'].value_counts(dropna=True)
            counts.plot.pie(autopct='%1.1f%%', labels=['Positive', 'Negative', 'Neutral'], shadow=True)

            plt.title('Emotion Percentage of Reviews')

            plt.axis('equal')
            plt.show()
            st.pyplot()

        else:

            df=pd.read_csv('Airline Passenger Reviews.csv')
            df = df.rename(columns={"customer_review": "Review", "NPS Score": "Rating"})

            y = df["Rating"]

            for i in range(0,len(y)):
                if (y[i] == 'Promoter'):
                    y[i] = 2
                elif (y[i] == 'Passive'):
                    y[i] = 1
                else:
                    y[i] = 0

            y = y.astype('int')

            y = df["Rating"]
            X = df.iloc[:,:-1]
            y = y.fillna(y.median())

            st.header('Exploratory Data Analysis')

            st.markdown('**Character Count: The count of characters for each review**')

            length = len(new_df['Review'][0])
            st.write(f'Length of a sample review: {length}')

            new_df['Length'] = new_df['Review'].str.len()
            st.write(new_df.head(10))

            new_df['Length'].plot(kind="line", title="Character Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Characters')
            st.pyplot()

            st.markdown('**Word Count: The cound of words for each review**')

            word_count = new_df['Review'][0].split()
            st.write(f'Word count in a sample review: {len(word_count)}')

            def word_count(review):
                review_list = review.split()
                return len(review_list)
            
            new_df['Word_count'] = new_df['Review'].apply(word_count)
            st.write(new_df.head(10))

            new_df['Word_count'].plot(kind="line", title="Word Count Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Number of Words')
            st.pyplot()


            st.markdown('**Mean Word Length: average length of words per review**')
            new_df['mean_word_length'] = new_df['Review'].map(lambda rev: np.mean([len(word) for word in rev.split()]))
            st.write(new_df.head(10))

            new_df['mean_word_length'].plot(kind="line", title="Mean Word Length Distribution Line Plot", xlabel = 'Review Number', ylabel = 'Mean Word Length')
            st.pyplot()

            #st.markdown('Mean sentence length: Average length of the sentences in the review')
            #st.write(np.mean([len(sent) for sent in tokenize.sent_tokenize(new_df['Review'][0])]))

            #new_df['mean_sent_length'] = new_df['Review'].map(lambda rev: np.mean([len(sent) for sent in tokenize.sent_tokenize(rev)]))
            #st.write(new_df.head(10))

            st.markdown('**Reviews Wordcloud**')
            #Constructing the wordcloud for all the data
            comment_words = ''
            stopwords = set(STOPWORDS)
        
        # iterate through the csv file
            for val in new_df.Review:
            
            # typecaste each val to string
                val = str(val)
        
            # split the value
                tokens = val.split()
            
            # Converts each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            
            comment_words += " ".join(tokens)+" "
        
            wordcloud = WordCloud(max_words=15156633, width = 800, height = 800,
                        background_color ='white',
                        stopwords = stopwords,
                        min_font_size = 3).generate(comment_words)
        
            # plot the WordCloud image                      
            plt.figure(figsize = (8, 8), facecolor = None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad = 0)
        
            plt.show()
            st.pyplot()


            import re
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem.porter import PorterStemmer
            ps = PorterStemmer()
            corpus = []
            for i in range(0, len(X)):
                review = re.sub('[^a-zA-Z]',' ', str(X['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus.append(review) 

            corpus_test = []
            for i in range(0, len(new_df)):
                review = re.sub('[^a-zA-Z]',' ', str(new_df['Review'][i]))
                review = review.lower() #Lowering the words is very imporatant in avoiding classifying same words as different words
                review = review.split()
            
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #Eleminating words that do not put much value in sentences.
                review = ' '.join(review) #Reconstructing sentences
                corpus_test.append(review) 

            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features=9000) #After experimenting with 7500, 5000, 2500 ...9000 worked best.
            X = cv.fit_transform(corpus).toarray()

            from sklearn.naive_bayes import MultinomialNB
            clf = MLPClassifier()
            clf.fit(X, y)

            X_test = corpus_test
            #st.write(X_test)

            X_test = cv.transform(X_test)
            #X_test = X_test.toarray()

            y_pred = clf.predict(X_test)
            #st.write(y_pred)

            new_df['Classification Result'] = y_pred.tolist()

            st.markdown('**CSV file with metrics and classification result per review**')
            #new_df['Classification Result'] = new_df['Classification Result'].replace('2', 'Positive', inplace=True)
            
            new_df.loc[new_df['Classification Result'] == 2, 'Classification Result'] = 'Positive'
            new_df.loc[new_df['Classification Result'] == 0, 'Classification Result'] = 'Negative'
            new_df.loc[new_df['Classification Result'] == 1, 'Classification Result'] = 'Neutral'

            st.write(new_df)

            def convert_df(df):
                return df.to_csv().encode("utf-8")

            csv = convert_df(new_df)

            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="derived_dataframe.csv",
            mime="text/csv",
            )

            st.markdown('**Classification Results Summarization**')

            #new_df["Classification Result"].value_counts(dropna=True).plot(kind="pie")

            counts = new_df['Classification Result'].value_counts(dropna=True)
            counts.plot.pie(autopct='%1.1f%%', labels=['Positive', 'Negative', 'Neutral'], shadow=True)

            plt.title('Emotion Percentage of Reviews')

            plt.axis('equal')
            plt.show()
            st.pyplot()













