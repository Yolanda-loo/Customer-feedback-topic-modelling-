import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download NLTK resources (only needed once)
nltk.download("stopwords")
nltk.download("wordnet")

# Step 1: Load dataset
# Example columns: CustomerID, FeedbackText
df = pd.read_csv("data/customer_feedback.csv")

# Step 2: Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

df["CleanedFeedback"] = df["FeedbackText"].apply(preprocess)

# Step 3: Vectorization
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
X = vectorizer.fit_transform(df["CleanedFeedback"])

# Step 4: Topic Modeling (LDA)
lda = LatentDirichletAllocation(n_components=3, random_state=42)  # 3 topics
lda.fit(X)

# Step 5: Display topics
def display_topics(model, feature_names, no_top_words):
    for idx, topic in enumerate(model.components_):
        print(f"Topic {idx+1}:")
        print([feature_names[i] for i in topic.argsort()[-no_top_words:]])

display_topics(lda, vectorizer.get_feature_names_out(), 10)

# Step 6: Word Clouds for each topic
for idx, topic in enumerate(lda.components_):
    word_freq = {vectorizer.get_feature_names_out()[i]: topic[i] for i in topic.argsort()[-20:]}
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
    plt.figure(figsize=(8,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Topic {idx+1} Word Cloud")
    plt.show()
