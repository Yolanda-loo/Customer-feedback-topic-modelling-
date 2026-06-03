import pandas as pd
import nltk
import re
import joblib
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from wordcloud import WordCloud

# ----------------------------
# Download NLTK resources
# ----------------------------
nltk.download('stopwords')
nltk.download('wordnet')

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv('../data/customer_feedback.csv')

# ----------------------------
# Text Preprocessing
# ----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]

    return ' '.join(words)

df['clean_text'] = df['FeedbackText'].apply(clean_text)

# ----------------------------
# Vectorization
# ----------------------------
vectorizer = CountVectorizer(max_df=0.9, min_df=2)
dtm = vectorizer.fit_transform(df['clean_text'])

# ----------------------------
# LDA Model
# ----------------------------
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(dtm)

# Save model
joblib.dump(lda, '../models/feedback_topics.pkl')

# ----------------------------
# Print topics
# ----------------------------
print("\n📌 Topics:")
for i, topic in enumerate(lda.components_):
    words = [vectorizer.get_feature_names_out()[i]
             for i in topic.argsort()[-5:]]
    print(f"Topic {i+1}: {words}")

# ----------------------------
# Topic Distribution
# ----------------------------
topic_results = lda.transform(dtm)
df['Topic'] = topic_results.argmax(axis=1)

topic_counts = df['Topic'].value_counts().sort_index()

plt.figure()
topic_counts.plot(kind='bar')
plt.title("Topic Distribution")
plt.xlabel("Topic")
plt.ylabel("Count")
plt.savefig('../visuals/topic_distribution.png')

# ----------------------------
# Word Clouds per Topic
# ----------------------------
feature_names = vectorizer.get_feature_names_out()

for i, topic in enumerate(lda.components_):
    word_dict = {feature_names[j]: topic[j] for j in range(len(feature_names))}

    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(word_dict)

    plt.figure()
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Topic {i+1}")
    plt.savefig(f'../visuals/topic_{i+1}_wordcloud.png')

print("✅ Topic modeling complete!")
``
