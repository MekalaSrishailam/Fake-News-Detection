def predict_news(news_text, tfidf_vectorizer, model):
    """
    Predicts whether the input news is fake or real.
    
    Args:
        news_text (str): Raw news content.
        tfidf_vectorizer: Fitted TF-IDF vectorizer.
        model: Trained ML model (MLPClassifier).
    
    Returns:
        str: Prediction result ("Fake" or "Real")
    """
    import string
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Preprocess the text
    text = news_text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    processed_text = ' '.join(words)

    # Vectorize and predict
    vector = tfidf_vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(vector)[0]
    return "Real" if prediction == 1 else "Fake"
