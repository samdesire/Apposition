from sentence_transformers import SentenceTransformer, util


# Load the model once when the Python server starts.
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_competitor_apps(competitors):
    """
    Create an embedding for every competitor app.
    """

    if not competitors:
        return competitors

    texts = []

    for competitor in competitors:

        text = f"""
        {competitor["name"]}
        {competitor["description"]}
        {competitor["genre"]}
        """

        texts.append(text)

    embeddings = model.encode(
        texts,
        convert_to_tensor=True
    )

    for competitor, embedding in zip(
        competitors,
        embeddings
    ):
        competitor["_embedding"] = embedding

    return competitors


def cosine_similarity_score(
    app_idea,
    key_features,
    target_audience,
    competitors
):
    """
    Calculate cosine similarity between the user's
    app and each competitor.
    """

    if not competitors:
        return []

    user_text = f"""
    {app_idea}
    {key_features}
    {target_audience}
    """

    user_embedding = model.encode(
        user_text,
        convert_to_tensor=True
    )

    results = []

    for competitor in competitors:

        score = util.cos_sim(
            user_embedding,
            competitor["_embedding"]
        ).item()

        clean_competitor = {
            key: value
            for key, value in competitor.items()
            if key != "_embedding"
        }

        results.append({
            "competitor": clean_competitor,
            "similarityScore": round(score, 4)
        })

    results.sort(
        key=lambda result: result["similarityScore"],
        reverse=True
    )

    return results[:10]


def calculate_similarity(request):
    """
    Main similarity-analysis function.
    """

    competitors = request["competitors"]

    if not competitors:
        return []

    competitors = embed_competitor_apps(
        competitors
    )

    return cosine_similarity_score(
        request["appIdea"],
        request["keyFeatures"],
        request["targetAudience"],
        competitors
    )