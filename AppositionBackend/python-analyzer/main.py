from fastapi import FastAPI
import json
from pathlib import Path

from similarity_engine import calculate_similarity


app = FastAPI()

BASE_DIR = Path(__file__).parent


@app.post("/similarity")
def similarity(request: dict):

    print("\nReceived request from C#.")

    # Save the incoming request for debugging.
    competitors_file = BASE_DIR / "competitors.json"

    with open(competitors_file, "w") as file:
        json.dump(request, file, indent=4)

    print(
        "Competitor data saved to:",
        competitors_file
    )

    # Run the Sentence Transformer
    # + cosine similarity analysis.
    results = calculate_similarity(request)

    response = {
        "results": results
    }

    # Save the results for debugging.
    results_file = BASE_DIR / "results.json"

    with open(results_file, "w") as file:
        json.dump(response, file, indent=4)

    print("\nTop competitors:")

    for result in results:

        competitor = result["competitor"]

        print(
            f"{competitor['name']}: "
            f"{result['similarityScore']}"
        )

    print(
        "\nResults saved to:",
        results_file
    )

    return response