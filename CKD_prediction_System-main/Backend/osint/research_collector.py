import requests

def search_ckd_papers():
    query = "chronic kidney disease machine learning"
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        papers = data.get("data", [])

        results = []

        for paper in papers:
            results.append({
                "title": paper.get("title"),
                "year": paper.get("year"),
                "authors": [a["name"] for a in paper.get("authors", [])]
            })

        return results

    return []
