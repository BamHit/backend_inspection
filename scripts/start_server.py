"""Script pour démarrer le serveur FastAPI."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "industrial_inspection.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Désactiver en production
    )
