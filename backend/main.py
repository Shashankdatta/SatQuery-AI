import os
from app.main import app

# Entrypoint for Vercel Serverless / ASGI runners
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
